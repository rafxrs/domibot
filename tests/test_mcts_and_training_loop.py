import random
from collections import Counter

import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS
from domibot.models import END_ACTIONS
from training.agents import DomibotAgent
from training.evaluate import play_game
from training.heuristics import heuristic_reaction
from training.mcts import (
    materialize,
    merge_ensemble_roots,
    redeal_hidden_info,
    run_mcts,
    run_mcts_batch,
    run_mcts_ensemble,
    select_action,
    terminal_value,
    visit_distribution,
)
from training.network import DomibotNet
from training.self_play import (
    Example,
    ReplayBuffer,
    SUB_DECISION_CARDS,
    _backfill_value_targets,
    _sample_kingdom,
    play_cross_play_games,
    play_self_play_game,
    play_self_play_games_batch,
)
from training.train import train_step


def _tiny_kingdom():
    return list(KINGDOM_CARDS)[:10]


def test_clone_independence_and_mid_resolution_guard():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    clone = game.clone()
    clone.players[0].hand.append("MUTATED")
    assert "MUTATED" not in game.players[0].hand
    assert clone.rng is not game.rng

    game.players[0].hand = ["Cellar", "Copper", "Copper"]
    game.step(__import__("domibot").Action("PLAY", "Cellar"))
    assert game.pending_gen is not None
    try:
        game.clone()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_terminal_value_is_margin_based():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    for p in game.players:
        p.hand, p.discard, p.play_area, p.set_aside = [], [], [], []

    game.players[0].deck = ["Province"] * 3  # 18 VP
    game.players[1].deck = ["Province"] * 1  # 6 VP
    small_margin = terminal_value(game, 0)
    assert 0 < small_margin < 1.0
    assert terminal_value(game, 1) == -small_margin  # zero-sum in a 2p game

    game.players[0].deck = ["Province"] * 6  # 36 VP -- a bigger blowout
    big_margin = terminal_value(game, 0)
    assert big_margin > small_margin  # a bigger win scores higher, not just "+1" either way

    game.players[0].deck = ["Province"]
    game.players[1].deck = ["Province"]
    assert terminal_value(game, 0) == 0.0  # exact tie


def test_action_bias_shifts_prior_toward_playing_actions():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    game.players[0].hand = ["Village", "Copper", "Copper", "Copper", "Copper"]
    net = DomibotNet()
    net.eval()

    root_unbiased = run_mcts(game, net, num_simulations=1, action_bias=0.0)
    root_biased = run_mcts(game, net, num_simulations=1, action_bias=0.5)

    play_village = Action("PLAY", "Village")
    assert root_biased.P[play_village] > root_unbiased.P[play_village]
    assert root_biased.P[END_ACTIONS] < root_unbiased.P[END_ACTIONS]
    assert abs(sum(root_biased.P.values()) - 1.0) < 1e-5  # still a valid distribution
    assert game.legal_actions() == [Action("PLAY", "Village"), END_ACTIONS]  # root_game untouched


def test_mcts_only_visits_legal_actions_and_conserves_visit_count():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    game.step(END_ACTIONS)  # into BUY phase, several legal actions
    net = DomibotNet()
    net.eval()
    root = run_mcts(game, net, num_simulations=30, add_noise=True)

    legal = set(game.legal_actions())
    assert set(root.N.keys()) == legal
    assert sum(root.N.values()) == 30
    dist = visit_distribution(root)
    assert abs(sum(dist.values()) - 1.0) < 1e-6

    # root_game itself must be untouched by the search
    assert game.legal_actions() == list(root.game.legal_actions())
    action = select_action(root, temperature=0.0)
    assert action in legal


def test_run_mcts_batch_matches_single_game_semantics():
    games = [Game(_tiny_kingdom(), num_players=2, seed=s) for s in (10, 11, 12)]
    net = DomibotNet()
    net.eval()

    roots = run_mcts_batch(games, net, num_simulations=20, add_noise=True,
                            rngs=[np.random.default_rng(s) for s in (10, 11, 12)])

    assert len(roots) == 3
    for game, root in zip(games, roots):
        legal = set(game.legal_actions())
        assert set(root.N.keys()) == legal
        assert sum(root.N.values()) == 20
        dist = visit_distribution(root)
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        # root_game itself must be untouched by the batched search
        assert game.legal_actions() == list(root.game.legal_actions())


def test_run_mcts_batch_of_one_is_consistent_with_run_mcts():
    # not bit-identical (different RNG draws inside each implementation's
    # own loop structure), but should explore the same legal-action space
    # and conserve visit counts identically.
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    game.step(END_ACTIONS)
    net = DomibotNet()
    net.eval()

    root_single = run_mcts(game, net, num_simulations=15, add_noise=True, rng=np.random.default_rng(0))
    root_batch = run_mcts_batch(
        [game], net, num_simulations=15, add_noise=True, rngs=[np.random.default_rng(0)]
    )[0]

    assert set(root_single.N.keys()) == set(root_batch.N.keys())
    assert sum(root_single.N.values()) == sum(root_batch.N.values()) == 15


def test_self_play_games_batch_produces_consistent_examples_per_game():
    net = DomibotNet()
    net.eval()
    games_examples = play_self_play_games_batch(net, num_games=3, num_simulations=8, kingdom=_tiny_kingdom(), seed=3)

    assert len(games_examples) == 3
    for examples in games_examples:
        assert len(examples) > 0
        for ex in examples:
            assert isinstance(ex, Example)
            assert ex.obs.shape == (net.obs_dim,)
            assert ex.mask.shape == (net.num_actions,)
            assert abs(ex.policy_target.sum() - 1.0) < 1e-5
            assert np.all(ex.policy_target[~ex.mask] == 0.0)
            assert -1.0 <= ex.value_target <= 1.0


def test_self_play_game_produces_consistent_examples():
    net = DomibotNet()
    net.eval()
    examples = play_self_play_game(net, num_simulations=8, kingdom=_tiny_kingdom(), seed=3)
    assert len(examples) > 0
    for ex in examples:
        assert isinstance(ex, Example)
        assert ex.obs.shape == (net.obs_dim,)
        assert ex.mask.shape == (net.num_actions,)
        assert ex.mask.dtype == bool
        assert abs(ex.policy_target.sum() - 1.0) < 1e-5
        assert np.all(ex.policy_target[~ex.mask] == 0.0)
        assert -1.0 <= ex.value_target <= 1.0  # margin-based now, not just -1/0/1


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=5)
    net = DomibotNet()
    net.eval()
    for seed in range(3):
        buf.add_game(play_self_play_game(net, num_simulations=5, kingdom=_tiny_kingdom(), seed=seed))
    assert len(buf) <= 5
    assert len(buf.examples) == len(buf)


def test_train_step_updates_weights_and_reduces_joint_loss_on_repeat():
    torch.manual_seed(0)  # DomibotNet()'s init is otherwise unseeded, making this flaky
    net = DomibotNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.eval()
    examples = play_self_play_game(net, num_simulations=8, kingdom=_tiny_kingdom(), seed=4)
    net.train()
    batch = examples[: min(4, len(examples))]

    before = [p.clone() for p in net.parameters()]
    losses = []
    for _ in range(40):
        pl, vl = train_step(net, optimizer, batch, torch.device("cpu"))
        losses.append(pl + vl)
    after = list(net.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    # a handful of steps on a single small fixed batch is noisy; check the
    # trend (early vs. late average), not a strict step-to-step decrease
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early


def test_domibot_agent_plays_without_crashing():
    # temperature=1.0 (some randomness) rather than greedy: an untrained
    # network paired with argmax selection can otherwise deterministically
    # stall forever preferring END_BUY over any purchase, since nothing
    # forces a buy -- see evaluate.MAX_STEPS's docstring. What this test
    # actually checks is that DomibotAgent never produces an illegal action
    # (play_game would raise via Game.step's legality check if it did),
    # not that an undertrained agent necessarily reaches a natural game-over.
    net = DomibotNet()
    net.eval()
    agent = DomibotAgent(net, num_simulations=8, temperature=1.0)
    game = play_game(agent, agent, _tiny_kingdom(), seed=5)
    assert set(game.winners()) <= set(game.get_scores())


def test_materialize_matches_direct_stepping():
    # Force a reshuffle mid-effect (deck has only 1 card, Sentry needs to
    # peek 2) -- the trickiest case for the determinism claim in mcts.py's
    # module docstring: replaying the same actions against a fresh clone of
    # the boundary must reach bit-identical state, RNG included, to just
    # stepping the original game directly through the same actions.
    kingdom = ["Sentry", "Village", "Moat", "Smithy", "Militia",
               "Workshop", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=7)
    game.players[0].hand = ["Sentry", "Copper", "Copper"]
    game.players[0].deck = ["Silver"]
    game.players[0].discard = ["Gold", "Estate", "Copper"]
    boundary = game.clone()  # game.pending_gen is None here -- a fresh boundary

    path = [Action("PLAY", "Sentry")]
    game.step(path[0])
    assert game.pending_decision is not None  # Sentry always yields at least once
    while game.pending_decision is not None:
        opts = game.legal_actions()
        action = next((a for a in opts if a.card is None), opts[0])  # DONE/NONE if offered, else first
        game.step(action)
        path.append(action)

    reconstructed = materialize(boundary, path)

    for p_orig, p_recon in zip(game.players, reconstructed.players):
        assert p_orig.hand == p_recon.hand
        assert p_orig.deck == p_recon.deck
        assert p_orig.discard == p_recon.discard
        assert p_orig.play_area == p_recon.play_area
    assert game.trash == reconstructed.trash
    assert game.supply == reconstructed.supply
    assert game.rng.getstate() == reconstructed.rng.getstate()


def test_sub_decision_root_search_covers_legal_options():
    kingdom = ["Chapel", "Village", "Moat", "Smithy", "Militia",
               "Workshop", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.players[0].hand = ["Chapel", "Gold", "Silver", "Estate", "Copper"]
    net = DomibotNet()
    net.eval()

    path = [Action("PLAY", "Chapel")]
    root = run_mcts(game, net, num_simulations=30, path=path)

    legal = set(materialize(game, path).legal_actions())
    assert set(root.N.keys()) == legal
    assert sum(root.N.values()) == 30
    dist = visit_distribution(root)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert game.pending_decision is None  # root_game itself untouched


def test_run_mcts_sub_decision_decider_is_whoever_the_effect_forces():
    # Militia's forced discard belongs to the *opponent*, not whoever's
    # turn it structurally is -- this is the property that makes
    # self_play.py's `Example.decider = root.decider` correct for a
    # sub-decision node, not just for phase-action nodes.
    kingdom = ["Militia", "Village", "Moat", "Smithy", "Workshop",
               "Chapel", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.players[0].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    game.players[1].hand = ["Estate", "Copper", "Copper", "Smithy", "Village"]
    net = DomibotNet()
    net.eval()

    root = run_mcts(game, net, num_simulations=6, path=[Action("PLAY", "Militia")])
    assert root.decider == 1


def test_self_play_does_not_crash_with_an_attack_card_in_the_kingdom():
    # A kingdom likely to actually exercise sub-decision search during a
    # real self-play game (Militia's forced discard, Chapel's optional
    # trash), not just the forced/scripted scenarios above.
    kingdom = ["Militia", "Chapel", "Village", "Moat", "Smithy",
               "Workshop", "Bandit", "Council Room", "Festival", "Library"]
    net = DomibotNet()
    net.eval()
    examples = play_self_play_game(net, num_simulations=4, kingdom=kingdom, seed=11, max_moves=12)
    assert len(examples) > 0
    for ex in examples:
        assert isinstance(ex, Example)
        assert abs(ex.policy_target.sum() - 1.0) < 1e-5
        assert np.all(ex.policy_target[~ex.mask] == 0.0)


def test_domibot_agent_searches_sub_decisions_when_enabled():
    kingdom = ["Militia", "Village", "Moat", "Smithy", "Workshop",
               "Chapel", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.players[0].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    game.players[1].hand = ["Estate", "Copper", "Copper", "Smithy", "Village"]

    net = DomibotNet()
    net.eval()
    agents = [
        DomibotAgent(net, num_simulations=4, temperature=1.0, search_sub_decisions=True),
        DomibotAgent(net, num_simulations=4, temperature=1.0, search_sub_decisions=True),
    ]
    # Prime both agents' boundary caches on the pre-Militia state (a normal
    # phase-action call for each -- harmless, doesn't mutate `game`), then
    # force the actual Militia play so player 1's forced discard is the
    # *first* sub-decision either agent ever has to resolve.
    for agent in agents:
        agent.act(game)
    game.step(Action("PLAY", "Militia"))
    assert game.pending_decision is not None and game.current_decider() == 1

    for _ in range(20):
        if game.is_game_over():
            break
        game.step(agents[game.current_decider()].act(game))  # Game.step raises on an illegal action


def test_domibot_agent_ablation_matches_heuristic():
    kingdom = ["Chapel", "Village", "Moat", "Smithy", "Militia",
               "Workshop", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.players[0].hand = ["Chapel", "Gold", "Silver", "Estate", "Copper"]
    game.step(Action("PLAY", "Chapel"))
    assert game.pending_decision is not None

    net = DomibotNet()
    net.eval()
    agent = DomibotAgent(net, num_simulations=4, search_sub_decisions=False)
    assert agent.act(game) == heuristic_reaction(game)


def test_sample_kingdom_default_matches_plain_random_sample():
    # min_sub_decision_cards=0 (the default for every existing caller) must
    # be byte-for-byte identical to plain rng.sample -- no behavior change
    # for anything that doesn't opt in.
    import random as random_module
    rng_a, rng_b = random_module.Random(7), random_module.Random(7)
    assert _sample_kingdom(rng_a) == rng_b.sample(list(KINGDOM_CARDS), 10)


def test_sample_kingdom_curriculum_guarantees_minimum_density():
    import random as random_module
    rng = random_module.Random(3)
    for _ in range(20):
        kingdom = _sample_kingdom(rng, min_sub_decision_cards=6)
        assert len(kingdom) == 10
        assert len(set(kingdom)) == 10  # no duplicates
        assert sum(1 for c in kingdom if c in SUB_DECISION_CARDS) >= 6


# --- determinization ensembles (redeal_hidden_info / run_mcts_ensemble) ---

def test_redeal_hidden_info_preserves_from_players_own_hand_and_deck():
    import random as random_module
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    game.players[0].hand = ["Copper", "Copper", "Estate"]
    game.players[0].deck = ["Silver", "Gold"]
    original_hand, original_deck = list(game.players[0].hand), list(game.players[0].deck)

    redealt = redeal_hidden_info(game, from_player=0, rng=random_module.Random(1))

    assert redealt.players[0].hand == original_hand
    assert redealt.players[0].deck == original_deck
    assert game.players[0].hand == original_hand  # original untouched by the redeal


def test_redeal_hidden_info_preserves_sizes_total_ownership_and_public_zones():
    import random as random_module
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    p1 = game.players[1]
    p1.hand = ["Copper", "Copper", "Estate", "Estate", "Silver"]
    p1.deck = ["Copper", "Copper", "Estate"]
    p1.discard = ["Gold"]
    p1.play_area = ["Village"]
    p1.set_aside = ["Moat"]
    original_total = Counter(p1.all_cards())

    redealt = redeal_hidden_info(game, from_player=0, rng=random_module.Random(2))
    rp1 = redealt.players[1]

    assert len(rp1.hand) == len(p1.hand)
    assert len(rp1.deck) == len(p1.deck)
    assert Counter(rp1.all_cards()) == original_total
    assert rp1.discard == p1.discard
    assert rp1.play_area == p1.play_area
    assert rp1.set_aside == p1.set_aside


def test_redeal_hidden_info_reshuffles_other_players_hidden_contents():
    import random as random_module
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    p1 = game.players[1]
    p1.hand = ["Copper", "Estate", "Silver", "Gold", "Village"]
    p1.deck = ["Copper", "Estate", "Smithy", "Market", "Workshop"]

    redealt_hands = [
        tuple(sorted(redeal_hidden_info(game, from_player=0, rng=random_module.Random(seed)).players[1].hand))
        for seed in range(10)
    ]
    # 10 distinct cards split 5/5 -- different seeds should produce at
    # least *some* different hands (not asserting every pair differs, a
    # coincidence is possible, just that it's not always identical).
    assert len(set(redealt_hands)) > 1


def test_redeal_hidden_info_decorrelates_rng_across_redeals():
    import random as random_module
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    driver = random_module.Random(5)
    a = redeal_hidden_info(game, from_player=0, rng=driver)
    b = redeal_hidden_info(game, from_player=0, rng=driver)
    # Game.clone() copies the exact rng state; without reseeding, two
    # redeals of the same boundary would see identical future chance events.
    assert a.rng.getstate() != b.rng.getstate()


def test_run_mcts_ensemble_size_one_matches_run_mcts():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    net = DomibotNet()
    net.eval()
    root = run_mcts_ensemble(game, net, num_simulations=10, ensemble_size=1)
    assert sum(root.N.values()) == 10
    assert set(root.N.keys()) == set(game.legal_actions())
    assert root.decider == game.current_decider()


def test_run_mcts_ensemble_merges_visit_counts_across_members():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    net = DomibotNet()
    net.eval()
    root = run_mcts_ensemble(game, net, num_simulations=20, ensemble_size=4)
    assert sum(root.N.values()) == 20  # 5 sims/member x 4 members
    assert set(root.N.keys()) == set(game.legal_actions())


def test_run_mcts_ensemble_root_legal_actions_match_true_game():
    game = Game(_tiny_kingdom(), num_players=2, seed=9)
    net = DomibotNet()
    net.eval()
    root = run_mcts_ensemble(game, net, num_simulations=6, ensemble_size=3)
    # The invariant self_play.py's merge relies on: redeal never touches the
    # deciding player's own hand, so every ensemble member -- and therefore
    # the merged root -- has exactly the true game's legal actions.
    assert set(root.legal_actions) == set(game.legal_actions())


def test_play_self_play_game_with_ensemble_produces_valid_examples():
    net = DomibotNet()
    net.eval()
    examples = play_self_play_game(
        net, num_simulations=8, kingdom=_tiny_kingdom(), seed=3, determinization_ensemble_size=4,
    )
    assert len(examples) > 0
    for ex in examples:
        assert ex.obs.shape == (net.obs_dim,)
        assert ex.mask.shape == (net.num_actions,)
        assert abs(ex.policy_target.sum() - 1.0) < 1e-5
        assert np.all(ex.policy_target[~ex.mask] == 0.0)
        assert -1.0 <= ex.value_target <= 1.0


def test_play_self_play_games_batch_with_ensemble_produces_valid_examples():
    net = DomibotNet()
    net.eval()
    games_examples = play_self_play_games_batch(
        net, num_games=3, num_simulations=8, kingdom=_tiny_kingdom(), seed=3, determinization_ensemble_size=4,
    )
    assert len(games_examples) == 3
    for examples in games_examples:
        assert len(examples) > 0
        for ex in examples:
            assert ex.obs.shape == (net.obs_dim,)
            assert ex.mask.shape == (net.num_actions,)
            assert abs(ex.policy_target.sum() - 1.0) < 1e-5
            assert np.all(ex.policy_target[~ex.mask] == 0.0)
            assert -1.0 <= ex.value_target <= 1.0


def test_play_self_play_games_batch_with_ensemble_completes_across_several_seeds():
    # Regression: run_mcts_ensemble's root.boundary/.children point at a
    # *hypothetical* redealt clone, not the real game -- self_play.py's
    # ensemble_positions handling must advance the real game from the true
    # boundary instead. If it didn't, the real game would silently start
    # simulating a wrong hidden reality, which (given enough moves) would
    # plausibly surface as an illegal-action crash somewhere.
    net = DomibotNet()
    net.eval()
    for seed in range(5):
        games_examples = play_self_play_games_batch(
            net, num_games=2, num_simulations=6, kingdom=_tiny_kingdom(),
            seed=seed, max_moves=30, determinization_ensemble_size=3,
        )
        assert len(games_examples) == 2


def _dummy_examples(deciders):
    return [Example(obs=np.zeros(1), mask=np.zeros(1, dtype=bool), policy_target=np.zeros(1), decider=d)
            for d in deciders]


def test_merge_ensemble_roots_sums_W_and_N_across_members():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)
    net = DomibotNet()
    net.eval()
    redealt = [redeal_hidden_info(game, from_player=game.current_decider(), rng=random.Random(i))
               for i in range(3)]
    roots = run_mcts_batch(redealt, net, num_simulations=6)
    expected_N = {a: sum(r.N[a] for r in roots) for a in roots[0].legal_actions}
    expected_W = {a: sum(r.W[a] for r in roots) for a in roots[0].legal_actions}
    merged = merge_ensemble_roots(roots)
    for a in merged.legal_actions:
        assert merged.N[a] == expected_N[a]
        assert abs(merged.W[a] - expected_W[a]) < 1e-9


def test_backfill_value_targets_lambda_zero_matches_terminal_only():
    # td_lambda=0.0 must reproduce the original terminal-only backfill byte
    # for byte, regardless of what root_values/turn_numbers contain.
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    deciders = [0, 1, 0, 1]
    examples = _dummy_examples(deciders)
    root_values = [0.9, -0.9, 0.9, -0.9]  # must be entirely ignored at lambda=0
    turn_numbers = [1, 1, 2, 2]
    _backfill_value_targets(examples, root_values, turn_numbers, game, game_over=True, td_lambda=0.0)
    for ex, d in zip(examples, deciders):
        assert ex.value_known is True
        assert abs(ex.value_target - terminal_value(game, d)) < 1e-9

    truncated_examples = _dummy_examples([0, 1])
    _backfill_value_targets(truncated_examples, [0.5, 0.5], [1, 1], game, game_over=False, td_lambda=0.0)
    for ex in truncated_examples:
        assert ex.value_known is False
        assert ex.value_target == 0.0


def test_backfill_value_targets_pure_bootstrap_uses_next_own_turn():
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    # decider 0 gets two decisions in its own turn 1 (indices 0, 2) before
    # its turn 2 (index 4); decider 1 likewise (indices 1, 3, then 5).
    deciders = [0, 1, 0, 1, 0, 1]
    turn_numbers = [1, 1, 1, 1, 2, 2]
    root_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    examples = _dummy_examples(deciders)
    _backfill_value_targets(examples, root_values, turn_numbers, game, game_over=True,
                             td_lambda=1.0, td_turns_ahead=1)
    # Both of decider 0's turn-1 examples bootstrap from its turn-2 example (index 4).
    assert abs(examples[0].value_target - 0.5) < 1e-9
    assert abs(examples[2].value_target - 0.5) < 1e-9
    # Both of decider 1's turn-1 examples bootstrap from its turn-2 example (index 5).
    assert abs(examples[1].value_target - 0.6) < 1e-9
    assert abs(examples[3].value_target - 0.6) < 1e-9
    # The last recorded turn for each decider has no later same-decider turn
    # to bootstrap from, so it falls back to the true terminal outcome.
    assert examples[4].value_known is True
    assert abs(examples[4].value_target - terminal_value(game, 0)) < 1e-9
    assert examples[5].value_known is True
    assert abs(examples[5].value_target - terminal_value(game, 1)) < 1e-9


def test_backfill_value_targets_recovers_truncated_game_examples():
    # The real point of td_lambda>0: a truncated game (game_over=False) used
    # to waste every example's value signal. Now only the tail (no later
    # same-decider turn recorded) should be unrecoverable.
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    deciders = [0, 0, 0]
    turn_numbers = [1, 2, 3]
    root_values = [0.1, 0.2, 0.3]
    examples = _dummy_examples(deciders)
    _backfill_value_targets(examples, root_values, turn_numbers, game, game_over=False, td_lambda=1.0)
    assert examples[0].value_known is True
    assert abs(examples[0].value_target - 0.2) < 1e-9
    assert examples[1].value_known is True
    assert abs(examples[1].value_target - 0.3) < 1e-9
    assert examples[2].value_known is False
    assert examples[2].value_target == 0.0


def test_backfill_value_targets_blends_terminal_and_bootstrap():
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    turn_numbers = [1, 2]
    root_values = [0.0, 0.8]
    examples = _dummy_examples([0, 0])
    _backfill_value_targets(examples, root_values, turn_numbers, game, game_over=True, td_lambda=0.5)
    terminal = terminal_value(game, 0)
    assert abs(examples[0].value_target - (0.5 * terminal + 0.5 * 0.8)) < 1e-9
    assert examples[0].value_known is True


def test_play_cross_play_games_only_current_seat_produces_examples():
    net_a = DomibotNet()
    net_a.eval()
    net_b = DomibotNet()
    net_b.eval()
    seats_seen: set[int] = set()
    for seed in range(8):
        games_examples = play_cross_play_games(
            net_a, net_b, num_games=2, num_simulations=6, kingdom=_tiny_kingdom(),
            seed=seed, max_moves=30,
        )
        assert len(games_examples) == 2
        for examples in games_examples:
            assert len(examples) > 0
            deciders = {ex.decider for ex in examples}
            assert len(deciders) == 1  # only the assigned current_seat ever produces examples
            seats_seen |= deciders
            for ex in examples:
                assert ex.obs.shape == (net_a.obs_dim,)
                assert ex.mask.shape == (net_a.num_actions,)
                assert abs(ex.policy_target.sum() - 1.0) < 1e-5
                assert np.all(ex.policy_target[~ex.mask] == 0.0)
                assert -1.0 <= ex.value_target <= 1.0
    assert seats_seen == {0, 1}  # both seat assignments actually occurred


def test_play_cross_play_games_with_td_lambda_and_ensemble_completes():
    net_a = DomibotNet()
    net_a.eval()
    net_b = DomibotNet()
    net_b.eval()
    for seed in range(3):
        games_examples = play_cross_play_games(
            net_a, net_b, num_games=2, num_simulations=6, kingdom=_tiny_kingdom(),
            seed=seed, max_moves=30, determinization_ensemble_size=2,
            td_lambda=0.5, td_turns_ahead=1,
        )
        assert len(games_examples) == 2
        for examples in games_examples:
            assert len(examples) > 0
