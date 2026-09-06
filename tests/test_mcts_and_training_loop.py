import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS
from domibot.models import END_ACTIONS
from training.agents import DomibotAgent
from training.evaluate import play_game
from training.mcts import run_mcts, run_mcts_batch, select_action, terminal_value, visit_distribution
from training.network import DomibotNet
from training.self_play import Example, ReplayBuffer, play_self_play_game, play_self_play_games_batch
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
