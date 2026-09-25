import numpy as np

from domibot import Game, KINGDOM_CARDS
from training.mcts import terminal_value
from training.network import DomibotNet
from training.ppo.gae import Transition, compute_gae
from training.ppo.rollout import collect_cross_play_rollouts, collect_rollouts


def _tiny_kingdom():
    return list(KINGDOM_CARDS)[:10]


def _dummy_transition(decider: int, value: float, turn_number: int = 0) -> Transition:
    return Transition(obs=np.zeros(1), mask=np.zeros(1, dtype=bool), action=0, log_prob=0.0,
                       value=value, decider=decider, turn_number=turn_number)


def test_compute_gae_returns_equal_terminal_reward_when_undiscounted():
    # With gamma=1, lam=1, GAE's return (advantage + value) telescopes to
    # exactly the terminal reward at every step, regardless of the value
    # estimates used -- a strong, value-independent correctness check.
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    transitions = [_dummy_transition(0, v) for v in [0.1, 0.2, 0.3, -0.5]]
    compute_gae(transitions, game, game_over=True, gamma=1.0, lam=1.0)
    expected = terminal_value(game, 0)
    for t in transitions:
        assert abs(t.return_ - expected) < 1e-9
    assert abs(transitions[-1].reward - expected) < 1e-9
    assert all(t.reward == 0.0 for t in transitions[:-1])


def test_compute_gae_truncated_episode_has_zero_reward_and_zero_final_advantage():
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    transitions = [_dummy_transition(0, v) for v in [0.1, 0.2, 0.3]]
    compute_gae(transitions, game, game_over=False, gamma=1.0, lam=0.95)
    assert all(t.reward == 0.0 for t in transitions)
    # last transition bootstraps from its own value -> delta=0 there
    assert abs(transitions[-1].advantage) < 1e-9
    assert abs(transitions[-1].return_ - transitions[-1].value) < 1e-9


def test_compute_gae_per_decider_subsequence_is_independent_of_interleaving():
    # decider 0's transitions and decider 1's transitions interleaved;
    # each decider's own GAE must match what you'd get by extracting just
    # their subsequence and running compute_gae on it alone.
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    interleaved = [
        _dummy_transition(0, 0.10), _dummy_transition(1, 0.50),
        _dummy_transition(0, 0.20), _dummy_transition(1, 0.60),
        _dummy_transition(0, 0.30),
    ]
    compute_gae(interleaved, game, game_over=True, gamma=0.99, lam=0.9)

    solo0 = [_dummy_transition(0, v) for v in [0.10, 0.20, 0.30]]
    compute_gae(solo0, game, game_over=True, gamma=0.99, lam=0.9)
    dec0 = [t for t in interleaved if t.decider == 0]
    for a, b in zip(dec0, solo0):
        assert abs(a.advantage - b.advantage) < 1e-9
        assert abs(a.return_ - b.return_) < 1e-9

    solo1 = [_dummy_transition(1, v) for v in [0.50, 0.60]]
    compute_gae(solo1, game, game_over=True, gamma=0.99, lam=0.9)
    dec1 = [t for t in interleaved if t.decider == 1]
    for a, b in zip(dec1, solo1):
        assert abs(a.advantage - b.advantage) < 1e-9
        assert abs(a.return_ - b.return_) < 1e-9


def test_collect_rollouts_produces_well_formed_transitions():
    net = DomibotNet()
    net.eval()
    games = collect_rollouts(net, num_games=3, kingdom=_tiny_kingdom(), max_moves=40, seed=1)
    assert len(games) == 3
    for transitions in games:
        assert len(transitions) > 0
        for t in transitions:
            assert t.obs.shape == (net.obs_dim,)
            assert t.mask.shape == (net.num_actions,)
            assert t.mask[t.action]  # the sampled action must have been legal
            assert t.decider in (0, 1)
            assert np.isfinite(t.log_prob)
            assert np.isfinite(t.value)
            assert np.isfinite(t.advantage)
            assert np.isfinite(t.return_)
        # each decider's own turn_number is non-decreasing along their subsequence
        for decider in (0, 1):
            own = [t.turn_number for t in transitions if t.decider == decider]
            assert own == sorted(own)


def test_collect_rollouts_completes_across_several_seeds():
    net = DomibotNet()
    net.eval()
    for seed in range(5):
        games = collect_rollouts(net, num_games=2, kingdom=_tiny_kingdom(), max_moves=30, seed=seed)
        assert len(games) == 2
        assert all(len(g) > 0 for g in games)


def test_collect_cross_play_rollouts_only_records_current_networks_seat():
    net = DomibotNet()
    net.eval()
    opponent = DomibotNet()
    opponent.eval()
    games = collect_cross_play_rollouts(net, opponent, num_games=4, kingdom=_tiny_kingdom(), max_moves=40, seed=2)
    assert len(games) == 4
    for transitions in games:
        assert len(transitions) > 0
        # every recorded transition belongs to a single seat per game (the
        # randomly assigned current_seat) -- cross-play never records the
        # frozen opponent's own decisions
        deciders = {t.decider for t in transitions}
        assert len(deciders) == 1
        for t in transitions:
            assert t.obs.shape == (net.obs_dim,)
            assert t.mask.shape == (net.num_actions,)
            assert t.mask[t.action]
            assert np.isfinite(t.advantage)
            assert np.isfinite(t.return_)


# ------------------------------------------------ domibot2.2 additions ---
import torch  # noqa: E402

from training import encoding  # noqa: E402
from training.ppo.gae import win_weighted_value  # noqa: E402
from training.ppo.train import ppo_update  # noqa: E402


def test_public_extras_at_game_start():
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    extras = encoding.encode_public_extras(game, 0)
    assert extras.shape == (encoding.EXTRA_DIM,)
    n = encoding.NUM_CARDS
    in_game = extras[:n]
    assert in_game.sum() == 17  # 7 basic piles + 10 kingdom
    assert extras[n] == 0  # empty piles
    assert extras[n + 1] == 3  # my VP: 3 Estates
    opp_counts = extras[n + 2:n + 2 + n]
    assert opp_counts[encoding.CARD_INDEX["Copper"]] == 7
    assert opp_counts[encoding.CARD_INDEX["Estate"]] == 3
    assert extras[n + 2 + n] == 3  # opponent VP
    assert extras[-1] == 0  # VP lead
    full = encoding.encode_full_observation(game, 0)
    assert full.shape == (encoding.FULL_OBS_DIM,)
    assert np.array_equal(full[:encoding.OBS_DIM], encoding.encode_observation(game, 0))


def test_empty_pile_count_and_lead():
    game = Game(_tiny_kingdom(), num_players=2, seed=0)
    game.supply[_tiny_kingdom()[0]] = 0
    game.players[0].discard.append("Province")
    extras = encoding.encode_public_extras(game, 0)
    n = encoding.NUM_CARDS
    assert extras[n] == 1
    assert extras[-1] == 6


def test_with_extra_inputs_preserves_outputs_and_round_trips(tmp_path):
    torch.manual_seed(0)
    base = DomibotNet()
    base.eval()
    upgraded = base.with_extra_inputs()
    upgraded.eval()
    obs = torch.randn(8, encoding.FULL_OBS_DIM)
    with torch.no_grad():
        lb, vb = base(obs)  # a base network reads only the prefix of a full encoding
        lu, vu = upgraded(obs)
    assert torch.allclose(lb, lu) and torch.allclose(vb, vu)

    path = tmp_path / "net.pt"
    upgraded.save(path)
    loaded = DomibotNet.load(path)
    assert loaded.extra_dim == encoding.EXTRA_DIM
    loaded.eval()
    with torch.no_grad():
        ll, vl = loaded(obs)
    assert torch.allclose(lu, ll) and torch.allclose(vu, vl)


def _finished_game(seed: int) -> Game:
    import random as _random
    game = Game(_tiny_kingdom(), num_players=2, seed=seed)
    rng = _random.Random(seed)
    while not game.is_game_over():
        game.step(rng.choice(game.legal_actions()))
    return game


def test_win_weighted_value():
    game = _finished_game(3)
    for p in (0, 1):
        assert abs(win_weighted_value(game, p, win_weight=0.0) - terminal_value(game, p)) < 1e-12
    winners = game.winners()
    if len(winners) == 1:
        w = winners[0]
        assert win_weighted_value(game, w) >= 0.8
        assert win_weighted_value(game, 1 - w) <= -0.8


def test_ppo_update_ignores_forced_moves_and_reports_kl():
    net = DomibotNet(extra_dim=encoding.EXTRA_DIM)
    net.eval()
    games = collect_rollouts(net, num_games=2, kingdom=_tiny_kingdom(), max_moves=60, seed=5)
    transitions = [t for g in games for t in g]
    assert all(t.obs.shape == (encoding.FULL_OBS_DIM,) for t in transitions)
    assert any(t.mask.sum() == 1 for t in transitions)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    stats = ppo_update(net, opt, transitions, torch.device("cpu"), epochs=2, minibatch_size=32)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert np.isfinite(stats[k]), k
    assert stats["updates"] > 0

    forced_only = [t for t in transitions if t.mask.sum() == 1]
    stats = ppo_update(net, opt, forced_only, torch.device("cpu"), epochs=1, minibatch_size=32)
    assert stats["policy_loss"] == 0.0 and stats["entropy"] == 0.0


def test_cross_play_mixes_extras_and_base_networks():
    net = DomibotNet(extra_dim=encoding.EXTRA_DIM)
    opponent = DomibotNet()
    net.eval()
    opponent.eval()
    games = collect_cross_play_rollouts(net, opponent, num_games=2, kingdom=_tiny_kingdom(), max_moves=40, seed=6)
    assert all(t.obs.shape == (encoding.FULL_OBS_DIM,) for g in games for t in g)
