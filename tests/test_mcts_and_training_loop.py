import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS
from domibot.models import END_ACTIONS
from training.agents import DomibotAgent
from training.evaluate import play_game
from training.mcts import run_mcts, select_action, visit_distribution
from training.network import DomibotNet
from training.self_play import Example, ReplayBuffer, play_self_play_game
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
        assert ex.value_target in (-1.0, 0.0, 1.0)


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
