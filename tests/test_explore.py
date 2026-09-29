import random

import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS, Phase
from domibot.models import END_ACTIONS
from training import encoding
from training.network import DomibotNet
from training.ppo.explore import ExploreConfig, FocusPlan, buy_floor_loss, make_plan, steer
from training.ppo.gae import Transition, compute_gae
from training.ppo.rollout import collect_rollouts
from training.ppo.train import ppo_update

KINGDOM = list(KINGDOM_CARDS)[:10]
ALWAYS = ExploreConfig(frac=1.0, override_prob=1.0)


def _buy_phase(coins: int) -> Game:
    game = Game(KINGDOM, num_players=2, seed=0)
    game.step(END_ACTIONS)
    game.players[0].coins = coins
    return game


def test_steer_buys_a_wanted_affordable_focus_card_for_the_steered_player_only():
    rng = random.Random(0)
    card = next(c for c in KINGDOM if 3 <= Game(KINGDOM).cards[c].cost <= 5)
    plan = FocusPlan(seat=0, wanted={card: 2})
    assert steer(_buy_phase(5), plan, ALWAYS, rng) == Action("BUY", card)
    assert steer(_buy_phase(0), plan, ALWAYS, rng) is None  # can't afford it
    assert steer(_buy_phase(5), FocusPlan(seat=1, wanted={card: 2}), ALWAYS, rng) is None  # not its turn
    game = _buy_phase(5)
    game.players[0].discard += [card, card]
    assert steer(game, plan, ALWAYS, rng) is None  # already has the copies it wants
    game = _buy_phase(5)
    game.players[0].turns_taken = ALWAYS.turn_limit
    assert steer(game, plan, ALWAYS, rng) is None  # past the steered turns
    game = Game(KINGDOM, num_players=2, seed=0)
    assert game.phase == Phase.ACTION and steer(game, plan, ALWAYS, rng) is None  # only buys are steered


def test_make_plan_draws_from_the_kingdom():
    rng = random.Random(1)
    plans = [make_plan(KINGDOM, 2, ExploreConfig(frac=0.5), rng) for _ in range(200)]
    steered = [p for p in plans if p is not None]
    assert 60 < len(steered) < 140
    for plan in steered:
        assert plan.seat in (0, 1) and 1 <= len(plan.wanted) <= 2
        assert all(c in KINGDOM and 1 <= n <= 3 for c, n in plan.wanted.items())


def _t(value: float, explore: bool = False) -> Transition:
    return Transition(obs=np.zeros(1), mask=np.zeros(1, dtype=bool), action=0, log_prob=0.0, value=value,
                      decider=0, turn_number=0, explore=explore)


def test_gae_trace_stops_at_an_exploring_buy():
    game = Game(KINGDOM, num_players=2, seed=0)
    plain = [_t(0.1), _t(0.2), _t(0.4), _t(0.3)]
    cut = [_t(0.1), _t(0.2, explore=True), _t(0.4), _t(0.3)]
    compute_gae(plain, game, game_over=True, gamma=1.0, lam=0.9)
    compute_gae(cut, game, game_over=True, gamma=1.0, lam=0.9)
    # the decision before the exploring buy gets a one-step advantage only...
    assert abs(cut[0].advantage - (0.2 - 0.1)) < 1e-9
    assert abs(plain[0].advantage - (0.2 - 0.1)) > 1e-3
    # ...and everything from the exploring buy on is unchanged
    for a, b in zip(plain[1:], cut[1:]):
        assert abs(a.advantage - b.advantage) < 1e-9


def test_rollouts_record_steered_buys_as_exploration():
    torch.manual_seed(0)
    net = DomibotNet(hidden_dim=32, num_blocks=1, extra_dim=encoding.EXTRA_DIM, zones_dim=encoding.ZONES_DIM)
    games = collect_rollouts(net, 4, seed=0, max_moves=400, explore=ALWAYS)
    steered = [t for g in games for t in g if t.explore]
    assert steered
    for t in steered:
        action = encoding.index_to_action(t.action)
        assert action.verb == "BUY" and action.card in KINGDOM_CARDS and t.mask[t.action]
    assert not any(t.explore for g in collect_rollouts(net, 2, seed=0, max_moves=400) for t in g)


def test_exploring_buys_train_nothing():
    torch.manual_seed(0)
    net = DomibotNet(hidden_dim=32, num_blocks=1, extra_dim=encoding.EXTRA_DIM, zones_dim=encoding.ZONES_DIM)
    transitions = [t for g in collect_rollouts(net, 2, seed=1, max_moves=300) for t in g]
    for t in transitions:
        t.explore = True
    before = [p.detach().clone() for p in net.parameters()]
    ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), transitions, torch.device("cpu"), epochs=1)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_buy_floor_loss_lifts_only_kingdom_buys_below_the_floor():
    kingdom_buy = encoding.action_to_index(Action("BUY", KINGDOM[0]))
    silver = encoding.action_to_index(Action("BUY", "Silver"))
    legal = torch.zeros(1, encoding.NUM_ACTIONS, dtype=torch.bool)
    legal[0, [kingdom_buy, silver]] = True
    logits = torch.zeros(1, encoding.NUM_ACTIONS)
    logits[0, silver] = 10.0
    logits.requires_grad_(True)
    masked = logits.masked_fill(~legal, -1e9)
    loss = buy_floor_loss(masked, legal, floor=0.01)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad[0, kingdom_buy] < 0  # descending the loss raises the kingdom buy's logit
    even = torch.zeros(1, encoding.NUM_ACTIONS).masked_fill(~legal, -1e9)
    assert buy_floor_loss(even, legal, floor=0.01).item() == 0.0
