import random

import torch

from domibot import Action, Game, KINGDOM_CARDS, Phase
from domibot.models import END_ACTIONS, END_BUY
from training import encoding
from training.network import DomibotNet
from training.plan_search import MONEY, Plan, menu_options
from training.ppo.explore import ExploreConfig, SteeredPlayer, buy_floor_loss, make_plan, steer
from training.ppo.rollout import collect_rollouts
from training.ppo.train import ppo_update

KINGDOM = list(KINGDOM_CARDS)[:10]
BOARD = list(KINGDOM_CARDS)[-10:]


def _buy_phase(coins: int) -> Game:
    game = Game(KINGDOM, num_players=2, seed=0)
    game.step(END_ACTIONS)
    game.players[0].coins = coins
    return game


def test_steer_buys_by_the_plan_for_the_steered_player_during_its_first_turns_only():
    card = next(c for c in KINGDOM if 3 <= Game(KINGDOM).cards[c].cost <= 5)
    steered = SteeredPlayer(seat=0, turns=3, plan=Plan(((card, 2),) + MONEY))
    assert steer(_buy_phase(5), steered) == Action("BUY", card)
    assert steer(_buy_phase(0), steered) == END_BUY  # nothing on the plan it can afford
    assert steer(_buy_phase(5), SteeredPlayer(seat=1, turns=3, plan=steered.plan)) is None  # not its turn
    game = _buy_phase(5)
    game.players[0].turns_taken = 3
    assert steer(game, steered) is None  # past its steered turns
    game = Game(KINGDOM, num_players=2, seed=0)
    assert game.phase == Phase.ACTION and steer(game, steered) is None  # only buys are steered


def test_plans_come_from_the_board_or_from_the_searched_plans():
    rng = random.Random(1)
    config = ExploreConfig(frac=0.5)
    steered = [s for s in (make_plan(KINGDOM, 2, config, rng) for _ in range(200)) if s is not None]
    assert 60 < len(steered) < 140
    options = set(menu_options(KINGDOM))
    for s in steered:
        assert s.seat in (0, 1) and 1 <= s.turns <= config.turn_limit and s.board is None
        assert s.plan.menu != MONEY and all(card in options for card, _n in s.plan.menu)
    searched = Plan(((BOARD[0], 2),) + MONEY)
    s = make_plan(KINGDOM, 2, ExploreConfig(frac=1.0, plans=((BOARD, searched),)), rng)
    assert s.plan == searched and s.board == BOARD


def test_rollouts_steer_buys_on_the_searched_plans_board():
    torch.manual_seed(0)
    net = DomibotNet(hidden_dim=32, num_blocks=1, extra_dim=encoding.EXTRA_DIM, zones_dim=encoding.ZONES_DIM)
    config = ExploreConfig(frac=1.0, plans=((BOARD, Plan(((BOARD[0], 2),) + MONEY)),))
    games = collect_rollouts(net, 4, seed=0, max_moves=400, explore=config)
    steered = [t for g in games for t in g if t.explore]
    assert steered
    for t in steered:
        assert encoding.index_to_action(t.action).verb in ("BUY", "END_BUY") and t.mask[t.action]
    off_board = [encoding.action_to_index(Action("BUY", c)) for c in KINGDOM_CARDS if c not in BOARD]
    assert not any(t.mask[off_board].any() for g in games for t in g)
    assert not any(t.explore for g in collect_rollouts(net, 2, seed=0, max_moves=400) for t in g)


def test_steered_buys_train_only_the_value_head():
    torch.manual_seed(0)
    net = DomibotNet(hidden_dim=32, num_blocks=1, extra_dim=encoding.EXTRA_DIM, zones_dim=encoding.ZONES_DIM)
    transitions = [t for g in collect_rollouts(net, 2, seed=1, max_moves=300) for t in g]
    for t in transitions:
        t.explore = True
    policy_before = [p.detach().clone() for p in net.policy_head.parameters()]
    value_before = [p.detach().clone() for p in net.value_head.parameters()]
    ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), transitions, torch.device("cpu"), epochs=1)
    assert all(torch.equal(a, b) for a, b in zip(policy_before, net.policy_head.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(value_before, net.value_head.parameters()))


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
