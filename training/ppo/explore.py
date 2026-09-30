"""Exploration that doesn't die out once the policy stops buying a card.

On-policy PPO only learns about a card from games where the policy itself
bought it. A card bought before the policy can play it well loses to
Silver and Gold, its buy probability falls toward zero, and after that no
game ever holds it again, so nothing can find out what it's worth.
`domibot2.2` lost Throne Room, Village and Chapel this way (see
training/README.md, "Card use"). Steering one or two random cards into its
early buys didn't bring them back: those decks were incoherent and lost,
and an engine only pays as a whole. Two remedies, both used only while
collecting training games (evals and real play are unaffected):

- **Steered players** (`ExploreConfig`, `steer`): in a share of the games,
  every player's buys during its first K turns (1 to `turn_limit`, drawn
  per player) follow a whole buy plan (`plan_search.Plan`) for the board.
  Half the time it's an engine when the board has a village and a draw
  card; otherwise Big Money with one or two copies of one kingdom card, or
  a Gardens rush. Every player is steered because a steered engine loses
  most games to the policy's own buying, which says little about how well
  it was played; against another steered deck, how it's played decides.
  With `plans`, one player follows a plan found by plan search instead, on
  its own board, against the policy's own buying. Every decision besides
  the plan's buys, every card played and every choice, is the policy's
  own, so it learns to play those decks, and the value head learns what
  they're worth. The plan's buys aren't the policy's choice, so they train
  no policy.
- **A buy floor** (`buy_floor_loss`): a penalty whenever the policy gives
  an affordable kingdom card less than a minimum probability. The policy
  keeps sampling every such card now and then, so its own buy decisions
  keep getting feedback, and once the value head learns a card is good
  those samples pull its probability up.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from domibot import KINGDOM_CARDS, Action, Game, Phase

from .. import encoding
from ..plan_search import Plan, PlanAgent, engine_plans, seed_plans


@dataclass
class ExploreConfig:
    frac: float = 0.25  # share of self-play games with one steered player
    turn_limit: int = 16  # its buys follow the plan during its first K turns, K drawn from 1 to this
    plans: tuple = ()  # (board, Plan) pairs to steer with, each on its board; empty: plans for the game's own board


@dataclass
class SteeredPlayer:
    seat: int
    turns: int
    plan: Plan
    engine: bool = False  # the plan is one of plan_search.engine_plans
    board: list[str] | None = None  # the board the game must be played on, for a searched plan


def make_steered(kingdom: list[str], num_players: int, config: ExploreConfig,
                 rng: random.Random) -> list[SteeredPlayer]:
    """The steered players for one game: none (a normal game) with
    probability 1 - `config.frac`; else one player with a searched plan if
    `config.plans`, or every player with its own plan for `kingdom`."""
    if rng.random() >= config.frac:
        return []
    if config.plans:
        board, plan = rng.choice(config.plans)
        return [SteeredPlayer(rng.randrange(num_players), rng.randint(1, config.turn_limit), plan, board=list(board))]
    engines = engine_plans(kingdom)
    others = [p for p in seed_plans(kingdom, rng, n_random=0)[1:] if p not in engines]  # [0] is Big Money
    steered = []
    for seat in range(num_players):
        engine = bool(engines) and rng.random() < 0.5
        plan = rng.choice(engines if engine else others)
        steered.append(SteeredPlayer(seat, rng.randint(1, config.turn_limit), plan, engine))
    return steered


def steer(game: Game, steered: list[SteeredPlayer]) -> Action | None:
    """The plan's buy at this decision, or None to let the policy decide."""
    if game.pending_decision is not None or game.phase != Phase.BUY:
        return None
    for s in steered:
        me = game.players[s.seat]
        if s.seat == game.current_player and me.turns_taken < s.turns:
            return PlanAgent(s.plan).buy(game, game.legal_actions(), me)
    return None


def focus_seat(steered: list[SteeredPlayer]) -> int | None:
    """The steered player whose results are logged (`steered_won`): a
    searched plan's player, or the only engine among plans for the board."""
    focus = steered if len(steered) == 1 else [s for s in steered if s.engine]
    return focus[0].seat if len(focus) == 1 else None


_KINGDOM_BUYS = np.zeros(encoding.NUM_ACTIONS, dtype=bool)
for _card in KINGDOM_CARDS:
    _KINGDOM_BUYS[encoding.action_to_index(Action("BUY", _card))] = True


def buy_floor_loss(masked_logits: torch.Tensor, legal: torch.Tensor, floor: float) -> torch.Tensor:
    """Mean over decisions where a kingdom card can be bought of
    sum(max(0, log(floor) - log pi(buy card))) over those cards: zero once
    every affordable kingdom card has at least probability `floor`."""
    kingdom_buys = legal & torch.from_numpy(_KINGDOM_BUYS).to(legal.device)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    deficit = torch.clamp(math.log(floor) - log_probs, min=0.0) * kingdom_buys
    decisions = kingdom_buys.any(dim=-1).sum().clamp(min=1)
    return deficit.sum() / decisions
