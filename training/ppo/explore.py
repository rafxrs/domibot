"""Steering: in a share of the self-play games, players' buys follow whole buy plans.

On-policy PPO never learns a buy it never makes. Steering makes it for the
policy, which still plays the cards, so it learns to play decks it wouldn't
build. Template plans (an engine half the time on boards with a village and a
draw card, else Big Money plus one kingdom card, or a Gardens rush) steer every
player for its first 1..`turn_limit` turns, so a weak scripted engine meets
another steered deck rather than the policy's own buying. Searched plans
(`plans`) steer one player, on their own board, against the policy's buying.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from domibot import Action, Game, Phase

from ..plan_search import Plan, PlanAgent, engine_plans, seed_plans


@dataclass
class ExploreConfig:
    frac: float = 0.25  # share of self-play games that are steered
    turn_limit: int = 16  # buys follow the plan for the first K turns, K drawn from 1..turn_limit
    plans: tuple = ()  # searched (board, Plan) pairs; empty: template plans for each game's board


@dataclass
class SteeredPlayer:
    seat: int
    turns: int
    plan: Plan
    engine: bool = False  # one of plan_search.engine_plans
    board: list[str] | None = None  # a searched plan's board, which the game must use


def make_steered(kingdom: list[str], num_players: int, config: ExploreConfig,
                 rng: random.Random) -> list[SteeredPlayer]:
    """No one (a normal game), one player with a searched plan, or every player with a template."""
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
    """Whose result `steered_won` logs: a searched plan's player, or the only engine player."""
    focus = steered if len(steered) == 1 else [s for s in steered if s.engine]
    return focus[0].seat if len(focus) == 1 else None

