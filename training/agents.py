"""Baseline agents that operate directly on domibot.Game (not through
DominionEnv/encoding.py — a heuristic never needs a numeric state, only a
network-based agent will). Any future agent, network-based or otherwise,
just needs to implement `act(game) -> Action`, so it drops straight into
evaluate.py's play_match alongside these.
"""
from __future__ import annotations

import random
from typing import Optional, Protocol

import torch

from domibot import Action, Game, Phase
from domibot.models import END_ACTIONS, END_BUY

from .heuristics import heuristic_reaction
from .mcts import run_mcts, select_action


class Agent(Protocol):
    def act(self, game: Game) -> Action: ...


class RandomAgent:
    """Picks uniformly among whatever's legal. The floor every other agent
    should beat."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game) -> Action:
        return self.rng.choice(game.legal_actions())


class BigMoneyAgent:
    """The classic simple, strong Dominion baseline: never buys Action
    cards, just Treasures and Victory cards on a fixed priority, spending
    every coin every turn.
    Action-card and reactive decisions can still arise from the *opponent's*
    attacks even though this agent never buys any itself, so it falls back
    to the shared `heuristic_reaction` default there rather than assuming
    they can't happen."""

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            return END_ACTIONS
        return self._choose_buy(game, game.legal_actions())

    def _choose_buy(self, game: Game, actions: list[Action]) -> Action:
        player = game.players[game.current_decider()]
        coins = player.coins
        provinces_left = game.supply.get("Province", 0)

        if coins >= 8 and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if provinces_left <= 4 and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        if coins >= 6 and Action("BUY", "Gold") in actions:
            return Action("BUY", "Gold")
        if coins >= 3 and Action("BUY", "Silver") in actions:
            return Action("BUY", "Silver")
        return END_BUY


class DomibotAgent:
    """A trained policy/value network driving MCTS at every decision --
    phase actions *and* card-effect sub-decisions alike (Chapel's trash
    choices, Militia's forced discard, ...) -- by default. Pass
    `search_sub_decisions=False` to fall back to the fixed heuristic
    (`BigMoneyAgent`'s always-on behavior) for sub-decisions instead, e.g.
    to A/B a checkpoint's strength with and without searching them.

    Unlike a plain function of `game`, this agent keeps a small cache
    (`_boundary`/`_boundary_log_len`) across `.act()` calls so it can search
    a sub-decision without needing to clone the shared, externally-stepped
    `game` object mid-effect -- see `mcts.py`'s module docstring for why
    that's unsafe. The cache is refreshed from `game.action_log` (a
    complete history for the single, never-cloned `Game` object real play
    loops use) every time this agent is asked to decide a phase action, so
    it self-heals at the very first such call and must not be shared
    between two games played concurrently (reuse across *sequential* games,
    e.g. in `evaluate.play_match`'s loop, is fine)."""

    def __init__(
        self,
        network: torch.nn.Module,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: Optional[torch.device] = None,
        search_sub_decisions: bool = True,
        sub_decision_simulations: Optional[int] = None,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device or next(network.parameters()).device
        self.search_sub_decisions = search_sub_decisions
        self.sub_decision_simulations = sub_decision_simulations
        self._boundary: Optional[Game] = None
        self._boundary_log_len = 0

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None and not self.search_sub_decisions:
            return heuristic_reaction(game)

        if game.pending_decision is None:
            boundary, path = game, []
        else:
            if self._boundary is None:
                raise RuntimeError("DomibotAgent.act called mid-effect before it ever saw a "
                                    "phase-action boundary for this game")
            path = [entry.action for entry in game.action_log[self._boundary_log_len:]]
            boundary = self._boundary

        sims = self.num_simulations if not path else (self.sub_decision_simulations or self.num_simulations)
        root = run_mcts(boundary, self.network, sims, c_puct=self.c_puct, device=self.device, path=path)
        if game.pending_decision is None:
            self._boundary = root.game  # run_mcts's own clone -- no extra clone needed
            self._boundary_log_len = len(game.action_log)
        return select_action(root, self.temperature)
