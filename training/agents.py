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
    """A trained policy/value network driving MCTS at each phase-action
    decision, falling back to the same fixed heuristic as BigMoneyAgent for card-effect sub-decisions (see mcts.py's module docstring for why those aren't searched)."""

    def __init__(
        self,
        network: torch.nn.Module,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device or next(network.parameters()).device

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        root = run_mcts(game, self.network, self.num_simulations, c_puct=self.c_puct, device=self.device)
        return select_action(root, self.temperature)
