"""Agents: anything with `act(game) -> Action` plays in `evaluate.play_match`."""
from __future__ import annotations

import random
from typing import Optional, Protocol

import torch

from domibot import Action, Game, Phase
from domibot.models import END_ACTIONS, END_BUY

from . import encoding
from .heuristics import heuristic_reaction
from .mcts import run_mcts, run_mcts_batch, select_action

# The terminal Actions a Big Money deck gains most from, best first.
TERMINAL_PRIORITY = ("Witch", "Smithy", "Council Room", "Library", "Militia", "Bandit")


class Agent(Protocol):
    def act(self, game: Game) -> Action: ...


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game) -> Action:
        return self.rng.choice(game.legal_actions())


def _buy_money(coins: int, actions: list[Action]) -> Action:
    for card, cost in (("Gold", 6), ("Silver", 3)):
        if coins >= cost and Action("BUY", card) in actions:
            return Action("BUY", card)
    return END_BUY


class BigMoneyAgent:
    """Buys only Treasures and Victory cards; card-effect choices use `heuristic_reaction`."""

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            return END_ACTIONS
        actions, coins = game.legal_actions(), game.players[game.current_decider()].coins
        if coins >= 8 and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if game.supply.get("Province", 0) <= 4 and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        return _buy_money(coins, actions)


class BigMoneyTerminalAgent:
    """Big Money plus the kingdom's best terminal (`TERMINAL_PRIORITY`): one copy,
    two with Witch or once the deck has 16+ cards; Duchy at <=5 Provinces left,
    Estate at <=2."""

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        actions = game.legal_actions()
        if game.phase == Phase.ACTION:
            return next((Action("PLAY", c) for c in TERMINAL_PRIORITY if Action("PLAY", c) in actions), END_ACTIONS)
        player = game.players[game.current_decider()]
        coins, provinces, owned = player.coins, game.supply.get("Province", 0), player.all_cards()
        if coins >= 8 and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if provinces <= 5 and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        if provinces <= 2 and 2 <= coins <= 4 and Action("BUY", "Estate") in actions:
            return Action("BUY", "Estate")
        terminals = sum(owned.count(c) for c in TERMINAL_PRIORITY)
        best = next((c for c in TERMINAL_PRIORITY if Action("BUY", c) in actions), None)
        if best is not None:
            cap = 2 if best == "Witch" or len(owned) >= 16 else 1
            if terminals < cap and (coins <= 5 or terminals == 0):  # at $6+ Gold beats a second terminal
                return Action("BUY", best)
        return _buy_money(coins, actions)


class PPOAgent:
    """A network's greedy policy, no search."""

    def __init__(self, network: torch.nn.Module, device: torch.device | None = None):
        self.network = network
        self.device = device or next(network.parameters()).device

    def act(self, game: Game) -> Action:
        obs = torch.from_numpy(encoding.encode_for(self.network, game, game.current_decider())).unsqueeze(0)
        mask = torch.from_numpy(encoding.legal_action_mask(game)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _value = self.network(obs.to(self.device))
        return encoding.index_to_action(int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item()))


class DomibotAgent:
    """MCTS over the true game state (opponent's hand and deck order included).

    Card-effect choices are searched too, unless `search_sub_decisions=False`
    (then `heuristic_reaction`). It caches the last phase-action boundary to
    search from mid-effect, so use one instance per game at a time.
    """

    def __init__(self, network: torch.nn.Module, num_simulations: int = 200, c_puct: float = 1.5,
                 temperature: float = 0.0, device: Optional[torch.device] = None, search_sub_decisions: bool = True):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device or next(network.parameters()).device
        self.search_sub_decisions = search_sub_decisions
        self._boundary: Optional[Game] = None
        self._boundary_log_len = 0

    def act(self, game: Game) -> Action:
        if game.pending_decision is None:
            root = run_mcts(game, self.network, self.num_simulations, self.c_puct, self.device)
            self._boundary, self._boundary_log_len = root.game, len(game.action_log)
        elif not self.search_sub_decisions:
            return heuristic_reaction(game)
        elif self._boundary is None:
            raise RuntimeError("DomibotAgent asked mid-effect before it saw a phase-action boundary")
        else:
            path = [entry.action for entry in game.action_log[self._boundary_log_len:]]
            root = run_mcts(self._boundary, self.network, self.num_simulations, self.c_puct, self.device, path)
        return select_action(root, self.temperature)


class DeterminizedSearchAgent:
    """MCTS over only what the player could know, as the relay tool searches.

    Each phase decision searches `determinizations` copies of the game, each
    with the opponent's hand and deck reshuffled together, its own deck
    reshuffled and a fresh random seed, and adds up their visits. Card-effect
    choices use the network's greedy policy.
    """

    def __init__(self, network: torch.nn.Module, num_simulations: int = 400, c_puct: float = 1.5,
                 device: Optional[torch.device] = None, seed: int | None = None, determinizations: int = 1):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device or next(network.parameters()).device
        self.rng = random.Random(seed)
        self.determinizations = max(1, determinizations)
        self.policy = PPOAgent(network, self.device)

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        if len(actions) == 1:
            return actions[0]
        if game.pending_decision is not None:
            return self.policy.act(game)
        worlds = [self._determinize(game, game.current_decider()) for _ in range(self.determinizations)]
        roots = run_mcts_batch(worlds, self.network, max(1, self.num_simulations // len(worlds)),
                               c_puct=self.c_puct, device=self.device)
        return max(actions, key=lambda a: sum(root.N[a] for root in roots))

    def _determinize(self, game: Game, me: int) -> Game:
        world = game.clone()
        for i, player in enumerate(world.players):
            if i == me:
                self.rng.shuffle(player.deck)
            else:
                pool = player.hand + player.deck
                self.rng.shuffle(pool)
                player.hand, player.deck = pool[:len(player.hand)], pool[len(player.hand):]
        world.rng.seed(self.rng.getrandbits(64))
        return world
