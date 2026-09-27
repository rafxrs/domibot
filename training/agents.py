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

from . import encoding
from .heuristics import heuristic_reaction
from .mcts import run_mcts, run_mcts_batch, run_mcts_ensemble, select_action


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


# Best-first: the single terminal Action a Big Money deck is most helped by,
# among whichever ones this kingdom actually offers. No kingdom is ever
# required to contain any of them -- with none present the agent below
# simply plays plain Big Money.
TERMINAL_PRIORITY = ("Witch", "Smithy", "Council Room", "Library", "Militia", "Bandit")


class BigMoneyTerminalAgent:
    """Big Money plus the best terminal Action this kingdom happens to have
    (`TERMINAL_PRIORITY`): a harder yardstick than `BigMoneyAgent` that
    still adapts to whatever kingdom it's dealt instead of demanding a
    specific card. Buys the first available terminal it can afford while
    it owns few of them (one, or two once the deck has 16+ cards; a Witch
    opening allows two), plays one whenever it holds one, and otherwise
    follows Big Money -- with the standard late-game greening tweaks
    (Duchy at <=5 Provinces left, Estate at <=2). Sub-decisions use the
    shared `heuristic_reaction`, same as `BigMoneyAgent`."""

    def act(self, game: Game) -> Action:
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        actions = game.legal_actions()
        if game.phase == Phase.ACTION:
            for card in TERMINAL_PRIORITY:
                if Action("PLAY", card) in actions:
                    return Action("PLAY", card)
            return END_ACTIONS
        return self._choose_buy(game, actions)

    def _choose_buy(self, game: Game, actions: list[Action]) -> Action:
        player = game.players[game.current_decider()]
        coins = player.coins
        provinces_left = game.supply.get("Province", 0)
        owned = player.all_cards()

        if coins >= 8 and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if provinces_left <= 5 and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        if provinces_left <= 2 and 2 <= coins <= 4 and Action("BUY", "Estate") in actions:
            return Action("BUY", "Estate")

        terminals_owned = sum(owned.count(c) for c in TERMINAL_PRIORITY)
        best = next((c for c in TERMINAL_PRIORITY if Action("BUY", c) in actions), None)
        if best is not None:
            cap = 2 if (best == "Witch" or len(owned) >= 16) else 1
            # at 6+ coins Gold is normally better, except for a first terminal
            if terminals_owned < cap and (coins <= 5 or terminals_owned == 0):
                return Action("BUY", best)

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
        determinization_ensemble_size: int = 1,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device or next(network.parameters()).device
        self.search_sub_decisions = search_sub_decisions
        self.sub_decision_simulations = sub_decision_simulations
        self.determinization_ensemble_size = determinization_ensemble_size
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
        if not path and self.determinization_ensemble_size > 1:
            root = run_mcts_ensemble(
                boundary, self.network, sims, self.determinization_ensemble_size,
                c_puct=self.c_puct, device=self.device,
            )
            # root.game here is one hypothetical redealt clone (see
            # mcts.redeal_hidden_info), not the real position -- cache a
            # clone of the true `boundary` instead.
            new_boundary = boundary.clone()
        else:
            root = run_mcts(boundary, self.network, sims, c_puct=self.c_puct, device=self.device, path=path)
            new_boundary = root.game  # run_mcts's own clone -- no extra clone needed
        if game.pending_decision is None:
            self._boundary = new_boundary
            self._boundary_log_len = len(game.action_log)
        return select_action(root, self.temperature)


class DeterminizedSearchAgent:
    """MCTS on top of a network, searching only what the player could
    actually know -- the setting the relay tool advises in. At each phase
    decision it searches a copy of the game in which the opponent's hand
    and deck are reshuffled together (sizes kept), its own deck order is
    reshuffled, and the random seed is re-rolled, so neither hidden cards
    nor upcoming draws leak into the search. Card-effect sub-decisions and
    reactions use the network's raw policy (greedy, like
    `ppo.train.PPOAgent`).

    Every simulation of one search replays the same copy, random seed
    included, so it sees one sampled future: the order of the next draws and
    shuffles is fixed within it. `determinizations` > 1 searches that many
    independent copies (`num_simulations` split between them, one batched
    network call per round) and adds up their visit counts, averaging the
    choice over sampled futures instead of tuning it to one.

    Compare `DomibotAgent`, which (with the default ensemble size of 1)
    searches the true game and so sees the opponent's hand and the order of
    every deck. One difference from the relay remains: this doesn't know
    about cards it itself put on top of its deck (a Sentry topdeck), which
    the relay does."""

    def __init__(self, network: torch.nn.Module, num_simulations: int = 400, c_puct: float = 1.5,
                 device: Optional[torch.device] = None, seed: int | None = None, determinizations: int = 1):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device or next(network.parameters()).device
        self.rng = random.Random(seed)
        self.determinizations = max(1, determinizations)

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        if len(actions) == 1:
            return actions[0]
        if game.pending_decision is not None:
            return self._policy_action(game)
        me = game.current_decider()
        worlds = [self._determinize(game, me) for _ in range(self.determinizations)]
        sims = max(1, self.num_simulations // len(worlds))
        roots = run_mcts_batch(worlds, self.network, sims, c_puct=self.c_puct, device=self.device)
        visits = {a: sum(root.N[a] for root in roots) for a in actions}
        return max(actions, key=lambda a: visits[a])

    def _determinize(self, game: Game, me: int) -> Game:
        """A copy with every hidden order/split resampled: the opponent's
        hand+deck reshuffled together (sizes kept), my deck reshuffled, and
        a fresh random seed for later shuffles."""
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

    def _policy_action(self, game: Game) -> Action:
        obs = torch.from_numpy(encoding.encode_for(self.network, game, game.current_decider())).unsqueeze(0)
        mask = torch.from_numpy(encoding.legal_action_mask(game)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _value = self.network(obs.to(self.device))
        return encoding.index_to_action(int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item()))
