from __future__ import annotations

import inspect
import random
from typing import Generator, Optional

from . import gamelog
from .cards import ALL_CARDS, KINGDOM_CARDS
from .effects import move
from .enums import CardType, DecisionKind, Phase
from .models import END_ACTIONS, END_BUY, Action, Decision, LogEntry
from .player import PlayerState

STARTING_COPPER = 7
STARTING_ESTATE = 3
STARTING_HAND = 5


class Game:
    """Dominion (base set) engine driven by legal_actions() / step(Action).

    There is always exactly one thing that can happen next: either a card
    effect is mid-resolution and waiting on a Decision (`pending_decision`
    is set, and `step` resumes the suspended effect generator), or the
    active player is choosing a top-level phase action (play/buy/end phase).
    This makes the engine trivial to drive from a random playout, a human
    CLI, or an MCTS search: always call legal_actions(), pick one, step().
    """

    def __init__(self, kingdom_cards: list[str], num_players: int = 2, seed: Optional[int] = None):
        if not (2 <= num_players <= 4):
            raise ValueError("the base set supports 2-4 players")
        if len(kingdom_cards) != 10 or len(set(kingdom_cards)) != 10:
            raise ValueError("must supply exactly 10 distinct kingdom card names")
        unknown = set(kingdom_cards) - set(KINGDOM_CARDS)
        if unknown:
            raise ValueError(f"unknown kingdom cards: {sorted(unknown)}")

        self.rng = random.Random(seed)
        self.seed = seed
        self.cards = ALL_CARDS
        self.kingdom = list(kingdom_cards)
        self.num_players = num_players

        victory_pile_size = 8 if num_players == 2 else 12
        self.supply: dict[str, int] = {
            "Copper": 60 - STARTING_COPPER * num_players,
            "Silver": 40,
            "Gold": 30,
            "Estate": victory_pile_size,
            "Duchy": victory_pile_size,
            "Province": victory_pile_size,
            "Curse": 10 * (num_players - 1),
        }
        for name in self.kingdom:
            self.supply[name] = 10

        self.trash: list[str] = []
        self.players = [PlayerState(name=f"P{i}") for i in range(num_players)]
        for player in self.players:
            player.deck = ["Copper"] * STARTING_COPPER + ["Estate"] * STARTING_ESTATE
            self.rng.shuffle(player.deck)
            player.draw(STARTING_HAND, self.rng)

        self.current_player = 0
        self.turn_number = 1
        self.phase = Phase.ACTION
        self.players[0].actions = 1
        self.players[0].buys = 1

        # Merchant's "first Silver played this turn" bonus is turn-scoped
        # state, not tied to any single card instance, hence tracked here.
        self.turn_merchant_bonus = 0
        self.turn_silver_played = False

        self.pending_gen: Optional[Generator[Decision, Action, None]] = None
        self.pending_decision: Optional[Decision] = None
        self.action_log: list[LogEntry] = []

    # ------------------------------------------------------------ query ---
    def current_decider(self) -> int:
        return self.pending_decision.player if self.pending_decision is not None else self.current_player

    def is_game_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def legal_actions(self) -> list[Action]:
        if self.pending_decision is not None:
            return list(self.pending_decision.options)
        return self._phase_actions()

    @property
    def current_turn_number(self) -> int:
        """1-indexed turn count for the player whose turn it currently is
        (i.e. `self.current_player`), independent of how many turns anyone
        else has taken. Player 0's second turn is turn 2, even though the
        opponent has taken a turn in between."""
        return self.players[self.current_player].turns_taken + 1

    def other_players_in_order(self, idx: int) -> list[int]:
        n = len(self.players)
        return [(idx + i) % n for i in range(1, n)]

    def get_scores(self) -> dict[int, int]:
        return {i: p.victory_points(self.cards) for i, p in enumerate(self.players)}

    def winners(self) -> list[int]:
        scores = self.get_scores()
        best = max(scores.values())
        tied = [i for i, s in scores.items() if s == best]
        if len(tied) == 1:
            return tied
        fewest_turns = min(self.players[i].turns_taken for i in tied)
        return [i for i in tied if self.players[i].turns_taken == fewest_turns]

    def save_log(self, path, fmt: str = "text") -> None:
        gamelog.save(self, path, fmt=fmt)

    def clone(self) -> "Game":
        """A deep-enough independent copy for search (MCTS and similar).
        Only safe when no card effect is mid-resolution: a suspended
        `pending_gen` is a live generator whose frame captured a reference
        to *this* Game object, so a clone would silently share (and corrupt)
        state with the original the moment that generator resumed. Every
        caller must therefore only clone at a decision boundary where
        `pending_decision is None` (a PHASE_ACTION point) or the game is
        already over."""
        if self.pending_gen is not None:
            raise RuntimeError("cannot clone a Game while a card effect is mid-resolution")
        new = Game.__new__(Game)
        new.rng = random.Random()
        new.rng.setstate(self.rng.getstate())
        new.seed = self.seed
        new.cards = self.cards  # immutable card definitions, safe to share
        new.kingdom = self.kingdom  # never mutated after __init__, safe to share
        new.num_players = self.num_players
        new.supply = dict(self.supply)
        new.trash = list(self.trash)
        new.players = [p.clone() for p in self.players]
        new.current_player = self.current_player
        new.turn_number = self.turn_number
        new.phase = self.phase
        new.turn_merchant_bonus = self.turn_merchant_bonus
        new.turn_silver_played = self.turn_silver_played
        new.pending_gen = None
        new.pending_decision = None
        new.action_log = []  # search clones don't need history
        return new

    # ------------------------------------------------------------- step ---
    def step(self, action: Action) -> None:
        if self.phase == Phase.GAME_OVER:
            raise RuntimeError("game is already over")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r} (legal: {self.legal_actions()})")

        self.action_log.append(LogEntry(self.current_turn_number, self.current_decider(), action))
        if self.pending_gen is not None:
            self._resume_generator(action)
        else:
            self._handle_phase_action(action)

        if self.pending_gen is None:
            self._maybe_end_game()

    def _resume_generator(self, action: Action) -> None:
        try:
            self.pending_decision = self.pending_gen.send(action)
        except StopIteration:
            self.pending_gen = None
            self.pending_decision = None

    def _start_generator(self, gen: Generator[Decision, Action, None]) -> None:
        self.pending_gen = gen
        try:
            self.pending_decision = next(gen)
        except StopIteration:
            self.pending_gen = None
            self.pending_decision = None

    def _phase_actions(self) -> list[Action]:
        player = self.players[self.current_player]
        actions: list[Action] = []
        if self.phase == Phase.ACTION:
            if player.actions > 0:
                for name in dict.fromkeys(player.hand):
                    if CardType.ACTION in self.cards[name].types:
                        actions.append(Action("PLAY", name))
            actions.append(END_ACTIONS)
        elif self.phase == Phase.BUY:
            if player.buys > 0:
                for name, count in self.supply.items():
                    if count > 0 and self.cards[name].cost <= player.coins:
                        actions.append(Action("BUY", name))
            actions.append(END_BUY)
        return actions

    def _handle_phase_action(self, action: Action) -> None:
        player = self.players[self.current_player]
        if self.phase == Phase.ACTION:
            if action == END_ACTIONS:
                self.phase = Phase.BUY
                self._auto_play_treasures(self.current_player)
                return
            name = action.card
            move(name, player.hand, player.play_area)
            player.actions -= 1
            self._start_generator(self.resolve_action(self.current_player, name))
        elif self.phase == Phase.BUY:
            if action == END_BUY:
                self._cleanup_and_advance()
                return
            self._buy_card(action.card)

    # ----------------------------------------------------- card effects ---
    def resolve_action(self, player_idx: int, card_name: str) -> Generator[Decision, Action, None]:
        """Apply an Action card's flat bonuses plus its effect. Shared by a
        normal PLAY, and by Throne Room / Vassal replaying a card."""
        card = self.cards[card_name]
        player = self.players[player_idx]
        player.actions += card.plus_actions
        player.buys += card.plus_buys
        player.coins += card.plus_coins
        if card.plus_cards:
            player.draw(card.plus_cards, self.rng)
        if card.effect is not None:
            result = card.effect(self, player_idx)
            if inspect.isgenerator(result):
                yield from result

    def play_treasure(self, player_idx: int, card_name: str) -> None:
        player = self.players[player_idx]
        move(card_name, player.hand, player.play_area)
        card = self.cards[card_name]
        player.coins += card.coin_value
        if card_name == "Silver" and not self.turn_silver_played:
            self.turn_silver_played = True
            player.coins += self.turn_merchant_bonus

    def _auto_play_treasures(self, player_idx: int) -> None:
        """All Treasures in hand are played the instant the Buy phase starts.
        At this stage there's never a reason to hold one back, and always
        maximizing coins keeps the action space smaller (no more manual
        PLAY(Copper) x N) for a policy that only needs to decide what to buy."""
        player = self.players[player_idx]
        for name in list(player.hand):
            if CardType.TREASURE in self.cards[name].types:
                self.play_treasure(player_idx, name)

    def _buy_card(self, card_name: str) -> None:
        player = self.players[self.current_player]
        card = self.cards[card_name]
        self.supply[card_name] -= 1
        player.coins -= card.cost
        player.buys -= 1
        player.discard.append(card_name)

    def _cleanup_and_advance(self) -> None:
        player = self.players[self.current_player]
        player.discard.extend(player.play_area)
        player.play_area = []
        player.discard.extend(player.set_aside)
        player.set_aside = []
        player.discard.extend(player.hand)
        player.hand = []
        player.draw(STARTING_HAND, self.rng)
        player.actions = 0
        player.buys = 0
        player.coins = 0
        player.turns_taken += 1

        self.current_player = (self.current_player + 1) % len(self.players)
        self.turn_number += 1
        self.turn_merchant_bonus = 0
        self.turn_silver_played = False
        self.phase = Phase.ACTION

        next_player = self.players[self.current_player]
        next_player.actions = 1
        next_player.buys = 1

    def _maybe_end_game(self) -> None:
        if self.phase == Phase.GAME_OVER:
            return
        if self.supply.get("Province", 0) == 0:
            self.phase = Phase.GAME_OVER
            return
        empty_piles = sum(1 for count in self.supply.values() if count == 0)
        if empty_piles >= 3:
            self.phase = Phase.GAME_OVER
