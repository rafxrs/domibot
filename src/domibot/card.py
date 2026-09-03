from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

from .enums import CardType
from .models import Action, Decision

# A card effect is a generator: it runs game/player mutations directly and
# `yield`s a Decision whenever it needs a player's input, resuming with the
# Action that player chose. Cards with sub-effects (Throne Room, Vassal)
# compose via `yield from` on another card's effect generator.
Effect = Callable[["object", int], Generator[Decision, Action, None]]


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    types: tuple[CardType, ...]

    # Treasure cards
    coin_value: int = 0

    # Victory / Curse cards. `vp_value` is either a flat int, or a callable
    # `(player_state) -> int` for cards like Gardens whose value depends on
    # deck size.
    vp_value: int | Callable[[object], int] = 0

    # Flat, unconditional bonuses applied the instant an Action card is
    # played (covers Village/Smithy/Market/Laboratory/Festival/etc. with no
    # extra generator needed).
    plus_cards: int = 0
    plus_actions: int = 0
    plus_buys: int = 0
    plus_coins: int = 0

    # Optional generator for anything beyond the flat bonuses above.
    effect: Optional[Effect] = None

    def is_type(self, t: CardType) -> bool:
        return t in self.types

    def __repr__(self) -> str:
        return self.name
