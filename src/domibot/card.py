from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Optional

from .enums import CardType
from .models import Action, Decision

# A card effect mutates the game directly and yields a Decision whenever it needs a
# player's input, resuming with the chosen Action. Throne Room and Vassal compose
# other effects with `yield from`.
Effect = Callable[["object", int], Generator[Decision, Action, None]]


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    types: tuple[CardType, ...]

    coin_value: int = 0
    vp_value: int | Callable[[object], int] = 0  # a callable for Gardens: (player_state) -> VP
    # Bonuses applied as an Action is played; `effect` does anything more.
    plus_cards: int = 0
    plus_actions: int = 0
    plus_buys: int = 0
    plus_coins: int = 0
    effect: Optional[Effect] = None

    def __repr__(self) -> str:
        return self.name
