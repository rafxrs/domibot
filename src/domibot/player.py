from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class PlayerState:
    name: str
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    play_area: list[str] = field(default_factory=list)
    # Cards temporarily out of all normal zones mid-resolution (Library).
    set_aside: list[str] = field(default_factory=list)

    actions: int = 0
    buys: int = 0
    coins: int = 0
    turns_taken: int = 0

    def draw(self, n: int, rng: random.Random) -> list[str]:
        drawn: list[str] = []
        for _ in range(n):
            if not self.deck:
                if not self.discard:
                    break
                self.deck = self.discard
                self.discard = []
                rng.shuffle(self.deck)
            card = self.deck.pop()
            self.hand.append(card)
            drawn.append(card)
        return drawn

    def all_cards(self) -> list[str]:
        return self.deck + self.hand + self.discard + self.play_area + self.set_aside

    def total_cards(self) -> int:
        """Every card this player owns, in any zone -- NOT the draw pile
        (that is `len(self.deck)`). Named `deck_size` until it turned out
        the RL encoder was reading it as a draw-pile count."""
        return len(self.all_cards())

    def victory_points(self, card_lookup) -> int:
        total = 0
        for name in self.all_cards():
            card = card_lookup[name]
            v = card.vp_value
            total += v(self) if callable(v) else v
        return total

    def clone(self) -> "PlayerState":
        return PlayerState(
            name=self.name,
            deck=list(self.deck),
            hand=list(self.hand),
            discard=list(self.discard),
            play_area=list(self.play_area),
            set_aside=list(self.set_aside),
            actions=self.actions,
            buys=self.buys,
            coins=self.coins,
            turns_taken=self.turns_taken,
        )
