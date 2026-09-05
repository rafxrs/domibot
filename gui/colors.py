"""Flat color-coding for cards -- no artwork, just a color by card type
(per the base-set kingdom + basic cards), with a few named-card overrides
since Treasures/Moat are visually distinct in real Dominion too."""
from __future__ import annotations

from domibot import ALL_CARDS, CardType

BACKGROUND = (34, 100, 60)          # felt-table green, like dominion.games
PANEL_BG = (24, 70, 42)
DECISION_PANEL_BG = (60, 60, 20)
TEXT = (20, 20, 20)
TEXT_LIGHT = (235, 235, 235)
BUTTON = (210, 210, 210)
BUTTON_HOVER = (255, 255, 255)
BORDER = (15, 15, 15)
DIM_OVERLAY = (0, 0, 0, 140)  # RGBA, for non-clickable cards

VICTORY = (144, 238, 144)   # light green
ACTION = (150, 150, 150)    # grey
REACTION_MOAT = (173, 216, 230)  # light blue
GOLD = (255, 215, 0)        # yellow
SILVER = (192, 192, 192)    # silver
COPPER = (184, 115, 51)     # bronze
CURSE = (106, 13, 173)      # purple (not specified by the ask; Dominion's own curse card is purple/black)
FALLBACK = (180, 180, 180)

_NAME_OVERRIDES = {
    "Moat": REACTION_MOAT,
    "Gold": GOLD,
    "Silver": SILVER,
    "Copper": COPPER,
    "Curse": CURSE,
}


def card_color(name: str) -> tuple[int, int, int]:
    if name in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[name]
    card = ALL_CARDS.get(name)
    if card is None:
        return FALLBACK
    if CardType.VICTORY in card.types:
        return VICTORY
    if CardType.ACTION in card.types:
        return ACTION
    return FALLBACK
