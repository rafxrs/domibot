from .basic import BASIC_CARDS
from .kingdom import KINGDOM_CARDS

ALL_CARDS = {**BASIC_CARDS, **KINGDOM_CARDS}

__all__ = ["BASIC_CARDS", "KINGDOM_CARDS", "ALL_CARDS"]
