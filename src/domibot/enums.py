from enum import Enum, auto


class CardType(Enum):
    TREASURE = auto()
    VICTORY = auto()
    CURSE = auto()
    ACTION = auto()
    REACTION = auto()
    ATTACK = auto()


class Phase(Enum):
    ACTION = auto()
    BUY = auto()
    CLEANUP = auto()
    GAME_OVER = auto()


class DecisionKind(Enum):
    """Classifies the pending Decision so a future policy/network can condition
    on *why* a set of actions is legal, since the same Action verb (e.g. YES/NO)
    is reused across semantically different card effects."""

    PHASE_ACTION = auto()  # top-level: play a card, buy a card, end phase
    SELECT_CARD = auto()  # pick zero-or-more cards from an offered list
    YES_NO = auto()
    REACT = auto()  # reveal a reaction (Moat) to block an attack, or decline
