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
    """Why a set of actions is legal: the same verb (YES/NO) serves different effects."""

    PHASE_ACTION = auto()  # play a card, buy a card, end a phase
    SELECT_CARD = auto()  # pick a card from an offered list
    YES_NO = auto()
    REACT = auto()  # reveal a reaction (Moat) to block an attack, or decline
