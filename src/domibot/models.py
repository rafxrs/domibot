from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from .enums import DecisionKind


class Action(NamedTuple):
    """A hashable choice: a verb and a card name, or None for verbs like DONE or END_BUY."""

    verb: str
    card: Optional[str] = None

    def __repr__(self) -> str:
        return self.verb if self.card is None else f"{self.verb}({self.card})"


DONE = Action("DONE")
YES = Action("YES")
NO = Action("NO")
END_ACTIONS = Action("END_ACTIONS")
END_BUY = Action("END_BUY")
REVEAL_MOAT = Action("REVEAL_MOAT")
NO_REVEAL = Action("NO_REVEAL")


class LogEntry(NamedTuple):
    """One action taken, with the decider's hand (sorted) just before it."""

    turn: int
    player: int
    hand: tuple[str, ...]
    action: Action


@dataclass
class Decision:
    """A pending choice; `options` are its legal Actions."""

    kind: DecisionKind
    player: int
    prompt: str
    options: list[Action] = field(default_factory=list)
    # The card whose effect raised it (Chapel's trash and Remodel's otherwise look alike).
    source_card: Optional[str] = None
