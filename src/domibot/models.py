from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from .enums import DecisionKind


class Action(NamedTuple):
    """An atomic, hashable choice. `card` is a card name, or None for
    verbs that don't target a card (END_ACTIONS, END_BUY, YES, NO, DONE)."""

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
    """One action taken during a game, for replay or saving to a log file.
    `hand` is a snapshot of the deciding player's hand immediately *before*
    `action` was taken, sorted for readability (hand order isn't meaningful
    in Dominion) — it's what the decision was actually made from."""

    turn: int
    player: int
    hand: tuple[str, ...]
    action: Action


@dataclass
class Decision:
    """A pending choice some player must make before the game can advance.
    `options` is the full set of legal Actions for this decision, computed
    up front so an agent (or random playout) never has to guess legality."""

    kind: DecisionKind
    player: int
    prompt: str
    options: list[Action] = field(default_factory=list)
