"""A fixed rule for card-effect choices, used by the scripted agents."""
from __future__ import annotations

from domibot import Action, DecisionKind, Game
from domibot.models import NO, NO_REVEAL, REVEAL_MOAT

_JUNK_PRIORITY = ["Curse", "Estate", "Copper"]  # what to give up first when forced to


def heuristic_reaction(game: Game) -> Action:
    """Answer the pending decision: reveal Moat, say no, gain the most expensive
    card, otherwise give up junk first and decline an optional give-up of
    anything better."""
    decision = game.pending_decision
    actions = game.legal_actions()
    if decision.kind == DecisionKind.REACT:
        return REVEAL_MOAT if REVEAL_MOAT in actions else NO_REVEAL
    if decision.kind == DecisionKind.YES_NO:
        return NO if NO in actions else actions[0]

    card_actions = [a for a in actions if a.card is not None]
    decline_actions = [a for a in actions if a.card is None]  # DONE / NONE
    if not card_actions:
        return actions[0]
    if card_actions[0].verb == "GAIN":
        return max(card_actions, key=lambda a: game.cards[a.card].cost)

    def rank(a: Action) -> int:
        return _JUNK_PRIORITY.index(a.card) if a.card in _JUNK_PRIORITY else len(_JUNK_PRIORITY)

    best = min(card_actions, key=rank)
    if decline_actions and rank(best) == len(_JUNK_PRIORITY):
        return decline_actions[0]
    return best
