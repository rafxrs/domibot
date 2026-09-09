"""Fixed, non-learned resolution of card-effect sub-decisions (discard,
trash, topdeck, react-to-attack, optional yes/no).

BigMoneyAgent always uses this. DomibotAgent/self-play use it only as an
ablation/fallback (`search_sub_decisions=False`) -- by default they search
and learn these sub-decisions via MCTS instead, same as phase actions; see
mcts.py's module docstring for how.
"""
from __future__ import annotations

from domibot import Action, DecisionKind, Game
from domibot.models import NO, NO_REVEAL, REVEAL_MOAT

# Cards ranked worst-to-best to give up when forced to (Militia's discard,
# Sentry's trash/discard, ...). No card-specific strategy behind this, just "keep the good stuff."
_JUNK_PRIORITY = ["Curse", "Estate", "Copper"]


def heuristic_reaction(game: Game) -> Action:
    """Resolve whatever the current pending Decision is. Only valid to call
    when `game.pending_decision is not None`."""
    decision = game.pending_decision
    actions = game.legal_actions()

    if decision.kind == DecisionKind.REACT:
        return REVEAL_MOAT if REVEAL_MOAT in actions else NO_REVEAL
    if decision.kind == DecisionKind.YES_NO:
        return NO if NO in actions else actions[0]

    card_actions = [a for a in actions if a.card is not None]
    decline_actions = [a for a in actions if a.card is None]  # DONE / NONE
    if not card_actions:
        return actions[0]  # nothing left worth giving up

    if card_actions[0].verb == "GAIN":
        # Gaining is the opposite of giving something up -- take the most
        # expensive card on offer (never Curse, cost 0, unless it's
        # somehow the only legal option) rather than reusing the
        # give-up-your-worst-card ranking below.
        return max(card_actions, key=lambda a: game.cards[a.card].cost)

    def rank(a: Action) -> int:
        try:
            return _JUNK_PRIORITY.index(a.card)
        except ValueError:
            return len(_JUNK_PRIORITY)  # an unrecognized (kingdom) card: give up last

    best = min(card_actions, key=rank)
    if decline_actions and rank(best) == len(_JUNK_PRIORITY):
        # An *optional* give-up (Chapel, Cellar, ...) where nothing on
        # offer is actual junk -- decline rather than trash/discard a
        # perfectly good Silver/Gold/kingdom card just because asked.
        return decline_actions[0]
    return best


def advance_to_next_phase_action(game: Game, action: Action) -> None:
    """Apply `action`, then keep resolving whatever forced sub-decisions
    follow from it (via `heuristic_reaction`) until control returns to a
    plain phase-action choice (`pending_decision is None`) or the game
    ends. Mutates `game` in place."""
    game.step(action)
    while (not game.is_game_over()) and game.pending_decision is not None:
        game.step(heuristic_reaction(game))
