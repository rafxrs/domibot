"""Reusable building blocks for card effect generators.

Every helper here either mutates game/player state directly, or is itself a
generator that `yield`s Decision objects and is meant to be driven with
`yield from` inside a card's effect generator.
"""
from __future__ import annotations

import inspect
from typing import Callable, Generator, Iterable, Optional

from .enums import DecisionKind
from .models import DONE, NO_REVEAL, REVEAL_MOAT, Action, Decision


def move(card: str, src: list[str], dst: list[str]) -> None:
    src.remove(card)
    dst.append(card)


def trash_from(game, card: str, src: list[str]) -> None:
    src.remove(card)
    game.trash.append(card)


def gain(game, player_idx: int, card_name: str, to: str = "discard") -> bool:
    """Gain a card from the supply into 'discard' | 'hand' | 'deck_top'.
    Returns False if the pile was already empty."""
    if game.supply.get(card_name, 0) <= 0:
        return False
    game.supply[card_name] -= 1
    p = game.players[player_idx]
    if to == "discard":
        p.discard.append(card_name)
    elif to == "hand":
        p.hand.append(card_name)
    elif to == "deck_top":
        p.deck.append(card_name)
    else:
        raise ValueError(to)
    return True


def choose_cards(
    game,
    player_idx: int,
    candidates: Iterable[str],
    verb: str,
    prompt: str,
    *,
    min_count: int = 0,
    max_count: Optional[int] = None,
) -> Generator[Decision, Action, list[str]]:
    """Repeatedly offer the remaining candidates (deduped by name, one option
    per distinct card name) plus a DONE option once `min_count` is met.
    Returns the list of chosen card names, in choice order."""
    chosen: list[str] = []
    remaining = list(candidates)
    limit = len(remaining) if max_count is None else max_count
    while remaining and len(chosen) < limit:
        distinct = list(dict.fromkeys(remaining))
        options = [Action(verb, c) for c in distinct]
        if len(chosen) >= min_count:
            options = options + [DONE]
        action = yield Decision(DecisionKind.SELECT_CARD, player_idx, prompt, options)
        if action == DONE:
            break
        remaining.remove(action.card)
        chosen.append(action.card)
    return chosen


def choose_one(
    game, player_idx: int, candidates: Iterable[str], verb: str, prompt: str, *, allow_none: bool = False
) -> Generator[Decision, Action, Optional[str]]:
    distinct = list(dict.fromkeys(candidates))
    options = [Action(verb, c) for c in distinct]
    if allow_none:
        options = options + [Action("NONE")]
    if not distinct:
        return None
    action = yield Decision(DecisionKind.SELECT_CARD, player_idx, prompt, options)
    return action.card


def choose_from_supply(
    game,
    player_idx: int,
    prompt: str,
    verb: str,
    predicate: Callable[[object], bool],
    *,
    allow_none: bool = True,
) -> Generator[Decision, Action, Optional[str]]:
    candidates = [name for name, count in game.supply.items() if count > 0 and predicate(game.cards[name])]
    options = [Action(verb, c) for c in candidates]
    if allow_none:
        options = options + [Action("NONE")]
    if not options:
        return None
    action = yield Decision(DecisionKind.SELECT_CARD, player_idx, prompt, options)
    return action.card


def yes_no(player_idx: int, prompt: str) -> Generator[Decision, Action, bool]:
    action = yield Decision(DecisionKind.YES_NO, player_idx, prompt, [Action("YES"), Action("NO")])
    return action.verb == "YES"


def attack_each_opponent(game, attacker_idx: int, per_opponent) -> Generator[Decision, Action, None]:
    """`per_opponent(game, opponent_idx)` runs for every opponent who doesn't
    reveal a Moat; it may be a plain function (e.g. Witch's curse-gain has no
    decision to make) or a generator (e.g. Bureaucrat's topdeck choice).
    Handles the Moat reaction itself."""
    for opp in game.other_players_in_order(attacker_idx):
        opp_state = game.players[opp]
        if "Moat" in opp_state.hand:
            attacker_name = game.players[attacker_idx].name
            action = yield Decision(
                DecisionKind.REACT,
                opp,
                f"{attacker_name} plays an Attack. Reveal Moat to block it?",
                [REVEAL_MOAT, NO_REVEAL],
            )
            if action == REVEAL_MOAT:
                continue
        result = per_opponent(game, opp)
        if inspect.isgenerator(result):
            yield from result


def peek_top(game, player) -> Optional[str]:
    """Look at (without removing) the top card of `player`'s deck, reshuffling
    discard into deck first if the deck is empty."""
    if not player.deck:
        if not player.discard:
            return None
        player.deck = player.discard
        player.discard = []
        game.rng.shuffle(player.deck)
    return player.deck[-1] if player.deck else None
