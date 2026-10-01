"""Rebuild a `Game` from what a player can see at a real table, for the move advisor
(`examples/domibot_relay.py`).

Your own cards are known exactly. The opponent's total ownership is exact by
elimination (every card is in the supply, the trash, or one of the two
players); whatever of it isn't in their visible discard or play area is split
at random into a hand and draw pile of the reported sizes, and your own draw
pile is shuffled. Cards known to be on top of a deck or in the opponent's hand
are placed there.

`reconstruct_game` gives a phase-action boundary. Choices inside a card's
effect are reached by replaying a path from a boundary (`mcts.materialize`):
`reconstruct_opponent_turn_boundary` starts at the opponent's turn so their
attack can be replayed, and `replay_open_play` replays your own last Action.
"""
from __future__ import annotations

import dataclasses
import random
from collections import Counter
from dataclasses import dataclass, field

from domibot import ALL_CARDS, Action, CardType, Game
from domibot.enums import DecisionKind, Phase
from domibot.models import DONE

from .log_parser import remove_one

# Short codes for the kingdom cards: 2-3 letters, 3 where several share a prefix.
CARD_ABBREVIATIONS: dict[str, str] = {
    "ART": "Artisan", "BAN": "Bandit", "BUR": "Bureaucrat", "CEL": "Cellar", "CHA": "Chapel", "CR": "Council Room",
    "FES": "Festival", "GAR": "Gardens", "HAR": "Harbinger", "LAB": "Laboratory", "LIB": "Library", "MAR": "Market",
    "MER": "Merchant", "MIL": "Militia", "MIN": "Mine", "MOA": "Moat", "MLR": "Moneylender", "POA": "Poacher",
    "REM": "Remodel", "SEN": "Sentry", "SMI": "Smithy", "TR": "Throne Room", "VAS": "Vassal", "VIL": "Village",
    "WIT": "Witch", "WOR": "Workshop",
}


def resolve_card_name(token: str) -> str:
    """A card name or abbreviation, case-insensitive, as its full name."""
    if token in ALL_CARDS:
        return token
    if token.upper() in CARD_ABBREVIATIONS:
        return CARD_ABBREVIATIONS[token.upper()]
    for name in ALL_CARDS:
        if name.lower() == token.lower():
            return name
    raise ValueError(f"not a recognized card name or abbreviation: {token!r}")


@dataclass
class TableState:
    """Everything one query needs, as it appears on screen."""

    kingdom: list[str]
    supply: dict[str, int]
    trash: list[str]
    my_hand: list[str]
    my_discard: list[str]
    my_play_area: list[str] = field(default_factory=list)
    my_total: list[str] = field(default_factory=list)  # every card you own, any zone
    my_actions: int = 1
    my_buys: int = 1
    my_coins: int = 0
    my_phase: str = "ACTION"  # "ACTION" or "BUY"
    my_turns_taken: int = 0  # completed turns before this one
    opp_discard: list[str] = field(default_factory=list)
    opp_play_area: list[str] = field(default_factory=list)
    opp_hand_size: int = 5
    opp_draw_pile_size: int = 5
    my_deck_top: list[str] = field(default_factory=list)  # known top cards, next draw first
    opp_deck_top: list[str | None] = field(default_factory=list)  # None: a card known to be there, unnamed
    opp_known_hand: list[str] = field(default_factory=list)  # e.g. a Moat they reacted with
    my_merchant_bonus: int = 0  # Merchants played this turn
    my_silver_played: bool = False
    my_bought: bool = False  # bought this turn: treasures left in hand stay unplayed


def table_state(parsed, kingdom: list[str], supply: dict[str, int] | None = None) -> TableState:
    """A `TableState` from a `log_parser.ParsedLog` (or None), with a fresh-turn default
    for anything the log couldn't derive."""
    if parsed is None:
        return TableState(kingdom=kingdom, supply=supply or {}, trash=[], my_hand=[], my_discard=[])

    def get(value, default):
        return default if value is None else value

    phase = get(parsed.my_phase, "ACTION")
    return TableState(
        kingdom=kingdom, supply=supply or parsed.supply, trash=parsed.trash, my_hand=get(parsed.my_hand, []),
        my_discard=get(parsed.my_discard, []), my_play_area=get(parsed.my_play_area, []), my_total=parsed.my_total,
        my_actions=get(parsed.my_actions, 1 if phase == "ACTION" else 0), my_buys=get(parsed.my_buys, 1),
        my_coins=get(parsed.my_coins, 0), my_phase=phase, my_turns_taken=parsed.my_turns_taken,
        opp_discard=get(parsed.opp_discard, []), opp_play_area=get(parsed.opp_play_area, []),
        opp_hand_size=get(parsed.opp_hand_size, 5), opp_draw_pile_size=get(parsed.opp_draw_pile_size, 5),
        my_deck_top=get(parsed.my_deck_top, []), opp_deck_top=get(parsed.opp_deck_top, []),
        opp_known_hand=get(parsed.opp_known_hand, []), my_merchant_bonus=get(parsed.my_merchant_bonus, 0),
        my_silver_played=get(parsed.my_silver_played, False), my_bought=get(parsed.my_bought, False))


def _over_accounted(known: Counter, total: Counter) -> Counter:
    """Cards `known` holds more copies of than `total` (subtract keeps negatives, unlike `-`)."""
    diff = known.copy()
    diff.subtract(total)
    return +diff


def _with_known_top(deck: list[str], top: list[str | None]) -> list[str] | None:
    """`deck` with `top` (next draw first; None = any card) moved to its top, the
    end of the list; None if `deck` lacks a named card."""
    rest = list(deck)
    placed: list[str | None] = []
    for card in top:
        if card is not None and not remove_one(rest, card):
            return None
        placed.append(card)
    for i, card in enumerate(placed):
        if card is None:
            if not rest:
                return None
            placed[i] = rest.pop()
    return rest + placed[::-1]


def reconstruct_game(state: TableState, seed: int | None = None) -> Game:
    """A `Game` at your current phase decision, consistent with `state`. Raises
    `ValueError` naming any count that doesn't add up (usually a typo). In the Buy
    phase, Treasures still in hand are played first, as the engine does, unless you
    already bought."""
    game = Game(state.kingdom, num_players=2, seed=seed)
    rng = random.Random(seed)
    total = Counter(game.supply)  # every copy in the game, from a fresh setup
    for p in game.players:
        total.update(p.all_cards())
    game.supply = dict(state.supply)
    game.trash = list(state.trash)

    me = game.players[0]
    me.hand, me.discard, me.play_area, me.set_aside = list(state.my_hand), list(state.my_discard), \
        list(state.my_play_area), []
    my_total = Counter(state.my_total)
    my_known = Counter(me.hand) + Counter(me.discard) + Counter(me.play_area)
    if bad := _over_accounted(my_known, my_total):
        raise ValueError(f"my_total doesn't include enough copies of {dict(bad)} to cover my_hand/my_discard/"
                         f"my_play_area -- my_total should list *everything* you currently own, any zone")
    deck = my_total.copy()
    deck.subtract(my_known)
    me.deck = list(deck.elements())
    rng.shuffle(me.deck)
    if state.my_deck_top:  # a top that doesn't fit means the tracking is off: keep the plain shuffle
        me.deck = _with_known_top(me.deck, state.my_deck_top) or me.deck
    me.actions, me.buys, me.coins, me.turns_taken = state.my_actions, state.my_buys, state.my_coins, \
        state.my_turns_taken

    opp = game.players[1]
    opp.discard, opp.play_area, opp.set_aside = list(state.opp_discard), list(state.opp_play_area), []
    opp_total = total.copy()
    opp_total.subtract(Counter(game.supply))
    opp_total.subtract(Counter(game.trash))
    opp_total.subtract(my_total)
    if bad := -opp_total:
        raise ValueError(f"supply + trash + my_total account for {dict(bad)} more copies than exist in the game "
                         f"-- double check them against what's on screen")
    opp_known = Counter(opp.discard) + Counter(opp.play_area)
    if bad := _over_accounted(opp_known, opp_total):
        raise ValueError(f"opp_discard/opp_play_area claim {dict(bad)} more copies than the opponent could "
                         f"possibly own by elimination -- double check my_total and the supply/trash counts")
    hidden = opp_total.copy()
    hidden.subtract(opp_known)
    opp_hidden = list(hidden.elements())
    if len(opp_hidden) != state.opp_hand_size + state.opp_draw_pile_size:
        raise ValueError(
            f"the opponent must be holding {len(opp_hidden)} unseen cards by elimination (their total minus "
            f"their visible discard/play area), but you reported opp_hand_size={state.opp_hand_size} + "
            f"opp_draw_pile_size={state.opp_draw_pile_size} = {state.opp_hand_size + state.opp_draw_pile_size} "
            f"-- check those two, and also my_total (an error there shows up here)")
    rng.shuffle(opp_hidden)
    # Known hand cards and deck-top cards are placed; knowledge that doesn't fit the counts is dropped.
    pool, known_hand = list(opp_hidden), list(state.opp_known_hand)
    if len(known_hand) > state.opp_hand_size or not all(remove_one(pool, c) for c in known_hand):
        pool, known_hand = list(opp_hidden), []
    top = list(state.opp_deck_top) if len(state.opp_deck_top) <= state.opp_draw_pile_size else []
    top_named = [c for c in top if c is not None]
    before_top = list(pool)
    if not all(remove_one(pool, c) for c in top_named):
        pool, top, top_named = before_top, [], []
    n_random = state.opp_hand_size - len(known_hand)
    opp.hand = known_hand + pool[:n_random]
    opp.deck = pool[n_random:] + top_named
    if top:
        opp.deck = _with_known_top(opp.deck, top) or opp.deck
    opp.actions = opp.buys = opp.coins = 0
    opp.turns_taken = max(state.my_turns_taken - 1, 0)

    game.current_player = 0
    game.turn_number = state.my_turns_taken + 1
    game.phase = Phase[state.my_phase]
    game.turn_merchant_bonus = state.my_merchant_bonus
    game.turn_silver_played = state.my_silver_played
    game.pending_decision = game.pending_gen = None
    game.action_log = []
    if game.phase == Phase.BUY and not state.my_bought:
        for name in list(me.hand):
            if CardType.TREASURE in game.cards[name].types:
                game.play_treasure(0, name)
    return game


def reconstruct_opponent_turn_boundary(
    state: TableState, opp_turn_start_discard: list[str], path: list[Action] | None = None,
    seed: int | None = None, my_deck_top: list[str] | None = None, opp_gains: list[str] | None = None,
) -> Game:
    """Like `reconstruct_game`, but at the start of the opponent's open turn, so their
    plays so far (`path`) can be replayed up to a reaction pending on you.

    Only valid for a path of `log_parser._SAFE_MIDTURN_ACTION_CARDS`, which move no
    cards of yours and gain only their attack's own card (`opp_gains`, put back in
    the supply here since the replay gains them again). At any turn start the
    opponent's play area is empty and their hand is 5, so only their discard
    (`opp_turn_start_discard`) and your deck top (`my_deck_top`) differ from now.
    The cards `path` plays are moved into their reconstructed hand.
    """
    opp_gains = list(opp_gains or [])
    opp_total_now = state.opp_hand_size + state.opp_draw_pile_size + len(state.opp_discard) + \
        len(state.opp_play_area)
    supply = dict(state.supply)
    for card in opp_gains:
        supply[card] += 1
    start = dataclasses.replace(
        state, supply=supply, opp_discard=opp_turn_start_discard, opp_play_area=[], opp_hand_size=5,
        opp_draw_pile_size=opp_total_now - len(opp_gains) - len(opp_turn_start_discard) - 5,
        my_deck_top=list(my_deck_top or []), opp_deck_top=[], opp_known_hand=[], my_merchant_bonus=0,
        my_silver_played=False)
    game = reconstruct_game(start, seed=seed)
    game.current_player, game.phase = 1, Phase.ACTION
    opp = game.players[1]
    opp.actions = opp.buys = 1

    needed = Counter(action.card for action in path or [])
    for card, count in needed.items():
        while opp.hand.count(card) < count:
            if card not in opp.deck:
                raise ValueError(f"path plays {card!r} x{count}, but the opponent's reconstructed turn-start "
                                 f"hand+deck don't contain that many -- their total ownership must be wrong")
            spare = next((c for c in opp.hand if opp.hand.count(c) > needed[c]), None)
            if spare is None:
                raise ValueError("the opponent's path plays more cards than a 5-card hand holds")
            opp.hand.remove(spare)
            opp.deck.remove(card)
            opp.deck.append(spare)
            opp.hand.append(card)
    random.Random(seed).shuffle(opp.deck)
    return game


@dataclass
class OpenPlayResult:
    """`status`: "pending" (a choice waits on you: search `boundary` + `path`),
    "resolved" (the card finished), "opponent" (it waits on them), or
    "unsupported" (the log can't be replayed exactly; see `reason`)."""
    status: str
    boundary: Game | None = None
    path: list[Action] = field(default_factory=list)
    reason: str = ""


_CHOICE_VERBS = {"trash": "TRASH", "discard": "DISCARD", "gain": "GAIN", "topdeck": "TOPDECK", "play": "PLAY"}
_NONE = Action("NONE")


def _cards_off_my_deck(events) -> list[str]:
    """Cards your events take off your deck, next draw first: draws, looks and reveals
    (not Harbinger's, which is the discard) and Vassal's discard, minus any a topdeck
    in the same events had put back."""
    put_back: list[str] = []
    taken: list[str] = []
    for e in events:
        if not e.mine:
            continue
        if e.kind == "topdeck":
            put_back = list(e.cards) + put_back
            continue
        if e.kind == "draw" or (e.kind in ("look", "reveal") and e.context != "Harbinger") or \
                (e.kind == "discard" and e.context == "Vassal"):
            taken += [c for c in e.cards if not remove_one(put_back, c)]
    return taken


def replay_open_play(open_play, kingdom: list[str], final_my_hand: list[str],
                     seed: int | None = None) -> OpenPlayResult:
    """Replay `open_play` (a `log_parser.OpenPlay`) through the engine, applying each
    choice the log shows, to find whether a choice of that card still waits on you.

    Your deck is stacked with the cards the log shows it giving up, across a
    reshuffle partway if needed, so the engine draws what you drew. One logged
    multi-card line (Chapel trashing 3) is one confirmed selection; whatever the
    engine still offers after it is declined. `final_my_hand` checks the result.
    """
    events = open_play.events
    shuffles = [i for i, e in enumerate(events) if e.mine and e.kind == "shuffle"]
    if len(shuffles) > 1:
        return OpenPlayResult("unsupported", reason="your deck was reshuffled twice partway through it")
    b = open_play.boundary
    if shuffles:  # the shuffle line comes before the draw that spans it
        before = _cards_off_my_deck(events[:shuffles[0]])
        after = _cards_off_my_deck(events[shuffles[0] + 1:])
        top: list[str] = []
    else:
        before = after = []
        top = _cards_off_my_deck(events)
        known = b.my_deck_top or []
        if top[:len(known)] == known[:len(top)]:
            top = top + known[len(top):]
    try:
        boundary = reconstruct_game(dataclasses.replace(table_state(b, kingdom), my_deck_top=top), seed=seed)
    except ValueError as e:
        return OpenPlayResult("unsupported", reason=f"the state before it doesn't add up ({e})")
    me = boundary.players[0]
    if shuffles:
        # The deck as it will be right after the shuffle: cards taken before it, then the old deck's
        # leftovers and the rest of later draws from the old discard. Cards discarded partway stay in
        # the discard here instead of the new deck -- same cards, different pile.
        leftover = Counter(me.deck)
        leftover.subtract(Counter(before))
        from_discard, pool = list(after), list(me.discard)
        if (-leftover or not all(remove_one(from_discard, c) for c in leftover.elements())
                or not all(remove_one(pool, c) for c in from_discard)):
            return OpenPlayResult("unsupported", reason="the cards drawn around its reshuffle don't match your "
                                                        "deck and discard")
        random.Random(seed).shuffle(pool)
        me.deck = pool + from_discard[::-1] + list(leftover.elements())[::-1] + before[::-1]
        me.discard = []
    elif top and me.deck[-len(top):][::-1] != top[:len(me.deck)]:
        return OpenPlayResult("unsupported", reason="the cards it drew aren't all in your deck by elimination")

    path = [Action("PLAY", open_play.card)]
    game = boundary.clone()

    def step(action: Action) -> None:
        game.step(action)
        path.append(action)

    def unsupported(reason: str) -> OpenPlayResult:
        return OpenPlayResult("unsupported", reason=reason)

    try:
        game.step(path[0])
        i = 1
        while i < len(events):
            pending = game.pending_decision
            if pending is None:
                break  # the card finished; later lines are ordinary play
            if pending.player != 0:
                return OpenPlayResult("resolved" if any(not e.mine for e in events[i:]) else "opponent",
                                      boundary, path)
            e = events[i]
            if not e.mine or e.kind in ("draw", "look", "reveal", "gets", "shuffle", "other") or \
                    (e.kind == "play" and e.again):
                i += 1
                continue
            if e.kind == "discard" and e.context == "Vassal":
                revealed = game.players[0].set_aside[-1] if game.players[0].set_aside else None
                if pending.kind == DecisionKind.YES_NO and pending.source_card == "Vassal" and e.cards == [revealed]:
                    nxt = next((j for j in range(i + 1, len(events)) if events[j].mine and events[j].kind != "gets"),
                               None)
                    if nxt is None:
                        break  # play it or not: exactly what waits on you
                    if events[nxt].kind == "play" and not events[nxt].again and events[nxt].cards == e.cards:
                        step(Action("YES"))
                        i = nxt + 1
                    else:
                        step(Action("NO"))
                        i += 1
                    continue
                i += 1  # a Vassal discard of a non-Action: nothing to decide
                continue
            if pending.kind == DecisionKind.YES_NO:
                if not any(ev.mine for ev in events[i:]):
                    break
                return unsupported(f"{pending.source_card}'s yes/no choice can't be read from the log yet")
            verb = _CHOICE_VERBS.get(e.kind)
            if verb is None:  # a treasure or buy: an optional choice still open was declined
                if DONE in pending.options:
                    step(DONE)
                elif _NONE in pending.options:
                    step(_NONE)
                else:
                    return unsupported(f"the log moved on while the engine was asking: {pending.prompt}")
                continue
            for card in e.cards:
                action = Action(verb, card)
                while game.pending_decision is not None and action not in game.pending_decision.options:
                    options = game.pending_decision.options
                    if game.pending_decision.player != 0 or not (DONE in options or _NONE in options):
                        return unsupported(f"the log shows {action} but the engine is asking: "
                                           f"{game.pending_decision.prompt}")
                    step(DONE if DONE in options else _NONE)
                if game.pending_decision is None:
                    return unsupported(f"the log shows {action} after the card finished")
                step(action)
            after = game.pending_decision  # the line was one confirmed selection: close it
            if after is not None and after.player == 0 and DONE in after.options and \
                    all(o.verb in (verb, "DONE") for o in after.options):
                step(DONE)
            i += 1
    except ValueError as e:
        return unsupported(f"the engine rejected a replayed step ({e})")

    if game.pending_decision is None:
        return OpenPlayResult("resolved", boundary, path)
    if game.pending_decision.player != 0:
        return OpenPlayResult("opponent", boundary, path)
    if Counter(game.players[0].hand) != Counter(final_my_hand):
        return OpenPlayResult("unsupported", boundary, path, reason="the replayed hand doesn't match the log's")
    return OpenPlayResult("pending", boundary, path)
