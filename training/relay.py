"""Reconstructs a `domibot.Game` from what's actually visible to a player
at the table during a real game (e.g. on dominion.games), so `DomibotAgent`
can be asked "what would you do here?" as a move advisor -- see
`examples/domibot_relay.py`. You play every move yourself; this only tells
you what Domibot recommends, using exactly the information available to a
player: your own hand and total card ownership exactly, everything else
only through what Dominion actually makes public (the supply, the trash,
discard piles, and your opponent's hand/draw-pile *sizes*, never their
contents).

Hidden information (the opponent's hand and draw-pile contents, and the
literal draw order of your own deck) is filled in by determinization:
- The opponent's *exact* total card ownership needs no guessing at all --
  every card is somewhere (the supply, the trash, or one of the two
  players), so `opponent_total = game_total - supply - trash - your_total`
  is exact arithmetic, not a guess.
- Whatever of that total isn't in their visible discard/play area is
  randomly split into a fake hand and draw pile matching the sizes you
  report, and your own draw pile (a known *set* of cards, since you know
  everything you've gained, but not the order they're stacked in) is
  randomly shuffled.

This is "perfect information Monte Carlo", a standard approximation for
search under hidden information -- not exact, but exactly as much as a
human opponent has to guess with.

Cards known to sit on top of a deck (a Sentry or Harbinger topdeck,
Bureaucrat's Silver) are placed there rather than shuffled in.

`reconstruct_game` only ever produces boundary states (`pending_decision is
None`). Choices made *inside* a card's effect are reached by replaying a
path of actions from such a boundary (`training.mcts.materialize`), two
ways:
- `reconstruct_opponent_turn_boundary` positions the Game at the
  *opponent's* turn start, so a `log_parser`-derived path of their plays so
  far can be replayed to reach a reaction pending on you (Militia's
  discard, Bureaucrat's topdeck, Bandit's trash, or revealing Moat -- see
  `log_parser._SUPPORTED_TERMINAL_ATTACKS`).
- `replay_open_play` replays your own most recent Action play, and every
  choice the log shows you making for it, to find a choice of yours that's
  still pending (what to trash with Chapel, what Throne Room plays, ...).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from domibot import ALL_CARDS, Action, CardType, Game
from domibot.enums import DecisionKind, Phase
from domibot.models import DONE

# Short codes for the 26 base-set kingdom cards, so you don't have to type
# the full name at every prompt (dominion.games' log never lists the
# kingdom, so this is entered by hand every game). 2 letters where that's
# unambiguous; 3 where several cards share a prefix (the Market/Merchant/
# Militia/Mine/Moat/Moneylender cluster all start with "M").
CARD_ABBREVIATIONS: dict[str, str] = {
    "ART": "Artisan",
    "BAN": "Bandit",
    "BUR": "Bureaucrat",
    "CEL": "Cellar",
    "CHA": "Chapel",
    "CR": "Council Room",
    "FES": "Festival",
    "GAR": "Gardens",
    "HAR": "Harbinger",
    "LAB": "Laboratory",
    "LIB": "Library",
    "MAR": "Market",
    "MER": "Merchant",
    "MIL": "Militia",
    "MIN": "Mine",
    "MOA": "Moat",
    "MLR": "Moneylender",
    "POA": "Poacher",
    "REM": "Remodel",
    "SEN": "Sentry",
    "SMI": "Smithy",
    "TR": "Throne Room",
    "VAS": "Vassal",
    "VIL": "Village",
    "WIT": "Witch",
    "WOR": "Workshop",
}


def resolve_card_name(token: str) -> str:
    """A full card name, unchanged; or an abbreviation from
    CARD_ABBREVIATIONS (case-insensitive) expanded to one. Raises
    ValueError naming the token otherwise."""
    if token in ALL_CARDS:
        return token
    expanded = CARD_ABBREVIATIONS.get(token.upper())
    if expanded is not None:
        return expanded
    # case-insensitive full-name match too (e.g. "copper", "COPPER")
    for name in ALL_CARDS:
        if name.lower() == token.lower():
            return name
    raise ValueError(f"not a recognized card name or abbreviation: {token!r}")


@dataclass
class TableState:
    """Everything needed for one query, straight off what's on screen."""

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
    my_turns_taken: int = 0  # completed turns *before* this one (0 on your first turn)

    opp_discard: list[str] = field(default_factory=list)
    opp_play_area: list[str] = field(default_factory=list)
    opp_hand_size: int = 5
    opp_draw_pile_size: int = 5

    # Cards known to be on top of each deck, next draw first; for the
    # opponent, None is a card known to be there without knowing which.
    my_deck_top: list[str] = field(default_factory=list)
    opp_deck_top: list[str | None] = field(default_factory=list)
    # Merchants you've played this turn, and whether you've played a
    # Silver yet (each Merchant adds $1 to the first Silver).
    my_merchant_bonus: int = 0
    my_silver_played: bool = False


def _over_accounted(known: Counter, total: Counter) -> Counter:
    """Cards where `known` claims more copies than `total` allows for --
    using Counter.subtract (not the `-` operator, which silently clips
    negative results at every step and would hide exactly the mismatches
    this is meant to catch)."""
    diff = known.copy()
    diff.subtract(total)
    return +diff  # unary + keeps only strictly-positive entries


def _remove_one(zone: list, card) -> bool:
    if card in zone:
        zone.remove(card)
        return True
    return False


def _with_known_top(deck: list[str], top: list[str | None]) -> list[str] | None:
    """`deck` (already shuffled) rearranged so its top matches `top` (next
    draw first; None = any card), or None if `deck` doesn't hold the named
    cards. The top of a deck is the end of the list (`deck.pop()` draws)."""
    rest = list(deck)
    placed: list[str | None] = []
    for card in top:
        if card is not None and not _remove_one(rest, card):
            return None
        placed.append(card)
    for i, card in enumerate(placed):
        if card is None:
            if not rest:
                return None
            placed[i] = rest.pop()
    return rest + placed[::-1]


def reconstruct_game(state: TableState, seed: int | None = None) -> Game:
    """Builds a `Game` positioned at your current phase-action decision,
    consistent with everything you reported. Raises `ValueError` (naming
    the mismatch) if the counts don't add up -- almost always a sign one
    of the inputs was mistyped, not a bug in this reconstruction, since
    the arithmetic it checks is exact.

    In the Buy phase, any Treasures still in your hand are played first,
    as the engine always plays them all at once."""
    game = Game(state.kingdom, num_players=2, seed=seed)
    rng = random.Random(seed)

    # Fixed for the whole game: every copy of every card is currently in
    # exactly one of {supply, trash, player 0, player 1} -- read off this
    # fresh game's own starting setup, before anything below overwrites it.
    total: Counter = Counter()
    for name, count in game.supply.items():
        total[name] += count
    for p in game.players:
        total.update(p.all_cards())

    game.supply = dict(state.supply)
    game.trash = list(state.trash)

    me = game.players[0]
    me.hand = list(state.my_hand)
    me.discard = list(state.my_discard)
    me.play_area = list(state.my_play_area)
    me.set_aside = []
    my_total_counter = Counter(state.my_total)
    my_known = Counter(me.hand) + Counter(me.discard) + Counter(me.play_area)
    bad = _over_accounted(my_known, my_total_counter)
    if bad:
        raise ValueError(
            f"my_total doesn't include enough copies of {dict(bad)} to cover my_hand/my_discard/"
            f"my_play_area -- my_total should list *everything* you currently own, any zone"
        )
    my_deck_counter = my_total_counter.copy()
    my_deck_counter.subtract(my_known)
    me.deck = list(my_deck_counter.elements())  # elements() already skips non-positive counts
    rng.shuffle(me.deck)  # the order is unknown, even to you, apart from any known top cards
    if state.my_deck_top:
        # A top that doesn't fit the deck means the tracking behind it is
        # off; fall back to a plain shuffle rather than refuse to advise.
        me.deck = _with_known_top(me.deck, state.my_deck_top) or me.deck
    me.actions = state.my_actions
    me.buys = state.my_buys
    me.coins = state.my_coins
    me.turns_taken = state.my_turns_taken

    opp = game.players[1]
    opp.discard = list(state.opp_discard)
    opp.play_area = list(state.opp_play_area)
    opp.set_aside = []
    opp_total_counter = total.copy()
    opp_total_counter.subtract(Counter(game.supply))
    opp_total_counter.subtract(Counter(game.trash))
    opp_total_counter.subtract(my_total_counter)
    bad = -opp_total_counter  # unary - keeps only strictly-negative entries (as positive magnitudes)
    if bad:
        raise ValueError(
            f"supply + trash + my_total account for {dict(bad)} more copies than exist in the game -- "
            f"double check them against what's on screen"
        )
    opp_known = Counter(opp.discard) + Counter(opp.play_area)
    bad = _over_accounted(opp_known, opp_total_counter)
    if bad:
        raise ValueError(
            f"opp_discard/opp_play_area claim {dict(bad)} more copies than the opponent could possibly "
            f"own by elimination -- double check my_total and the supply/trash counts"
        )
    opp_hidden_counter = opp_total_counter.copy()
    opp_hidden_counter.subtract(opp_known)
    opp_hidden = list(opp_hidden_counter.elements())
    reported_hidden = state.opp_hand_size + state.opp_draw_pile_size
    if len(opp_hidden) != reported_hidden:
        raise ValueError(
            f"the opponent must be holding {len(opp_hidden)} unseen cards by elimination "
            f"(their total minus their visible discard/play area), but you reported "
            f"opp_hand_size={state.opp_hand_size} + opp_draw_pile_size={state.opp_draw_pile_size} "
            f"= {reported_hidden} -- check those two, and also my_total (an error there shows up here, "
            f"not on your own side, since it throws off the opponent's total by elimination)"
        )
    rng.shuffle(opp_hidden)
    opp.hand = opp_hidden[: state.opp_hand_size]
    opp.deck = opp_hidden[state.opp_hand_size:]
    if state.opp_deck_top and len(state.opp_deck_top) <= state.opp_draw_pile_size:
        # Keep the named top cards out of the random hand, then stack them.
        known_top = [c for c in state.opp_deck_top if c is not None]
        pool = list(opp_hidden)
        if all(_remove_one(pool, c) for c in known_top):
            deck = _with_known_top(pool[state.opp_hand_size:] + known_top, state.opp_deck_top)
            if deck is not None:
                opp.hand = pool[: state.opp_hand_size]
                opp.deck = deck
    opp.actions = opp.buys = opp.coins = 0
    opp.turns_taken = max(state.my_turns_taken - 1, 0)  # never read for a non-perspective player; kept plausible

    game.current_player = 0
    game.turn_number = state.my_turns_taken + 1
    game.phase = Phase[state.my_phase]
    game.turn_merchant_bonus = state.my_merchant_bonus
    game.turn_silver_played = state.my_silver_played
    game.pending_decision = None
    game.pending_gen = None
    game.action_log = []
    if game.phase == Phase.BUY:
        for name in list(me.hand):
            if CardType.TREASURE in game.cards[name].types:
                game.play_treasure(0, name)
    return game


def reconstruct_opponent_turn_boundary(
    state: TableState, opp_turn_start_discard: list[str], path: list[Action] | None = None,
    seed: int | None = None, my_deck_top: list[str] | None = None, opp_gains: list[str] | None = None,
) -> Game:
    """Like reconstruct_game, but positions the Game at the START of the
    opponent's still-open turn (current_player=1, Phase.ACTION) instead of
    your own phase-action decision -- for replaying
    log_parser.ParsedLog.pending_reaction_path via training.mcts.materialize
    to reach whatever Decision (if any) is now pending on you (see
    log_parser._SUPPORTED_TERMINAL_ATTACKS).

    Only valid together with a path built from log_parser's
    _SAFE_MIDTURN_ACTION_CARDS whitelist: none of those cards trashes or
    topdecks anything, and the only gains among them are an attack's own
    (Bureaucrat's Silver, Bandit's Gold), passed as `opp_gains` and put back
    in the supply here since the replay gains them again. So every other
    field on `state` is safe to reuse as-is -- your own hand hasn't changed
    since it's not your turn yet, and the opponent's play_area/hand_size at
    *any* turn start are always [] and 5 (Game._cleanup_and_advance always
    empties play_area and draws back to a full hand). Only their *discard*
    can differ between "now" and "turn start" (whatever they've bought/
    gained this turn already sits there), hence the separate
    `opp_turn_start_discard` -- a snapshot taken the moment their turn
    began, before any of that.

    `my_deck_top` is the top of your deck as it was at their turn start --
    e.g. the cards their Bandit has since revealed, so the replay reveals
    those same cards.

    `path` is that same list of plays: pass it here too (not just to
    materialize/run_mcts afterward) so the cards it names are forced into
    the opponent's *reconstructed* turn-start hand rather than left to
    chance. Their hidden hand/deck split is otherwise random (perfect
    information Monte Carlo, see the module docstring) -- but we don't
    actually need to guess whether they held e.g. Militia, we *know* they
    did, they just played it. Without this, `materialize` would raise
    "illegal action" any time the random split happened to put a known
    play in their deck instead of their hand."""
    opp_gains = list(opp_gains or [])
    opp_total_now = (state.opp_hand_size + state.opp_draw_pile_size
                      + len(state.opp_discard) + len(state.opp_play_area))
    supply = dict(state.supply)
    for card in opp_gains:
        supply[card] += 1
    turn_start_state = TableState(
        kingdom=state.kingdom, supply=supply, trash=state.trash,
        my_hand=state.my_hand, my_discard=state.my_discard,
        my_play_area=state.my_play_area, my_total=state.my_total,
        my_actions=state.my_actions, my_buys=state.my_buys, my_coins=state.my_coins,
        my_phase=state.my_phase, my_turns_taken=state.my_turns_taken,
        opp_discard=opp_turn_start_discard, opp_play_area=[],
        opp_hand_size=5,
        opp_draw_pile_size=opp_total_now - len(opp_gains) - len(opp_turn_start_discard) - 5,
        my_deck_top=list(my_deck_top or []),
    )
    game = reconstruct_game(turn_start_state, seed=seed)  # reuses its own arithmetic/validation
    game.current_player = 1
    game.phase = Phase.ACTION
    game.players[1].actions = 1
    game.players[1].buys = 1

    opp = game.players[1]
    rng = random.Random(seed)
    needed = Counter(action.card for action in path or [])
    for card, count in needed.items():
        while opp.hand.count(card) < count:
            if card not in opp.deck:
                raise ValueError(
                    f"path plays {card!r} x{count}, but the opponent's reconstructed turn-start hand+deck "
                    f"don't contain that many -- their total ownership (by elimination) must be wrong"
                )
            # Make room by moving out a hand card the path doesn't need.
            spare = next((c for c in opp.hand if opp.hand.count(c) > needed[c]), None)
            if spare is None:
                raise ValueError("the opponent's path plays more cards than a 5-card hand holds")
            opp.hand.remove(spare)
            opp.deck.remove(card)
            opp.deck.append(spare)
            opp.hand.append(card)
    rng.shuffle(opp.deck)
    return game


@dataclass
class OpenPlayResult:
    """What `replay_open_play` found about your most recent Action play:
    - "pending": one of its choices is waiting on you -- search it with
      `run_mcts(boundary, ..., path=path)`.
    - "resolved": it finished; the parsed state is a plain phase decision.
    - "opponent": it's waiting on the opponent (e.g. their Militia discard).
    - "unsupported": the log can't be replayed exactly (`reason` says why);
      whether a choice is pending is unknown."""
    status: str
    boundary: Game | None = None
    path: list[Action] = field(default_factory=list)
    reason: str = ""


_CHOICE_VERBS = {"trash": "TRASH", "discard": "DISCARD", "gain": "GAIN", "topdeck": "TOPDECK", "play": "PLAY"}
_NONE = Action("NONE")


def _cards_off_my_deck(events) -> list[str]:
    """The cards your events take off your deck, next draw first -- drawn,
    looked at (not Harbinger's, which looks through the discard), or
    discarded by Vassal -- skipping any that a topdeck during the same
    events had just put back."""
    put_back: list[str] = []
    taken: list[str] = []
    for e in events:
        if not e.mine:
            continue
        if e.kind == "topdeck":
            put_back = list(e.cards) + put_back
            continue
        from_deck = (e.kind == "draw" or (e.kind in ("look", "reveal") and e.context != "Harbinger")
                     or (e.kind == "discard" and e.context == "Vassal"))
        if not from_deck:
            continue
        for card in e.cards:
            if not _remove_one(put_back, card):
                taken.append(card)
    return taken


def replay_open_play(open_play, kingdom: list[str], final_my_hand: list[str],
                     seed: int | None = None) -> OpenPlayResult:
    """Replays `open_play` (a `log_parser.OpenPlay`: your most recent
    phase-level Action play and every log line since) through the engine,
    applying each choice the log shows you making, to find whether one of
    that card's choices is still waiting on you.

    Your deck is stacked with exactly the cards the log shows it giving up
    (draws, Sentry/Library looks, Vassal's discard), so the engine draws
    what you actually drew. A multi-card choice the log shows (Chapel
    trashing 3 cards) is one confirmed selection: whatever the engine still
    offers afterward is declined (DONE/NONE). `final_my_hand` (the hand
    parsed from the whole log) must match the replay's, as a check."""
    events = open_play.events
    if any(e.mine and e.kind == "shuffle" for e in events):
        return OpenPlayResult("unsupported", reason="your deck was reshuffled partway through it")
    b = open_play.boundary
    top = _cards_off_my_deck(events)
    known = b.my_deck_top or []
    if top[:len(known)] == known[:len(top)]:
        top = top + known[len(top):]
    state = TableState(
        kingdom=kingdom, supply=b.supply, trash=b.trash,
        my_hand=b.my_hand, my_discard=b.my_discard, my_play_area=b.my_play_area, my_total=b.my_total,
        my_actions=b.my_actions, my_buys=b.my_buys, my_coins=b.my_coins, my_phase=b.my_phase,
        my_turns_taken=b.my_turns_taken, opp_discard=b.opp_discard, opp_play_area=b.opp_play_area,
        opp_hand_size=b.opp_hand_size, opp_draw_pile_size=b.opp_draw_pile_size,
        my_deck_top=top, opp_deck_top=b.opp_deck_top or [],
        my_merchant_bonus=b.my_merchant_bonus or 0, my_silver_played=bool(b.my_silver_played),
    )
    try:
        boundary = reconstruct_game(state, seed=seed)
    except ValueError as e:
        return OpenPlayResult("unsupported", reason=f"the state before it doesn't add up ({e})")
    if top and boundary.players[0].deck[-len(top):][::-1] != top[: len(boundary.players[0].deck)]:
        return OpenPlayResult("unsupported", reason="the cards it drew aren't all in your deck by elimination")

    path = [Action("PLAY", open_play.card)]
    game = boundary.clone()

    def step(action: Action) -> None:
        game.step(action)
        path.append(action)

    try:
        game.step(path[0])
        i = 1
        while i < len(events):
            pending = game.pending_decision
            if pending is None:
                break  # the card finished; later lines are ordinary phase play
            if pending.player != 0:
                later_opp_lines = any(not e.mine for e in events[i:])
                return OpenPlayResult("resolved" if later_opp_lines else "opponent", boundary, path)
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
                        break  # play it or not: exactly what's waiting on you
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
                return OpenPlayResult("unsupported", reason=f"{pending.source_card}'s yes/no choice can't be read "
                                                            f"from the log yet")
            verb = _CHOICE_VERBS.get(e.kind)
            if verb is None:
                # A treasure or buy: whatever optional choice was still open
                # was declined (e.g. Chapel trashing nothing logs no line).
                if DONE in pending.options:
                    step(DONE)
                elif _NONE in pending.options:
                    step(_NONE)
                else:
                    return OpenPlayResult("unsupported", reason=f"the log moved on while the engine was asking: "
                                                                f"{pending.prompt}")
                continue
            for card in e.cards:
                action = Action(verb, card)
                while game.pending_decision is not None and action not in game.pending_decision.options:
                    options = game.pending_decision.options
                    if game.pending_decision.player != 0 or not (DONE in options or _NONE in options):
                        return OpenPlayResult("unsupported", reason=f"the log shows {action} but the engine is "
                                                                    f"asking: {game.pending_decision.prompt}")
                    step(DONE if DONE in options else _NONE)
                if game.pending_decision is None:
                    return OpenPlayResult("unsupported", reason=f"the log shows {action} after the card finished")
                step(action)
            # The logged line was one confirmed selection: close it if the
            # engine is still offering more of the same.
            after = game.pending_decision
            if after is not None and after.player == 0 and DONE in after.options and \
                    all(o.verb in (verb, "DONE") for o in after.options):
                step(DONE)
            i += 1
    except ValueError as e:
        return OpenPlayResult("unsupported", reason=f"the engine rejected a replayed step ({e})")

    if game.pending_decision is None:
        return OpenPlayResult("resolved", boundary, path)
    if game.pending_decision.player != 0:
        return OpenPlayResult("opponent", boundary, path)
    if Counter(game.players[0].hand) != Counter(final_my_hand):
        return OpenPlayResult("unsupported", reason="the replayed hand doesn't match the log's")
    return OpenPlayResult("pending", boundary, path)
