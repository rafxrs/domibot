"""Derives as much of a `training.relay.TableState` as possible from a
pasted dominion.games text log, instead of entering it by hand. See
`examples/domibot_relay.py`.

Two layers, in increasing order of ambition:

1. `supply`, `trash`, and `my_total` come from reading every buy/gain/
   trash line, which are always public and always named for *either*
   player -- this layer is simple and has no real failure mode.
2. A full turn-by-turn replay additionally derives your own hand,
   discard, play area, phase, actions, buys, and coins, *and* the
   opponent's hand size, draw-pile size, and discard pile -- covering
   plays (including Throne-Room-style "plays X again" replays, which
   reapply a card's effect without moving it again), buys/gains (to
   discard, except a card-triggered gain whose source card sends it
   elsewhere -- Artisan/Mine to hand, Bureaucrat's own Silver to the
   deck top), trashes/discards/
   topdecks (sourced from hand normally, from a just-revealed/looked-at
   card for Sentry/Bandit-style effects, or from discard for Harbinger),
   explicit "gets +N Action/Buy/$" lines, and end-of-turn cleanup
   (detected positionally: the last draw in a player's turn block, i.e.
   the one immediately before the next `Turn` header or the game-end
   line, is the cleanup redraw; every other draw in that block just adds
   to hand).

   The opponent's hand contents are never tracked, only their *count* --
   so at their cleanup, only their known play-area cards (always named)
   flush into their tracked discard; whatever was in their hand simply
   stops being separately counted; it's still correctly accounted for
   because reconstruct_game only needs total/discard/play-area/sizes,
   never "which specific unseen card is where."

   This layer is inherently best-effort: dominion.games occasionally
   renders a card as a bare, unnamed "a card" (e.g. a Cellar-style
   discard can read `discards a card and a Remodel`, naming one and not
   the other), which makes exact reconstruction impossible for that
   event. Rather than guess, this layer simply gives up (leaving its
   fields `None`) the moment it hits something it can't resolve exactly
   -- layer 1's fields are computed independently and always still
   returned, so a layer-2 failure never costs you the baseline you
   already had.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from domibot import ALL_CARDS, Action, CardType

_RATING_LINE = re.compile(r"^([\w.\- ]+): [\d.]+$")
_TURN_LINE = re.compile(r"^Turn (\d+) - (.+)$")
_GAME_END_LINE = re.compile(r"^The game has ended\.?$", re.IGNORECASE)
_STARTS_WITH_LINE = re.compile(r"^(\S+) starts with \d+ \S+\.$")
_SHUFFLE_LINE = re.compile(r"^(\S+) shuffles their deck\.$")
_BUY_GAIN_LINE = re.compile(r"^(\S+) buys and gains (.+)\.$")
_GAIN_LINE = re.compile(r"^(\S+) gains (.+)\.$")
_TRASH_LINE = re.compile(r"^(\S+) trashes (.+)\.$")
_DRAW_LINE = re.compile(r"^(\S+) draws (.+)\.$")
_GENERIC_DRAW = re.compile(r"^(?:(\d+) cards?|an? card)$", re.IGNORECASE)
_PLAY_TREASURE_LINE = re.compile(r"^(\S+) plays (.+)\. \(\+\$(\d+)\)$")
_PLAY_ACTION_LINE = re.compile(r"^(\S+) plays (?:an? )?(.+?)( again)?\.$")
_DISCARD_LINE = re.compile(r"^(\S+) discards (.+)\.$")
_TOPDECK_LINE = re.compile(r"^(\S+) topdecks (.+)\.$")
_REVEALS_LINE = re.compile(r"^(\S+) reveals (.+)\.$")
_LOOKS_AT_LINE = re.compile(r"^(\S+) looks at (.+)\.$")
# Bureaucrat's "no Victory card in hand" fallback: the whole hand is shown
# as proof, but nothing moves anywhere -- a different phrasing from a
# Sentry/Bandit-style reveal-then-resolve, and purely informational.
_REVEALS_HAND_LINE = re.compile(r"^(\S+) reveals their hand: (.+)\.$")
_GETS_ACTIONS_LINE = re.compile(r"^(\S+) gets \+(\d+) Actions?\.$")
_GETS_BUYS_LINE = re.compile(r"^(\S+) gets \+(\d+) Buys?\.$")
_GETS_COINS_LINE = re.compile(r"^(\S+) gets \+\$(\d+)\.")

# Which card *caused* a gain determines its destination (per
# src/domibot/cards/kingdom.py), never the gained card's own name -- a
# bought Bureaucrat, say, still goes to discard like anything else.
_GAIN_TO_HAND_SOURCES = {"Artisan", "Mine"}
_GAIN_TO_DECK_TOP_SOURCES = {"Bureaucrat"}

# Cards whose effect can never yield a Decision for the player who played
# them -- either a flat bonus with no `effect` at all (Village/Smithy/
# Festival/Laboratory/Market), an `effect` that never yields (Merchant),
# or an attack whose only Decision targets the *victim*, never the
# attacker (Witch/Bureaucrat/Bandit/Militia; Moat has no effect of its own,
# only a REACT it can trigger for someone else's attack). Used to decide
# whether a still-open opponent turn is safe to replay past via
# training.mcts.materialize: anything NOT in this set (Chapel, Sentry,
# Workshop, Artisan, Vassal, Harbinger, Cellar, Moneylender, Poacher,
# Remodel, Mine, Library, Throne Room, and Council Room -- which silently
# draws a card into *your* hand as a side effect) can itself demand a
# choice from the player who played it, which nothing here models.
_SAFE_MIDTURN_ACTION_CARDS = {
    "Village", "Smithy", "Festival", "Laboratory", "Market", "Merchant",
    "Moat", "Witch", "Bureaucrat", "Bandit", "Militia",
}
# Attack cards this module can build a pending-reaction path for -- see
# `_PendingReaction`. Militia only for now; Bureaucrat/Bandit are
# mechanically identical to support (same "path ends in a safe attack"
# shape) but are deliberately out of scope for this pass.
_SUPPORTED_TERMINAL_ATTACKS = {"Militia"}

_STARTING_COPPER = 7
_STARTING_ESTATE = 3


@dataclass
class ParsedLog:
    supply: dict[str, int]
    trash: list[str]
    my_total: list[str]
    turns_taken: dict[str, int] = field(default_factory=dict)  # full player name -> turns started so far
    # Only set when the full replay (layer 2) succeeds end to end.
    my_hand: list[str] | None = None
    my_discard: list[str] | None = None
    my_play_area: list[str] | None = None
    my_phase: str | None = None
    my_actions: int | None = None
    my_buys: int | None = None
    my_coins: int | None = None
    opp_discard: list[str] | None = None
    opp_play_area: list[str] | None = None
    opp_hand_size: int | None = None
    opp_draw_pile_size: int | None = None
    # Set only when the log ends with the opponent's turn still open and a
    # reaction pending on you (see _SUPPORTED_TERMINAL_ATTACKS) -- the
    # ordered PLAY actions of their turn so far (ending in the attack), and
    # a snapshot of their discard at the moment their turn started (their
    # play_area/hand_size at turn start are always [] and 5, so nothing to
    # snapshot there -- see training.relay.reconstruct_opponent_turn_boundary).
    pending_reaction_path: list[Action] | None = None
    pending_reaction_opp_discard: list[str] | None = None


def _singularize(word: str) -> str:
    if word in ALL_CARDS:
        return word
    if word.endswith("ies") and (word[:-3] + "y") in ALL_CARDS:
        return word[:-3] + "y"
    if word.endswith("s") and word[:-1] in ALL_CARDS:
        return word[:-1]
    raise ValueError(f"not a recognized card name: {word!r}")


_SEGMENT_RE = re.compile(r"^(?:(\d+)|an?)\s+(.+)$", re.IGNORECASE)


def _parse_card_list(text: str) -> list[str]:
    """'3 Coppers and 2 Estates' / 'a Copper, a Silver, and an Estate' ->
    the expanded list of card names. Raises on anything it can't resolve
    to a real card name (e.g. a bare 'a card') rather than guessing."""
    text = text.strip().rstrip(".")
    if not text:
        return []
    text = text.replace(", and ", ", ").replace(" and ", ", ")
    cards: list[str] = []
    for segment in text.split(","):
        segment = segment.strip()
        if not segment:
            continue
        m = _SEGMENT_RE.match(segment)
        if not m:
            raise ValueError(f"can't parse {segment!r} as a card list segment")
        count = int(m.group(1)) if m.group(1) else 1
        cards.extend([_singularize(m.group(2).strip())] * count)
    return cards


# "N other cards" (Cellar discarding several unnamed cards alongside a
# named one) and the singular "a card" (Militia's forced discard down to 3,
# e.g. "discards a card and a Copper") -- dominion.games drops "other" in
# the one-card case rather than saying "1 other card".
_ANON_SEGMENT_RE = re.compile(r"^(?:(\d+)\s+other\s+cards?|an?\s+card)$", re.IGNORECASE)


def _parse_card_list_with_anonymous(text: str) -> tuple[list[str], int]:
    """Like `_parse_card_list`, but also tolerates one or more anonymous
    segments (`_ANON_SEGMENT_RE`) -- dominion.games' placeholder for cards
    it won't name from a hidden hand (e.g. Cellar discarding a mix of named
    and unnamed cards: 'discards 3 other cards and a Copper', or Militia's
    forced discard: 'discards a card and a Copper') -- returned separately
    as a count rather than real names, since we don't (and, for an
    opponent's hidden hand, can't) know which cards those were."""
    text = text.strip().rstrip(".")
    if not text:
        return [], 0
    text = text.replace(", and ", ", ").replace(" and ", ", ")
    named: list[str] = []
    anonymous = 0
    for segment in text.split(","):
        segment = segment.strip()
        if not segment:
            continue
        anon_match = _ANON_SEGMENT_RE.match(segment)
        if anon_match:
            anonymous += int(anon_match.group(1)) if anon_match.group(1) else 1
            continue
        m = _SEGMENT_RE.match(segment)
        if not m:
            raise ValueError(f"can't parse {segment!r} as a card list segment")
        count = int(m.group(1)) if m.group(1) else 1
        named.extend([_singularize(m.group(2).strip())] * count)
    return named, anonymous


def _remove_one(zone: list[str], card: str) -> bool:
    if card in zone:
        zone.remove(card)
        return True
    return False


def _card_name_from_play_line(raw_name: str) -> str:
    return raw_name if raw_name in ALL_CARDS else _singularize(raw_name)


@dataclass
class _MyState:
    hand: list[str]
    discard: list[str]
    play_area: list[str]
    actions: int = 1
    buys: int = 1
    coins: int = 0
    phase: str = "ACTION"


@dataclass
class _OppState:
    hand_size: int
    discard: list[str]
    play_area: list[str]


@dataclass
class _PendingReaction:
    """The log ended with the opponent's turn still open and a reaction
    pending on you -- see _SUPPORTED_TERMINAL_ATTACKS."""
    path: list[Action]
    opp_turn_start_discard: list[str]


def _replay_full_state(
    lines: list[str], my_full_name: str, opp_full_name: str, resolve
) -> tuple[_MyState, _OppState, "_PendingReaction | None"]:
    """Raises ValueError the moment it hits anything it can't resolve
    exactly. Returns final states only if it makes it to the end clean."""
    me = _MyState(hand=[], discard=[], play_area=[])
    # Starts at 0, not 5: the log's own opening "X draws 5 cards." line
    # (shown before Turn 1) is what brings this up to 5 -- pre-seeding it
    # would double-count that draw.
    opp = _OppState(hand_size=0, discard=[], play_area=[])
    current_turn_player: str | None = None
    # Cards currently known to be sitting on top of a player's deck
    # (revealed by Sentry/Bandit-style effects), awaiting a trash/discard/
    # topdeck that resolves them -- as opposed to a plain hand-sourced one.
    pending_reveal: dict[str, list[str]] = {my_full_name: [], opp_full_name: []}
    last_played: dict[str, str | None] = {my_full_name: None, opp_full_name: None}

    # Tracking for _PendingReaction: the opponent's PLAY actions so far on
    # their still-open turn, provided every one of them is in
    # _SAFE_MIDTURN_ACTION_CARDS (anything else -- a card that could itself
    # demand a choice from them, or any non-PLAY event touching their
    # zones beyond an automatic echo of a safe card's flat bonus --
    # disqualifies the whole turn, not just that one play). Reset at every
    # _TURN_LINE; opp_turn_start_discard is snapshotted there too, since
    # discard is the one opponent zone that isn't provably constant at
    # turn start (see training.relay.reconstruct_opponent_turn_boundary).
    current_turn_safe_path: list[Action] = []
    current_turn_disqualified = False
    opp_turn_start_discard: list[str] | None = None
    game_ended = False
    # A second (or later) Throne Room played by the opponent in the same
    # turn, after they've also played a Vassal that turn, is where exact
    # replay breaks down: Vassal can play a revealed card straight from the
    # deck top without it ever touching hand (logged identically to a plain
    # hand play -- no "reveals" line precedes it, unlike Sentry/Bandit), so
    # once a second Throne Room enters the mix there's no way to tell from
    # the log text alone which "plays a Throne Room"/"discards a Throne
    # Room" lines are independent hand-sourced copies and which are a
    # Vassal reveal of a card already accounted for elsewhere -- see the
    # log_parser.py module docstring's "best-effort... give up rather than
    # guess" policy. A single Throne Room (however many cards it doubles)
    # or a Vassal with no second Throne Room are both unambiguous and stay
    # fully supported; only the combination bails.
    opp_throne_room_plays_this_turn = 0
    opp_played_vassal_this_turn = False
    # True exactly between the opponent playing a Vassal (any resolution of
    # it -- the "again" doubled one included) and the single play/discard
    # that resolves its reveal: that one card comes straight off the deck
    # top, never touching hand, so (unlike every other opponent play/
    # discard, which the code below otherwise always assumes is
    # hand-sourced) it must not decrement opp.hand_size. Doesn't affect
    # *which* zone the card ends up counted in (play_area if they chose to
    # play it, discard if not) -- only whether it's charged against their
    # hand.
    opp_vassal_pending = False
    # True exactly when the most recent play in current_turn_safe_path was
    # a supported attack (Militia) whose reaction hasn't shown up yet --
    # cleared the moment a line about *me* appears during the opponent's
    # still-open turn (my discard once logged, or a Moat reveal blocking
    # it), since either means there's nothing left pending to recommend.
    reaction_pending = False

    def is_boundary(next_line: str | None) -> bool:
        return next_line is None or bool(_TURN_LINE.match(next_line)) or bool(_GAME_END_LINE.match(next_line))

    def cleanup(player: str) -> None:
        if player == my_full_name:
            me.discard.extend(me.hand)
            me.discard.extend(me.play_area)
            me.hand = []
            me.play_area = []
            me.actions, me.buys, me.coins, me.phase = 1, 1, 0, "ACTION"
        else:
            opp.discard.extend(opp.play_area)
            opp.play_area = []
            opp.hand_size = 0

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        m = _TURN_LINE.match(line)
        if m:
            current_turn_player = m.group(2)
            current_turn_safe_path = []
            current_turn_disqualified = False
            reaction_pending = False
            opp_throne_room_plays_this_turn = 0
            opp_played_vassal_this_turn = False
            opp_vassal_pending = False
            if current_turn_player == opp_full_name:
                opp_turn_start_discard = list(opp.discard)
            continue
        if _GAME_END_LINE.match(line):
            game_ended = True
            continue
        if _STARTS_WITH_LINE.match(line) or _RATING_LINE.match(line):
            continue

        m = _SHUFFLE_LINE.match(line)
        if m:
            player = resolve(m.group(1))
            # Real Cleanup order is: discard hand + play area, *then* draw
            # a new hand (shuffling discard into deck first if it's short).
            # dominion.games doesn't log that automatic discard as its own
            # line, so when this shuffle is immediately followed by this
            # same player's cleanup draw, the about-to-be-discarded hand/
            # play area is *also* swept into it -- logged one line "early"
            # relative to when it mechanically happens. Otherwise this is
            # just a plain mid-turn shuffle (e.g. an action card needing to
            # draw with an empty deck).
            upcoming_cleanup = (
                player == current_turn_player
                and next_line is not None and bool(_DRAW_LINE.match(next_line))
                and is_boundary(lines[i + 2] if i + 2 < len(lines) else None)
            )
            # Discard becomes the new deck; deck contents/order are never
            # tracked here (reconstruct_game derives them by elimination
            # anyway), so this is just "discard is now empty." When this is
            # the cleanup shuffle, the not-yet-logged hand/play-area discard
            # is folded into that same deck, so both must be wiped *now* --
            # otherwise the subsequent cleanup() call (fired by the actual
            # draw line) re-adds the stale leftover hand into the fresh
            # post-shuffle discard, resurrecting cards that already got
            # folded into the (untracked) deck.
            if player == my_full_name:
                me.discard = []
                if upcoming_cleanup:
                    me.play_area = []
                    me.hand = []
            else:
                opp.discard = []
                if upcoming_cleanup:
                    opp.play_area = []
            continue

        m = _PLAY_TREASURE_LINE.match(line)
        if m:
            abbrev, card_text, coins = m.groups()
            player = resolve(abbrev)
            cards = _parse_card_list(card_text)
            last_played[player] = cards[-1] if cards else last_played[player]
            if player == my_full_name:
                for c in cards:
                    if not _remove_one(me.hand, c):
                        raise ValueError(f"played {c!r} not found in tracked hand")
                me.play_area.extend(cards)
                me.coins += int(coins)
                me.phase = "BUY"
            else:
                opp.hand_size -= len(cards)
                opp.play_area.extend(cards)
                # Treasures are always played in the Buy phase, after any
                # attack in the same turn's Action phase would already
                # have appeared -- reaching this for the opponent means
                # their turn has moved past where a pending-reaction path
                # could still apply.
                current_turn_disqualified = True
            continue

        m = _PLAY_ACTION_LINE.match(line)
        if m:
            abbrev, raw_name, again = m.groups()
            player = resolve(abbrev)
            card_name = _card_name_from_play_line(raw_name)
            last_played[player] = card_name
            if player == my_full_name:
                me.phase = "ACTION"  # an action play only ever happens in the action phase
                if not again:
                    if not _remove_one(me.hand, card_name):
                        raise ValueError(f"played {card_name!r} not found in tracked hand")
                    me.play_area.append(card_name)
                    me.actions -= 1
            else:
                if not again:
                    if card_name == "Throne Room":
                        if opp_throne_room_plays_this_turn >= 1 and opp_played_vassal_this_turn:
                            raise ValueError(
                                "opponent played a second Throne Room this turn after also playing a Vassal -- "
                                "Vassal can play a card straight off the deck top without it ever touching hand, "
                                "logged identically to a plain hand play, so which 'plays'/'discards a Throne "
                                "Room' lines are independent copies vs. a Vassal reveal can't be told apart from "
                                "the log text alone once a second Throne Room is in the mix"
                            )
                        opp_throne_room_plays_this_turn += 1
                    elif card_name == "Vassal":
                        opp_played_vassal_this_turn = True
                    if opp_vassal_pending:
                        # this play *is* the pending Vassal's reveal
                        # resolution (they chose to play the revealed
                        # card) -- it came off the deck top, not hand.
                        opp_vassal_pending = False
                    else:
                        opp.hand_size -= 1
                    opp.play_area.append(card_name)
                    if card_name == "Vassal":
                        opp_vassal_pending = True  # this Vassal's own reveal is now pending
                # A Throne-Room-style replay (`again`) would need its own
                # sub-decision (which card to double) represented in the
                # path, which nothing here builds -- and any card outside
                # the whitelist could itself demand a choice from the
                # opponent, which replaying past would silently skip.
                if again or card_name not in _SAFE_MIDTURN_ACTION_CARDS:
                    current_turn_disqualified = True
                else:
                    current_turn_safe_path.append(Action("PLAY", card_name))
                    reaction_pending = card_name in _SUPPORTED_TERMINAL_ATTACKS
            continue

        m = _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            is_buy = bool(_BUY_GAIN_LINE.match(line))
            # A buy always lands in discard regardless of what was just
            # played; a card-triggered gain's destination depends on which
            # card caused it (e.g. Artisan/Mine gain to hand, Bureaucrat
            # gains its Silver straight to the deck top).
            source = None if is_buy else last_played.get(player)
            to_hand = source in _GAIN_TO_HAND_SOURCES
            to_deck_top = source in _GAIN_TO_DECK_TOP_SOURCES
            for card in _parse_card_list(card_text):
                if player == my_full_name:
                    if is_buy:
                        me.buys -= 1
                    if to_hand:
                        me.hand.append(card)
                    elif not to_deck_top:
                        me.discard.append(card)
                    # deck-top gains aren't tracked in any zone -- deck
                    # contents/order are always derived by elimination.
                else:
                    if to_hand:
                        opp.hand_size += 1
                    elif not to_deck_top:
                        opp.discard.append(card)
                    current_turn_disqualified = True
            continue

        if _REVEALS_HAND_LINE.match(line):
            continue  # informational only -- nothing to trash/discard/topdeck results from this

        m = _LOOKS_AT_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            if last_played.get(player) == "Harbinger":
                # Harbinger looks through its owner's own discard pile --
                # already tracked exactly (discard is always public), so
                # there's nothing here to learn from the line's text, which
                # dominion.games often abbreviates as a mix of named cards
                # and an anonymous "N other cards" count (not a real card
                # name _parse_card_list can resolve). The subsequent topdeck
                # (below) reads straight from the tracked discard instead of
                # relying on anything parsed from this line.
                continue
            # Library logs each drawn-and-examined card as a bare, unnamed
            # "looks at a card." (unlike Sentry/Bandit's named deck-top
            # reveals) -- the card itself is drawn from the deck, so it's
            # a draw for hand-size purposes; whether it's later set aside
            # is resolved separately by an ordinary named discard line.
            if _GENERIC_DRAW.match(card_text.strip()):
                if player == my_full_name:
                    raise ValueError("your own Library draw was unnamed -- can't track exact hand from here")
                opp.hand_size += 1
                current_turn_disqualified = True
            else:
                pending_reveal[player].extend(_parse_card_list(card_text))
                if player == opp_full_name:
                    current_turn_disqualified = True
            continue

        m = _REVEALS_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            revealed = _parse_card_list(card_text)
            pending_reveal[player].extend(revealed)
            if player == opp_full_name:
                current_turn_disqualified = True
            elif player == my_full_name and current_turn_player == opp_full_name and "Moat" in revealed:
                # Blocks the attack outright -- nothing left pending to
                # recommend. (Unverified against a real log: this exact
                # phrasing hasn't been seen in a Moat-reveal-to-block-an-
                # attack fixture yet, only inferred from _REVEALS_LINE's
                # general pattern.)
                reaction_pending = False
            continue

        m = _TRASH_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            for card in _parse_card_list(card_text):
                from_reveal = _remove_one(pending_reveal[player], card)
                if player == my_full_name:
                    if not from_reveal and not _remove_one(me.hand, card):
                        raise ValueError(f"trashed {card!r} not found in tracked hand")
                else:
                    if not from_reveal:
                        opp.hand_size -= 1
                    current_turn_disqualified = True
            continue

        m = _DISCARD_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            named, anonymous = _parse_card_list_with_anonymous(card_text)
            for card in named:
                from_reveal = _remove_one(pending_reveal[player], card)
                if player == my_full_name:
                    if not from_reveal and not _remove_one(me.hand, card):
                        raise ValueError(f"discarded {card!r} not found in tracked hand")
                    me.discard.append(card)
                    if current_turn_player == opp_full_name:
                        reaction_pending = False  # this resolves the Militia discard just requested
                else:
                    if not from_reveal:
                        if opp_vassal_pending:
                            # this discard *is* the pending Vassal's reveal
                            # resolution (declined) -- straight off the deck
                            # top, not hand.
                            opp_vassal_pending = False
                        else:
                            opp.hand_size -= 1
                    opp.discard.append(card)
                    # A discard belonging to the opponent, during their own
                    # still-open turn, can only be self-caused (Cellar/
                    # Sentry/Poacher) -- a reaction they force on *me* is
                    # a discard of mine, handled in the other branch above.
                    current_turn_disqualified = True
            if anonymous:
                if player == my_full_name:
                    raise ValueError("your own discard included unnamed 'other card(s)' -- can't track exact hand from here")
                # A real card each, just not one we can name -- counted out
                # of the opponent's hand, but left out of their tracked
                # discard (which must stay a list of actually-known cards).
                opp.hand_size -= anonymous
                current_turn_disqualified = True
            continue

        m = _TOPDECK_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            harbinger_context = last_played.get(player) == "Harbinger"
            # Artisan's own second decision (put a card from *hand* onto
            # your deck) is rendered as an unnamed "topdecks a card" for
            # the opponent, since hand contents are hidden. dominion.games
            # renders Harbinger's own discard-sourced topdeck unnamed too,
            # at least sometimes (seen for real) -- unlike Artisan's, this
            # one *is* fully knowable, since the discard it came from is
            # already tracked exactly, so rather than aborting the replay
            # this just removes an arbitrary card from the tracked discard.
            # Which specific one doesn't matter for anything this module
            # computes (only zone *counts* feed reconstruct_game's by-
            # elimination math), so this isn't the guess the module
            # docstring otherwise avoids -- it's a don't-care.
            if not harbinger_context and _GENERIC_DRAW.match(card_text.strip()):
                if player == my_full_name:
                    raise ValueError("your own Artisan topdeck was unnamed -- can't track exact hand from here")
                opp.hand_size -= 1
                continue
            if harbinger_context and _GENERIC_DRAW.match(card_text.strip()):
                if player == my_full_name:
                    raise ValueError("your own Harbinger topdeck was unnamed -- can't track exact hand from here")
                if not opp.discard:
                    raise ValueError(
                        "opponent's Harbinger topdeck was unnamed but their tracked discard is empty -- "
                        "something drifted"
                    )
                opp.discard.pop()
                continue
            for card in _parse_card_list(card_text):
                from_reveal = _remove_one(pending_reveal[player], card)
                if from_reveal:
                    continue  # was on top of the deck the whole time; no zone actually changes
                if player == my_full_name:
                    if harbinger_context:
                        if not _remove_one(me.discard, card):
                            raise ValueError(f"topdecked {card!r} not found in tracked discard")
                    elif not _remove_one(me.hand, card):
                        raise ValueError(f"topdecked {card!r} not found in tracked hand")
                else:
                    if harbinger_context:
                        if not _remove_one(opp.discard, card):
                            raise ValueError(f"opponent topdecked {card!r} not found in tracked discard")
                    else:
                        opp.hand_size -= 1
            continue

        m = _DRAW_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            card_text = card_text.strip()
            is_cleanup = player == current_turn_player and is_boundary(next_line)
            generic = _GENERIC_DRAW.match(card_text)
            if player == my_full_name:
                if generic:
                    raise ValueError("your own draw was unnamed -- can't track exact hand from here")
                cards = _parse_card_list(card_text)
                if is_cleanup:
                    cleanup(player)
                me.hand.extend(cards)
            else:
                if generic:
                    count = int(generic.group(1)) if generic.group(1) else 1
                else:
                    count = len(_parse_card_list(card_text))
                if is_cleanup:
                    cleanup(player)
                opp.hand_size += count
            continue

        m = _GETS_ACTIONS_LINE.match(line)
        if m:
            if resolve(m.group(1)) == my_full_name:
                me.actions += int(m.group(2))
            continue

        m = _GETS_BUYS_LINE.match(line)
        if m:
            if resolve(m.group(1)) == my_full_name:
                me.buys += int(m.group(2))
            continue

        m = _GETS_COINS_LINE.match(line)
        if m:
            if resolve(m.group(1)) == my_full_name:
                me.coins += int(m.group(2))
            continue

    pending_reaction = None
    if (reaction_pending and not current_turn_disqualified and not game_ended
            and current_turn_player == opp_full_name and opp_turn_start_discard is not None):
        pending_reaction = _PendingReaction(current_turn_safe_path, opp_turn_start_discard)
    return me, opp, pending_reaction


def parse_dominion_log(text: str, my_name: str, kingdom: list[str], num_players: int = 2) -> ParsedLog:
    """`my_name` is your account name as it appears in the log's player/
    rating list (e.g. 'domibot_v1.4') -- matched case-insensitively against
    both the rating-list names and the single-letter turn-line prefixes."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    player_names: list[str] = []
    for line in lines:
        m = _RATING_LINE.match(line)
        if m and not _TURN_LINE.match(line):
            player_names.append(m.group(1))
        if _TURN_LINE.match(line):
            break
    if not player_names:
        # No "name: rating" header pasted (e.g. a trimmed practice-game
        # log) -- default to you vs. dominion.games' own built-in bot,
        # since that's who a header-less log is overwhelmingly likely to
        # be against. If this guess is wrong, an unrecognized abbreviation
        # or a failed consistency check downstream will surface it rather
        # than silently mis-model the game.
        player_names = [my_name, "Lord Rattington"]

    my_matches = [n for n in player_names if n.lower() == my_name.lower()]
    if not my_matches:
        raise ValueError(f"{my_name!r} isn't one of the players in this log: {player_names}")
    my_full_name = my_matches[0]
    other_names = [n for n in player_names if n != my_full_name]

    # dominion.games abbreviates each player to (usually) their first
    # letter in action lines ("d plays...", "l plays..."); build that
    # mapping from the same name list rather than assuming which letter.
    abbrev_to_name: dict[str, str] = {}
    for name in player_names:
        letter = name[0].lower()
        if letter in abbrev_to_name:
            raise ValueError(
                f"two players start with the same letter ({name!r} and {abbrev_to_name[letter]!r}) -- "
                f"this log's per-line abbreviations are ambiguous, can't tell them apart automatically"
            )
        abbrev_to_name[letter] = name

    def resolve(abbrev: str) -> str:
        name = abbrev_to_name.get(abbrev.lower())
        if name is None:
            raise ValueError(f"unrecognized player abbreviation {abbrev!r} (known: {abbrev_to_name})")
        return name

    victory_pile_size = 8 if num_players == 2 else 12
    supply: Counter = Counter({"Copper": 60 - _STARTING_COPPER * num_players,
                                "Silver": 40, "Gold": 30,
                                "Estate": victory_pile_size,
                                "Duchy": victory_pile_size,
                                "Province": victory_pile_size,
                                "Curse": 10 * (num_players - 1)})
    for name in kingdom:
        # A Kingdom card that's also a Victory card (e.g. Gardens) uses the
        # same pile size as the basic Victory cards, not the flat 10 every
        # other Kingdom card gets -- see domibot.Game.__init__, the same
        # rule applied there.
        is_victory = CardType.VICTORY in ALL_CARDS[name].types
        supply[name] = victory_pile_size if is_victory else 10

    trash: list[str] = []
    my_total: Counter = Counter({"Copper": _STARTING_COPPER, "Estate": _STARTING_ESTATE})
    turns_taken: Counter = Counter()

    for line in lines:
        m = _TURN_LINE.match(line)
        if m:
            _, full_name = m.groups()
            turns_taken[full_name] += 1
            continue

        m = _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            for card in _parse_card_list(card_text):
                supply[card] -= 1
                if player == my_full_name:
                    my_total[card] += 1
            continue

        m = _TRASH_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            for card in _parse_card_list(card_text):
                trash.append(card)
                if player == my_full_name:
                    my_total[card] -= 1
            continue

    for card, count in supply.items():
        if count < 0:
            raise ValueError(f"supply for {card} went negative -- a line in this log wasn't parsed as expected")
    for card, count in my_total.items():
        if count < 0:
            raise ValueError(f"my_total for {card} went negative -- a trash line was likely mis-attributed")

    result = ParsedLog(
        supply=dict(supply),
        trash=trash,
        my_total=list(Counter(my_total).elements()),
        # "turns started" includes the turn in progress; TableState wants
        # completed turns *before* the current one.
        turns_taken={name: max(count - 1, 0) for name, count in turns_taken.items()},
    )

    if len(other_names) == 1:
        try:
            me, opp, pending_reaction = _replay_full_state(lines, my_full_name, other_names[0], resolve)
            # Every card is in exactly one of {supply, trash, mine, theirs} --
            # solved by elimination, the same trick reconstruct_game uses.
            # A Victory-type Kingdom card (Gardens) starts at victory_pile_size
            # total, not 10, same as the supply-init loop above.
            kingdom_total = sum(
                victory_pile_size if CardType.VICTORY in ALL_CARDS[name].types else 10
                for name in kingdom
            )
            total_sum = (
                60 + 40 + 30
                + victory_pile_size + _STARTING_ESTATE * num_players
                + victory_pile_size
                + victory_pile_size
                + 10 * (num_players - 1)
                + kingdom_total
            )
            opp_total = total_sum - sum(supply.values()) - len(trash) - sum(my_total.values())
            opp_draw_pile_size = opp_total - opp.hand_size - len(opp.discard) - len(opp.play_area)
            if opp_draw_pile_size < 0:
                raise ValueError("replay produced a negative opponent draw-pile size -- something drifted")
        except ValueError:
            pass
        else:
            result.my_hand = me.hand
            result.my_discard = me.discard
            result.my_play_area = me.play_area
            result.my_phase = me.phase
            result.my_actions = me.actions
            result.my_buys = me.buys
            result.my_coins = me.coins
            result.opp_discard = opp.discard
            result.opp_play_area = opp.play_area
            result.opp_hand_size = opp.hand_size
            result.opp_draw_pile_size = opp_draw_pile_size
            if pending_reaction is not None:
                result.pending_reaction_path = pending_reaction.path
                result.pending_reaction_opp_discard = pending_reaction.opp_turn_start_discard

    return result
