"""Derives as much of a `training.relay.TableState` as possible from a
pasted dominion.games text log, instead of entering it by hand. See
`examples/domibot_relay.py`.

Two layers, in increasing order of ambition:

1. `supply`, `trash`, and `my_total` come from reading every buy/gain/
   trash line, which are always public and always named for *either*
   player -- this layer is simple and has no real failure mode.
2. A full turn-by-turn replay additionally derives your own hand,
   discard, play area, phase, actions, buys, and coins, *and* the
   opponent's hand size, draw-pile size, and discard pile, plus any cards
   known to sit on top of either deck (Sentry/Artisan/Harbinger/Bureaucrat
   topdecks). It covers plays (including Throne-Room-style "plays X again"
   replays, which reapply a card's effect without moving it again, and
   Vassal playing a card from the discard pile), buys/gains (to discard,
   except a card-triggered gain whose source card sends it elsewhere --
   Artisan/Mine to hand, Bureaucrat's own Silver to the deck top),
   trashes/discards/topdecks (sourced from hand normally, from a
   just-revealed/looked-at card for Sentry/Bandit-style effects, from the
   deck top for Vassal, or from discard for Harbinger), explicit "gets +N
   Action/Buy/$" lines, and end-of-turn cleanup (detected positionally: a
   draw immediately before the next `Turn` header or the game-end line is
   the cleanup redraw; every other draw just adds to hand).

   The opponent's hand contents are never tracked, only their *count* --
   so at their cleanup, only their known play-area cards (always named)
   flush into their tracked discard; whatever was in their hand simply
   stops being separately counted; it's still correctly accounted for
   because reconstruct_game only needs total/discard/play-area/sizes,
   never "which specific unseen card is where."

   This layer is inherently best-effort: dominion.games sometimes renders
   a card as a bare, unnamed "a card" where exact reconstruction then
   isn't possible, and a line this module doesn't recognize could move
   cards in ways it can't follow. Rather than guess, this layer gives up
   (leaving its fields `None`) the moment it hits either -- layer 1's
   fields are computed independently and always still returned, so a
   layer-2 failure never costs you the baseline you already had.

When the log ends partway through your own turn, `ParsedLog.open_play`
also records the state just before your most recent phase-level Action
play and every log line since, so `training.relay.replay_open_play` can
replay that card through the engine and tell whether one of its choices
(which card to trash, what to gain, ...) is still waiting on you.
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
# bought Bureaucrat, say, still goes to discard like anything else. Only
# ever applies to the player whose turn it is: a gain by anyone else is
# always an attack's (Witch's Curse), which goes to discard.
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
# Attacks that can leave a choice pending on you mid-way through the
# opponent's turn -- see `_PendingReaction`: Militia's discard,
# Bureaucrat's topdeck, Bandit's trash, and (for all four) whether to
# reveal Moat.
_SUPPORTED_TERMINAL_ATTACKS = {"Militia", "Bureaucrat", "Bandit", "Witch"}
# An attack's own gain, which the engine replays by itself as part of the
# play -- not a sign the opponent's turn has moved past the attack.
_ATTACK_SELF_GAINS = {"Bureaucrat": "Silver", "Bandit": "Gold"}

_STARTING_COPPER = 7
_STARTING_ESTATE = 3


@dataclass
class LogEvent:
    """One log line, pre-parsed for `training.relay.replay_open_play`."""
    mine: bool
    kind: str  # play, treasure, draw, look, reveal, trash, discard, gain, buy, topdeck, shuffle, gets, other
    cards: list[str] = field(default_factory=list)
    again: bool = False
    # For look/reveal/topdeck: the card the actor most recently played,
    # i.e. whose effect this line belongs to (Harbinger looks through the
    # discard, not the deck).
    context: str | None = None


@dataclass
class OpenPlay:
    """Your most recent phase-level Action play, when the log ends during
    your own turn: `boundary` is the log parsed up to just *before* that
    play (a clean phase-action boundary, `card` still in hand), `events`
    everything from the play itself to the end of the log."""
    card: str
    boundary: "ParsedLog"
    events: list[LogEvent]


@dataclass
class ParsedLog:
    supply: dict[str, int]
    trash: list[str]
    my_total: list[str]
    turns_taken: dict[str, int] = field(default_factory=dict)  # full player name -> completed turns
    my_turns_taken: int = 0
    # Only set when the full replay (layer 2) succeeds end to end.
    my_hand: list[str] | None = None
    my_discard: list[str] | None = None
    my_play_area: list[str] | None = None
    my_phase: str | None = None
    my_actions: int | None = None
    my_buys: int | None = None
    my_coins: int | None = None
    # Merchant plays this turn, and whether a Silver has been played yet --
    # the engine's turn_merchant_bonus/turn_silver_played.
    my_merchant_bonus: int | None = None
    my_silver_played: bool | None = None
    opp_discard: list[str] | None = None
    opp_play_area: list[str] | None = None
    opp_hand_size: int | None = None
    opp_draw_pile_size: int | None = None
    # Cards known to be on top of each deck, next draw first. For the
    # opponent, None marks a card known to be there without knowing which
    # (e.g. their unnamed Artisan topdeck).
    my_deck_top: list[str] | None = None
    opp_deck_top: list[str | None] | None = None
    # Set only when the log ends with the opponent's turn still open and a
    # reaction possibly pending on you (see _SUPPORTED_TERMINAL_ATTACKS) --
    # the ordered PLAY actions of their turn so far (ending in the attack),
    # a snapshot of their discard at the moment their turn started (their
    # play_area/hand_size at turn start are always [] and 5, so nothing to
    # snapshot there -- see training.relay.reconstruct_opponent_turn_boundary),
    # and the top of your deck as it was when their turn started (Bandit's
    # already-revealed cards on top), and the cards their attack itself
    # has gained them so far (Bureaucrat's Silver, Bandit's Gold), which a
    # replay from their turn start gains again.
    pending_reaction_path: list[Action] | None = None
    pending_reaction_opp_discard: list[str] | None = None
    pending_reaction_my_deck_top: list[str] | None = None
    pending_reaction_opp_gains: list[str] | None = None
    open_play: OpenPlay | None = None


def _singularize(word: str) -> str:
    if word in ALL_CARDS:
        return word
    if word.endswith("ies") and (word[:-3] + "y") in ALL_CARDS:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2] in ALL_CARDS:  # Witches
        return word[:-2]
    if word.endswith("s") and word[:-1] in ALL_CARDS:
        return word[:-1]
    raise ValueError(f"not a recognized card name: {word!r}")


_SEGMENT_RE = re.compile(r"^(?:(\d+)|an?)\s+(.+)$", re.IGNORECASE)


def _parse_card_list(text: str) -> list[str]:
    """'3 Coppers and 2 Estates' / 'a Copper, a Silver, and an Estate' ->
    the expanded list of card names. Raises on anything it can't resolve
    to a real card name (e.g. a bare 'a card') rather than guessing."""
    text = text.strip().rstrip(".")
    if not text or text.lower() == "nothing":
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
# named one), "N cards" (all of them unnamed), and the singular "a card"
# (Militia's forced discard down to 3, e.g. "discards a card and a
# Copper") -- dominion.games drops "other" in the one-card case rather
# than saying "1 other card".
_ANON_SEGMENT_RE = re.compile(r"^(?:(\d+)\s+(?:other\s+)?cards?|an?\s+(?:other\s+)?card)$", re.IGNORECASE)


def _parse_card_list_with_anonymous(text: str) -> tuple[list[str], int]:
    """Like `_parse_card_list`, but also tolerates one or more anonymous
    segments (`_ANON_SEGMENT_RE`) -- dominion.games' placeholder for cards
    it won't name from a hidden hand -- returned separately as a count
    rather than real names, since we don't (and, for an opponent's hidden
    hand, can't) know which cards those were."""
    text = text.strip().rstrip(".")
    if not text or text.lower() == "nothing":
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


def _generic_count(text: str) -> int | None:
    m = _GENERIC_DRAW.match(text.strip())
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def _remove_one(zone: list, card) -> bool:
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
    merchant_bonus: int = 0
    silver_played: bool = False


@dataclass
class _OppState:
    hand_size: int
    discard: list[str]
    play_area: list[str]


@dataclass
class _PendingReaction:
    """The log ended with the opponent's turn still open and a reaction
    possibly pending on you -- see _SUPPORTED_TERMINAL_ATTACKS."""
    path: list[Action]
    opp_turn_start_discard: list[str]
    my_deck_top: list[str]
    opp_gains: list[str]


@dataclass
class _Replay:
    me: _MyState
    opp: _OppState
    my_deck_top: list[str]
    opp_deck_top: list[str | None]
    pending_reaction: _PendingReaction | None
    # Line index of your most recent phase-level Action play, if the log
    # ends during your own turn.
    my_open_play_index: int | None


def _replay_full_state(lines: list[str], my_full_name: str, opp_full_name: str, player_of) -> _Replay:
    """Raises ValueError the moment it hits anything it can't resolve
    exactly. Returns final states only if it makes it to the end clean.
    `player_of(token)` maps a line's first token to a full player name, or
    None if it isn't a player abbreviation."""
    me = _MyState(hand=[], discard=[], play_area=[])
    # Starts at 0, not 5: the log's own opening "X draws 5 cards." line
    # (shown before Turn 1) is what brings this up to 5 -- pre-seeding it
    # would double-count that draw.
    opp = _OppState(hand_size=0, discard=[], play_area=[])
    players = (my_full_name, opp_full_name)
    current_turn_player: str | None = None
    # Named cards just taken off a player's deck top (Sentry looks, Bandit
    # reveals), awaiting the trash/discard/topdeck that resolves them --
    # as opposed to a plain hand-sourced one. pending_anon is the same for
    # cards dominion.games didn't name (the opponent's "looks at 2 cards").
    pending_reveal: dict[str, list[str]] = {p: [] for p in players}
    pending_anon: dict[str, int] = {p: 0 for p in players}
    # Cards known to be on each deck's top, next draw first; None is a
    # known-to-be-there but unnamed card.
    known_top: dict[str, list[str | None]] = {p: [] for p in players}
    last_played: dict[str, str | None] = {p: None for p in players}
    # Vassal always discards its deck-top card (logged as a plain
    # "discards X"), and may then play it *from the discard pile* (logged
    # as a plain "plays X"): None, "await" (Vassal played, discard not yet
    # seen), or ("discarded", card) (that play may come next).
    vassal: dict[str, object] = {p: None for p in players}
    # Throne Rooms of mine whose pick (the next "plays X" of mine) hasn't
    # appeared yet -- that play costs no Action and isn't phase-level.
    my_pending_picks = 0
    my_open_play_index: int | None = None

    # Tracking for _PendingReaction: the opponent's PLAY actions so far on
    # their still-open turn, provided every one of them is in
    # _SAFE_MIDTURN_ACTION_CARDS (anything else -- a card that could itself
    # demand a choice from them, or any event touching their zones beyond
    # an automatic part of a safe card -- disqualifies the whole turn, not
    # just that one play). Reset at every _TURN_LINE; the opponent's discard
    # and my deck top are snapshotted there too, since those are the zones
    # that aren't provably constant at their turn start (see
    # training.relay.reconstruct_opponent_turn_boundary).
    current_turn_safe_path: list[Action] = []
    current_turn_self_gains: list[str] = []
    current_turn_disqualified = False
    opp_turn_start_discard: list[str] | None = None
    # True from the opponent's supported attack until a line shows its
    # effect on me resolving (my discard/topdeck/trash/gain, or a Moat
    # reveal blocking it).
    reaction_pending = False

    def is_boundary(next_line: str | None) -> bool:
        # The end of the pasted text is *not* a boundary: a log copied
        # mid-turn ends on whatever just happened (e.g. a Smithy's draw),
        # and the next Turn header is printed the moment a cleanup ends.
        return next_line is not None and bool(_TURN_LINE.match(next_line) or _GAME_END_LINE.match(next_line))

    def cleanup(player: str) -> None:
        if player == my_full_name:
            me.discard.extend(me.hand)
            me.discard.extend(me.play_area)
            me.hand = []
            me.play_area = []
            me.actions, me.buys, me.coins, me.phase = 1, 1, 0, "ACTION"
            me.merchant_bonus, me.silver_played = 0, False
        else:
            opp.discard.extend(opp.play_area)
            opp.play_area = []
            opp.hand_size = 0

    def take_from_top(player: str, cards: list[str | None]) -> None:
        """Cards leaving `player`'s deck top: drop matching known entries.
        A mismatch means the tracking went wrong -- forget it rather than
        keep a wrong belief."""
        top = known_top[player]
        for card in cards:
            if not top:
                return
            if top[0] is None or card is None or top[0] == card:
                top.pop(0)
            else:
                top.clear()
                return

    def put_on_top(player: str, cards: list[str | None]) -> None:
        for card in cards:
            known_top[player].insert(0, card)

    def take_revealed(player: str, card: str) -> bool:
        """`card` resolving a pending reveal/look of `player`'s, if any."""
        if _remove_one(pending_reveal[player], card):
            return True
        if pending_anon[player] > 0:
            pending_anon[player] -= 1
            return True
        return False

    def next_line_resolves(i: int, player: str, card: str) -> bool:
        """Whether `player`'s next line after `i` trashes/discards/topdecks
        `card` -- i.e. a reveal of it was from their deck (Bandit), not a
        Moat shown from hand to block an attack."""
        for later in lines[i + 1:]:
            if _TURN_LINE.match(later) or _GAME_END_LINE.match(later):
                return False
            if player_of(later.split(" ", 1)[0]) != player:
                continue
            m = _TRASH_LINE.match(later) or _DISCARD_LINE.match(later) or _TOPDECK_LINE.match(later)
            if not m:
                return False
            named, _anon = _parse_card_list_with_anonymous(m.group(2))
            return card in named
        return False

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        m = _TURN_LINE.match(line)
        if m:
            current_turn_player = m.group(2)
            current_turn_safe_path = []
            current_turn_self_gains = []
            current_turn_disqualified = False
            reaction_pending = False
            for p in players:
                last_played[p] = None
                pending_reveal[p] = []
                pending_anon[p] = 0
                vassal[p] = None
            my_pending_picks = 0
            my_open_play_index = None
            me.merchant_bonus, me.silver_played = 0, False
            if current_turn_player == opp_full_name:
                opp_turn_start_discard = list(opp.discard)
            continue
        if _GAME_END_LINE.match(line):
            break
        if _STARTS_WITH_LINE.match(line) or _RATING_LINE.match(line):
            continue
        player = player_of(line.split(" ", 1)[0])
        if player is None:
            continue  # a header or other line not about a player's cards
        mine = player == my_full_name
        on_their_turn = player == opp_full_name and current_turn_player == opp_full_name
        attacked_me = mine and current_turn_player == opp_full_name

        # Vassal: the first discard after it is its deck-top card, and a
        # play of that same card right after comes out of the discard pile.
        state = vassal[player]
        if state == "await":
            if _DISCARD_LINE.match(line):
                named, anonymous = _parse_card_list_with_anonymous(_DISCARD_LINE.match(line).group(2))
                if anonymous or len(named) != 1:
                    raise ValueError(f"Vassal's discard wasn't exactly one named card: {line!r}")
                take_from_top(player, named)
                (me.discard if mine else opp.discard).append(named[0])
                vassal[player] = ("discarded", named[0])
                if not mine:
                    current_turn_disqualified = True
                continue
            if not (_GETS_COINS_LINE.match(line) or _SHUFFLE_LINE.match(line)):
                vassal[player] = None  # an empty deck and discard: nothing to discard
        elif state is not None:
            vassal[player] = None
            m = _PLAY_ACTION_LINE.match(line)
            if m and not m.group(3) and _card_name_from_play_line(m.group(2)) == state[1]:
                card = state[1]
                if not _remove_one(me.discard if mine else opp.discard, card):
                    raise ValueError(f"Vassal played {card!r} but it isn't in the tracked discard")
                (me.play_area if mine else opp.play_area).append(card)
                last_played[player] = card
                if mine:
                    me.phase = "ACTION"
                    my_pending_picks += card == "Throne Room"
                    me.merchant_bonus += card == "Merchant"
                else:
                    current_turn_disqualified = True
                if card == "Vassal":
                    vassal[player] = "await"
                continue

        m = _SHUFFLE_LINE.match(line)
        if m:
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
            known_top[player] = []
            if mine:
                me.discard = []
                if upcoming_cleanup:
                    me.play_area = []
                    me.hand = []
                if current_turn_player == opp_full_name:
                    # Only Bandit reshuffles my deck on their turn; the
                    # reaction replay can't reproduce that shuffle's order.
                    current_turn_disqualified = True
            else:
                opp.discard = []
                if upcoming_cleanup:
                    opp.play_area = []
            continue

        m = _PLAY_TREASURE_LINE.match(line)
        if m:
            cards = _parse_card_list(m.group(2))
            last_played[player] = cards[-1] if cards else last_played[player]
            if mine:
                for c in cards:
                    if not _remove_one(me.hand, c):
                        raise ValueError(f"played {c!r} not found in tracked hand")
                me.play_area.extend(cards)
                me.coins += int(m.group(3))
                me.phase = "BUY"
                me.silver_played = me.silver_played or "Silver" in cards
            else:
                opp.hand_size -= len(cards)
                opp.play_area.extend(cards)
                # Treasures are played in the Buy phase, after any attack
                # in the same turn's Action phase would already have
                # appeared -- their turn has moved past any reaction.
                current_turn_disqualified = True
            continue

        m = _PLAY_ACTION_LINE.match(line)
        if m:
            again = bool(m.group(3))
            card_name = _card_name_from_play_line(m.group(2))
            last_played[player] = card_name
            if mine:
                me.phase = "ACTION"  # an action play only ever happens in the action phase
                if not again:
                    if not _remove_one(me.hand, card_name):
                        raise ValueError(f"played {card_name!r} not found in tracked hand")
                    me.play_area.append(card_name)
                    if my_pending_picks:
                        my_pending_picks -= 1  # a Throne Room's pick: no Action spent
                    else:
                        me.actions -= 1
                        my_open_play_index = i
                my_pending_picks += card_name == "Throne Room"
                me.merchant_bonus += card_name == "Merchant"
            else:
                if not again:
                    opp.hand_size -= 1
                    opp.play_area.append(card_name)
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
            if card_name == "Vassal":
                vassal[player] = "await"
            continue

        m = _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line)
        if m:
            is_buy = bool(_BUY_GAIN_LINE.match(line))
            # A buy always lands in discard, and so does anything gained
            # by a player whose turn it isn't; otherwise the card that
            # caused the gain decides (Artisan/Mine to hand, Bureaucrat to
            # the deck top).
            source = None if is_buy or player != current_turn_player else last_played[player]
            to_hand = source in _GAIN_TO_HAND_SOURCES
            to_deck_top = source in _GAIN_TO_DECK_TOP_SOURCES
            cards = _parse_card_list(m.group(2))
            for card in cards:
                if mine:
                    if is_buy:
                        me.buys -= 1
                    if to_hand:
                        me.hand.append(card)
                    elif to_deck_top:
                        put_on_top(player, [card])
                    else:
                        me.discard.append(card)
                else:
                    if to_hand:
                        opp.hand_size += 1
                    elif to_deck_top:
                        put_on_top(player, [card])
                    else:
                        opp.discard.append(card)
            if on_their_turn:
                if not is_buy and cards == [_ATTACK_SELF_GAINS.get(source)]:
                    current_turn_self_gains.extend(cards)
                else:
                    current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # e.g. Witch's Curse: the attack resolved against me
            continue

        m = _REVEALS_HAND_LINE.match(line)
        if m:
            if attacked_me:
                reaction_pending = False  # Bureaucrat found no Victory card: nothing to choose
            continue  # informational only -- nothing moves

        m = _LOOKS_AT_LINE.match(line)
        if m:
            text = m.group(2)
            context = last_played[player]
            if not mine:
                current_turn_disqualified = True
            if context == "Harbinger":
                # Harbinger looks through its owner's own discard pile --
                # already tracked exactly (discard is always public), so
                # there's nothing here to learn from the line's text, which
                # dominion.games often abbreviates as a mix of named cards
                # and an anonymous "N other cards" count. The subsequent
                # topdeck (below) reads straight from the tracked discard.
                continue
            count = _generic_count(text)
            if count is not None:
                if mine:
                    raise ValueError("your own look was unnamed -- can't track exact hand from here")
                take_from_top(player, [None] * count)
                if context == "Library":
                    # Library logs each drawn-and-examined card as a bare,
                    # unnamed "looks at a card." -- it's drawn from the
                    # deck, so it's a draw for hand-size purposes; whether
                    # it's later set aside is resolved separately by an
                    # ordinary named discard line.
                    opp.hand_size += count
                elif context == "Sentry":
                    pending_anon[player] += count  # resolved by the trash/discard/topdeck lines that follow
                else:
                    raise ValueError(f"unnamed 'looks at' after {context!r}: {line!r}")
                continue
            cards = _parse_card_list(text)
            take_from_top(player, cards)
            if context == "Library":
                # Assumed, not yet seen in a real log: your own Library
                # naming each card it examines. They go to hand, like the
                # opponent's; a set-aside one is discarded from there.
                if mine:
                    me.hand.extend(cards)
                else:
                    opp.hand_size += len(cards)
            else:
                pending_reveal[player].extend(cards)
            continue

        m = _REVEALS_LINE.match(line)
        if m:
            revealed = _parse_card_list(m.group(2))
            if (player != current_turn_player and revealed == ["Moat"]
                    and not next_line_resolves(i, player, "Moat")):
                # Moat shown from hand to block an attack: nothing moves.
                if attacked_me:
                    reaction_pending = False
                continue
            take_from_top(player, revealed)
            pending_reveal[player].extend(revealed)
            if not mine:
                current_turn_disqualified = True
            continue

        m = _TRASH_LINE.match(line)
        if m:
            for card in _parse_card_list(m.group(2)):
                from_reveal = take_revealed(player, card)
                if mine:
                    if not from_reveal and not _remove_one(me.hand, card):
                        raise ValueError(f"trashed {card!r} not found in tracked hand")
                elif not from_reveal:
                    opp.hand_size -= 1
            if not mine:
                current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # Bandit's trash choice made
            continue

        m = _DISCARD_LINE.match(line)
        if m:
            named, anonymous = _parse_card_list_with_anonymous(m.group(2))
            for card in named:
                from_reveal = take_revealed(player, card)
                if mine:
                    if not from_reveal and not _remove_one(me.hand, card):
                        raise ValueError(f"discarded {card!r} not found in tracked hand")
                    me.discard.append(card)
                else:
                    if not from_reveal:
                        opp.hand_size -= 1
                    opp.discard.append(card)
            if anonymous:
                if mine:
                    raise ValueError("your own discard included unnamed card(s) -- can't track exact hand from here")
                # A real card each, just not one we can name -- counted out
                # of the opponent's hand, but left out of their tracked
                # discard (which must stay a list of actually-known cards).
                opp.hand_size -= anonymous
            if not mine:
                # A discard of theirs during their own turn can only be
                # self-caused (Cellar/Sentry/Poacher); a reaction they
                # force on *me* is a discard of mine.
                current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # Militia's discard (or Bandit's leftovers) done
            continue

        m = _TOPDECK_LINE.match(line)
        if m:
            text = m.group(2)
            harbinger_context = last_played[player] == "Harbinger" and player == current_turn_player
            if not mine:
                current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # Bureaucrat's topdeck done
            count = _generic_count(text)
            if count is not None:
                if mine:
                    raise ValueError("your own topdeck was unnamed -- can't track exact hand from here")
                if pending_anon[player] >= count:
                    pending_anon[player] -= count  # Sentry putting back cards it looked at
                elif harbinger_context:
                    # dominion.games sometimes renders Harbinger's own
                    # discard-sourced topdeck unnamed. The discard it came
                    # from is tracked exactly, so which card moved doesn't
                    # matter here (only zone *counts* feed reconstruct_game)
                    # -- a don't-care, not a guess.
                    if len(opp.discard) < count:
                        raise ValueError("opponent's unnamed Harbinger topdeck exceeds their tracked discard")
                    del opp.discard[-count:]
                else:
                    opp.hand_size -= count  # Artisan's topdeck from their hidden hand
                put_on_top(player, [None] * count)
                continue
            cards = _parse_card_list(text)
            for card in cards:
                if take_revealed(player, card):
                    continue  # Sentry putting back a card it looked at
                if harbinger_context:
                    if not _remove_one(me.discard if mine else opp.discard, card):
                        raise ValueError(f"topdecked {card!r} not found in tracked discard")
                elif mine:
                    if not _remove_one(me.hand, card):
                        raise ValueError(f"topdecked {card!r} not found in tracked hand")
                else:
                    opp.hand_size -= 1
            put_on_top(player, cards)
            continue

        m = _DRAW_LINE.match(line)
        if m:
            text = m.group(2).strip()
            is_cleanup = player == current_turn_player and is_boundary(next_line)
            count = _generic_count(text)
            if mine:
                if count is not None:
                    raise ValueError("your own draw was unnamed -- can't track exact hand from here")
                cards = _parse_card_list(text)
                if is_cleanup:
                    cleanup(player)
                take_from_top(player, cards)
                me.hand.extend(cards)
            else:
                if count is None:
                    count = len(_parse_card_list(text))
                if is_cleanup:
                    cleanup(player)
                take_from_top(player, [None] * count)
                opp.hand_size += count
            continue

        m = _GETS_ACTIONS_LINE.match(line)
        if m:
            if mine:
                me.actions += int(m.group(2))
            continue

        m = _GETS_BUYS_LINE.match(line)
        if m:
            if mine:
                me.buys += int(m.group(2))
            continue

        m = _GETS_COINS_LINE.match(line)
        if m:
            if mine:
                me.coins += int(m.group(2))
            continue

        # A line about a player's cards that nothing above understood
        # (e.g. a phrasing this module has never seen) could move cards in
        # ways the replay can't follow -- stop rather than drift silently.
        raise ValueError(f"unrecognized log line: {line!r}")

    pending_reaction = None
    if (reaction_pending and not current_turn_disqualified
            and current_turn_player == opp_full_name and opp_turn_start_discard is not None):
        # At their turn start, my deck's top held whatever Bandit has since
        # revealed, then whatever is still known to be there.
        my_top = pending_reveal[my_full_name] + [c for c in known_top[my_full_name] if c is not None]
        pending_reaction = _PendingReaction(current_turn_safe_path, opp_turn_start_discard, my_top,
                                            current_turn_self_gains)
    open_play = my_open_play_index if current_turn_player == my_full_name else None
    return _Replay(me, opp, [c for c in known_top[my_full_name] if c is not None],
                   list(known_top[opp_full_name]), pending_reaction, open_play)


def _event_for_line(line: str, player_of, my_full_name: str, last_played: dict) -> LogEvent | None:
    player = player_of(line.split(" ", 1)[0])
    if player is None:
        return None
    mine = player == my_full_name

    def named(text: str) -> list[str]:
        return _parse_card_list_with_anonymous(text)[0]

    m = _PLAY_TREASURE_LINE.match(line)
    if m:
        return LogEvent(mine, "treasure", _parse_card_list(m.group(2)))
    m = _PLAY_ACTION_LINE.match(line)
    if m:
        card = _card_name_from_play_line(m.group(2))
        last_played[player] = card
        return LogEvent(mine, "play", [card], again=bool(m.group(3)))
    for kind, regex in (("buy", _BUY_GAIN_LINE), ("gain", _GAIN_LINE)):
        m = regex.match(line)
        if m:
            return LogEvent(mine, kind, _parse_card_list(m.group(2)))
    if _REVEALS_HAND_LINE.match(line):
        return LogEvent(mine, "other")
    for kind, regex in (("look", _LOOKS_AT_LINE), ("reveal", _REVEALS_LINE), ("trash", _TRASH_LINE),
                        ("discard", _DISCARD_LINE), ("topdeck", _TOPDECK_LINE), ("draw", _DRAW_LINE)):
        m = regex.match(line)
        if m:
            text = m.group(2)
            cards = [] if (kind == "look" and last_played.get(player) == "Harbinger") or _generic_count(text) \
                else named(text)
            return LogEvent(mine, kind, cards, context=last_played.get(player))
    if _SHUFFLE_LINE.match(line):
        return LogEvent(mine, "shuffle")
    if _GETS_ACTIONS_LINE.match(line) or _GETS_BUYS_LINE.match(line) or _GETS_COINS_LINE.match(line):
        return LogEvent(mine, "gets")
    return LogEvent(mine, "other")


def _resolve_player_names(lines: list[str], my_name: str) -> list[str]:
    """Full player names, from the "name: rating" header when there is one,
    else from the Turn lines, plus -- for a player who hasn't had a turn
    yet in a header-less log -- their abbreviation from the "starts with"
    lines, standing in as their name."""
    names: list[str] = []
    for line in lines:
        if _TURN_LINE.match(line):
            break
        m = _RATING_LINE.match(line)
        if m:
            names.append(m.group(1))
    if not names:
        for line in lines:
            m = _TURN_LINE.match(line)
            if m and m.group(2) not in names:
                names.append(m.group(2))
        for line in lines:
            m = _STARTS_WITH_LINE.match(line)
            if not m:
                continue
            abbrev = m.group(1)
            if any(n[0].lower() == abbrev[0].lower() for n in names):
                continue
            names.append(my_name if my_name[0].lower() == abbrev[0].lower() else abbrev)
    if not names:
        # Nothing to go on at all (e.g. a heavily trimmed practice-game
        # log) -- default to you vs. dominion.games' own built-in bot. If
        # that's wrong, an unrecognized abbreviation downstream surfaces it
        # rather than silently mis-modelling the game.
        names = [my_name, "Lord Rattington"]
    return names


def parse_dominion_log(text: str, my_name: str, kingdom: list[str], num_players: int = 2,
                       _with_open_play: bool = True) -> ParsedLog:
    """`my_name` is your account name as it appears in the log's player/
    rating list (e.g. 'domibot_v1.4') -- matched case-insensitively against
    both the player names and the per-line abbreviations."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    player_names = _resolve_player_names(lines, my_name)
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

    def player_of(token: str) -> str | None:
        return abbrev_to_name.get(token.lower()) if len(token) == 1 else None

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
            turns_taken[m.group(2)] += 1
            continue

        m = _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line)
        if m:
            player = resolve(m.group(1))
            for card in _parse_card_list(m.group(2)):
                supply[card] -= 1
                if player == my_full_name:
                    my_total[card] += 1
            continue

        m = _TRASH_LINE.match(line)
        if m:
            player = resolve(m.group(1))
            for card in _parse_card_list(m.group(2)):
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

    # "turns started" includes the turn in progress; TableState wants
    # completed turns *before* the current one.
    completed = {name: max(count - 1, 0) for name, count in turns_taken.items()}
    result = ParsedLog(
        supply=dict(supply),
        trash=trash,
        my_total=list(Counter(my_total).elements()),
        turns_taken=completed,
        my_turns_taken=completed.get(my_full_name, 0),
    )

    if len(other_names) == 1:
        try:
            replay = _replay_full_state(lines, my_full_name, other_names[0], player_of)
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
            opp = replay.opp
            opp_draw_pile_size = opp_total - opp.hand_size - len(opp.discard) - len(opp.play_area)
            if opp_draw_pile_size < 0:
                raise ValueError("replay produced a negative opponent draw-pile size -- something drifted")
        except ValueError:
            pass
        else:
            me = replay.me
            result.my_hand = me.hand
            result.my_discard = me.discard
            result.my_play_area = me.play_area
            result.my_phase = me.phase
            result.my_actions = me.actions
            result.my_buys = me.buys
            result.my_coins = me.coins
            result.my_merchant_bonus = me.merchant_bonus
            result.my_silver_played = me.silver_played
            result.opp_discard = opp.discard
            result.opp_play_area = opp.play_area
            result.opp_hand_size = opp.hand_size
            result.opp_draw_pile_size = opp_draw_pile_size
            result.my_deck_top = replay.my_deck_top
            result.opp_deck_top = replay.opp_deck_top[:opp_draw_pile_size]
            if replay.pending_reaction is not None:
                result.pending_reaction_path = replay.pending_reaction.path
                result.pending_reaction_opp_discard = replay.pending_reaction.opp_turn_start_discard
                result.pending_reaction_my_deck_top = replay.pending_reaction.my_deck_top
                result.pending_reaction_opp_gains = replay.pending_reaction.opp_gains
            if _with_open_play and replay.my_open_play_index is not None:
                try:
                    result.open_play = _open_play(lines, replay.my_open_play_index, my_name, kingdom,
                                                  num_players, player_of, my_full_name)
                except ValueError:
                    pass

    return result


def _open_play(lines: list[str], k: int, my_name: str, kingdom: list[str], num_players: int,
               player_of, my_full_name: str) -> OpenPlay | None:
    boundary = parse_dominion_log("\n".join(lines[:k]), my_name, kingdom, num_players, _with_open_play=False)
    if boundary.my_hand is None or boundary.my_phase != "ACTION":
        return None
    last_played: dict = {}
    events = [e for e in (_event_for_line(line, player_of, my_full_name, last_played) for line in lines[k:])
              if e is not None]
    return OpenPlay(card=events[0].cards[0], boundary=boundary, events=events)
