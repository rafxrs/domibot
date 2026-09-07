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
   discard, except Artisan/Mine which go to hand), trashes/discards/
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

from domibot import ALL_CARDS

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
_GETS_ACTIONS_LINE = re.compile(r"^(\S+) gets \+(\d+) Actions?\.$")
_GETS_BUYS_LINE = re.compile(r"^(\S+) gets \+(\d+) Buys?\.$")
_GETS_COINS_LINE = re.compile(r"^(\S+) gets \+\$(\d+)\.")

# Cards whose gain destination is hand, not discard (per src/domibot/cards/kingdom.py).
_GAINS_TO_HAND = {"Artisan", "Mine"}

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


def _replay_full_state(lines: list[str], my_full_name: str, opp_full_name: str, resolve) -> tuple[_MyState, _OppState]:
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
            continue
        if _GAME_END_LINE.match(line) or _STARTS_WITH_LINE.match(line) or _RATING_LINE.match(line):
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
            # anyway), so this is just "discard is now empty."
            if player == my_full_name:
                me.discard = []
                if upcoming_cleanup:
                    me.play_area = []
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
                    opp.hand_size -= 1
                    opp.play_area.append(card_name)
            continue

        m = _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            is_buy = bool(_BUY_GAIN_LINE.match(line))
            for card in _parse_card_list(card_text):
                if player == my_full_name:
                    if is_buy:
                        me.buys -= 1
                    (me.hand if card in _GAINS_TO_HAND else me.discard).append(card)
                else:
                    if card in _GAINS_TO_HAND:
                        opp.hand_size += 1
                    else:
                        opp.discard.append(card)
            continue

        m = _REVEALS_LINE.match(line) or _LOOKS_AT_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            pending_reveal[resolve(abbrev)].extend(_parse_card_list(card_text))
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
                elif not from_reveal:
                    opp.hand_size -= 1
            continue

        m = _DISCARD_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            for card in _parse_card_list(card_text):
                from_reveal = _remove_one(pending_reveal[player], card)
                if player == my_full_name:
                    if not from_reveal and not _remove_one(me.hand, card):
                        raise ValueError(f"discarded {card!r} not found in tracked hand")
                    me.discard.append(card)
                else:
                    if not from_reveal:
                        opp.hand_size -= 1
                    opp.discard.append(card)
            continue

        m = _TOPDECK_LINE.match(line)
        if m:
            abbrev, card_text = m.groups()
            player = resolve(abbrev)
            harbinger_context = last_played.get(player) == "Harbinger"
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

    return me, opp


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
        raise ValueError("couldn't find the 'name: rating' lines at the top of the log")

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

    supply: Counter = Counter({"Copper": 60 - _STARTING_COPPER * num_players,
                                "Silver": 40, "Gold": 30,
                                "Estate": 8 if num_players == 2 else 12,
                                "Duchy": 8 if num_players == 2 else 12,
                                "Province": 8 if num_players == 2 else 12,
                                "Curse": 10 * (num_players - 1)})
    for name in kingdom:
        supply[name] = 10

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
            me, opp = _replay_full_state(lines, my_full_name, other_names[0], resolve)
            # Every card is in exactly one of {supply, trash, mine, theirs} --
            # solved by elimination, the same trick reconstruct_game uses.
            total_sum = (
                60 + 40 + 30
                + (8 if num_players == 2 else 12) + _STARTING_ESTATE * num_players
                + (8 if num_players == 2 else 12)
                + (8 if num_players == 2 else 12)
                + 10 * (num_players - 1)
                + 10 * len(kingdom)
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

    return result
