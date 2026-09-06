"""Derives the tedious-to-hand-tally parts of a `training.relay.TableState`
-- the supply, the trash pile, and your own total card ownership -- from a
pasted dominion.games text log, instead of counting them by eye. See
`examples/domibot_relay.py`.

Deliberately narrow: only `supply`, `trash`, and `my_total` come from the
log. Everything else (your hand, either player's discard pile, the
opponent's hand/draw-pile sizes) stays a direct on-screen read, because
those are trivially re-checkable at the moment you actually need them,
and some of them are genuinely ambiguous in the log itself -- e.g.
dominion.games sometimes renders a Cellar-style discard as `discards a
card and a Remodel`, naming one card and not the other. Rather than guess
at an unnamed card's identity, this only ever reads events where the log
gives a name -- buys, gains, and trashes, which are always public and
always named -- and ignores everything else (draws, discards, topdecks,
shuffles, plays) since none of those are needed to get supply/trash/
my_total exactly right.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from domibot import ALL_CARDS

_RATING_LINE = re.compile(r"^([\w.\-]+): [\d.]+$")
_TURN_LINE = re.compile(r"^Turn (\d+) - (.+)$")
_BUY_GAIN_LINE = re.compile(r"^(\S+) buys and gains (.+)\.$")
_GAIN_LINE = re.compile(r"^(\S+) gains (.+)\.$")
_TRASH_LINE = re.compile(r"^(\S+) trashes (.+)\.$")

_STARTING_COPPER = 7
_STARTING_ESTATE = 3


@dataclass
class ParsedLog:
    supply: dict[str, int]
    trash: list[str]
    my_total: list[str]
    turns_taken: dict[str, int] = field(default_factory=dict)  # full player name -> turns started so far


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

    return ParsedLog(
        supply=dict(supply),
        trash=trash,
        my_total=list(Counter(my_total).elements()),
        # "turns started" includes the turn in progress; TableState wants
        # completed turns *before* the current one.
        turns_taken={name: max(count - 1, 0) for name, count in turns_taken.items()},
    )
