"""Derive a `relay.TableState` from a pasted dominion.games text log.

Layer 1 reads every buy, gain and trash (always public and named) for the
supply, the trash and your total. Layer 2 replays the log turn by turn for
your hand, discard, play area, phase and counters, the opponent's hand and
draw-pile sizes and discard, and cards known to sit on top of either deck.
The opponent's hand is tracked only as a count. Layer 2 gives up (leaving its
fields None) at anything it can't follow exactly: an unnamed card of yours, or
an unrecognized line. A cleanup is detected positionally: a draw right before
the next `Turn` line or the game-end line. A paste ending on the turn player's
draw that no card of theirs owes is a cleanup too, read as if the next `Turn`
line followed.

When the log ends during your turn, `ParsedLog.open_play` holds the state just
before your last Action play and every line since, for `relay.replay_open_play`.
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
_REVEALS_HAND_LINE = re.compile(r"^(\S+) reveals their hand: (.+)\.$")  # Bureaucrat finding nothing: no move
_REACTS_LINE = re.compile(r"^(\S+) reacts with (?:an? )?(.+)\.$")  # a Moat shown from hand: no move
_GETS_ACTIONS_LINE = re.compile(r"^(\S+) gets \+(\d+) Actions?\.$")
_GETS_BUYS_LINE = re.compile(r"^(\S+) gets \+(\d+) Buys?\.$")
_GETS_COINS_LINE = re.compile(r"^(\S+) gets \+\$(\d+)\.")
_SEGMENT_RE = re.compile(r"^(?:(\d+)|an?)\s+(.+)$", re.IGNORECASE)
# Unnamed cards: "N other cards", "N cards", "a card" (Militia's discard down to 3).
_ANON_SEGMENT_RE = re.compile(r"^(?:(\d+)\s+(?:other\s+)?cards?|an?\s+(?:other\s+)?card)$", re.IGNORECASE)

# Where the current player's gain goes depends on the card that caused it; any other gain goes to discard.
_GAIN_TO_HAND_SOURCES = {"Artisan", "Mine"}
_GAIN_TO_DECK_TOP_SOURCES = {"Bureaucrat"}
# Cards that never ask the player who played them anything, so an opponent's open
# turn of only these can be replayed (Council Room is out: it draws you a card).
_SAFE_MIDTURN_ACTION_CARDS = {"Village", "Smithy", "Festival", "Laboratory", "Market", "Merchant", "Moat", "Witch",
                              "Bureaucrat", "Bandit", "Militia"}
# Attacks that can leave a choice pending on you (and whether to reveal Moat).
_SUPPORTED_TERMINAL_ATTACKS = {"Militia", "Bureaucrat", "Bandit", "Witch"}
# Cards whose own draw is a "draws" line (Library's shows as looks).
_DRAWING_CARDS = {name for name, card in ALL_CARDS.items() if card.plus_cards} | {"Cellar"}
# An attack's own gain, which replaying the attack gains again.
_ATTACK_SELF_GAINS = {"Bureaucrat": "Silver", "Bandit": "Gold"}
_STARTING_COPPER = 7
_STARTING_ESTATE = 3


@dataclass
class LogEvent:
    """One log line, pre-parsed for `relay.replay_open_play`."""
    mine: bool
    kind: str  # play, treasure, draw, look, reveal, trash, discard, gain, buy, topdeck, shuffle, gets, other
    cards: list[str] = field(default_factory=list)
    again: bool = False
    context: str | None = None  # the card the actor last played, whose effect this line belongs to


@dataclass
class OpenPlay:
    """Your last Action play: `boundary` is the log parsed up to just before it,
    `events` everything from the play on."""
    card: str
    boundary: "ParsedLog"
    events: list[LogEvent]


@dataclass
class ParsedLog:
    supply: dict[str, int]
    trash: list[str]
    my_total: list[str]
    turns_taken: dict[str, int] = field(default_factory=dict)  # completed turns per player
    my_turns_taken: int = 0
    # Set only when the full replay (layer 2) succeeds.
    my_hand: list[str] | None = None
    my_discard: list[str] | None = None
    my_play_area: list[str] | None = None
    my_phase: str | None = None
    my_actions: int | None = None
    my_buys: int | None = None
    my_coins: int | None = None
    my_merchant_bonus: int | None = None
    my_silver_played: bool | None = None
    my_bought: bool | None = None
    opp_discard: list[str] | None = None
    opp_play_area: list[str] | None = None
    opp_hand_size: int | None = None
    opp_draw_pile_size: int | None = None
    my_deck_top: list[str] | None = None  # next draw first
    opp_deck_top: list[str | None] | None = None  # None: a card known to be there, unnamed
    opp_known_hand: list[str] | None = None
    # When the log ends on the opponent's open turn with a reaction possibly pending on you:
    # their plays so far, their discard and your deck top at their turn start, and their attack's own gains.
    pending_reaction_path: list[Action] | None = None
    pending_reaction_opp_discard: list[str] | None = None
    pending_reaction_my_deck_top: list[str] | None = None
    pending_reaction_opp_gains: list[str] | None = None
    open_play: OpenPlay | None = None


def remove_one(zone: list, card) -> bool:
    if card in zone:
        zone.remove(card)
        return True
    return False


def _singularize(word: str) -> str:
    if word in ALL_CARDS:
        return word
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", "")):  # Libraries, Witches, Coppers
        if word.endswith(suffix) and word[:-len(suffix)] + replacement in ALL_CARDS:
            return word[:-len(suffix)] + replacement
    raise ValueError(f"not a recognized card name: {word!r}")


def _card_name_from_play_line(raw_name: str) -> str:
    return raw_name if raw_name in ALL_CARDS else _singularize(raw_name)


def _parse_card_list_with_anonymous(text: str) -> tuple[list[str], int]:
    """'3 Coppers, a Silver and 2 other cards' -> (named cards, count of unnamed ones)."""
    text = text.strip().rstrip(".")
    if not text or text.lower() == "nothing":
        return [], 0
    named: list[str] = []
    anonymous = 0
    for segment in text.replace(", and ", ", ").replace(" and ", ", ").split(","):
        segment = segment.strip()
        if not segment:
            continue
        if anon := _ANON_SEGMENT_RE.match(segment):
            anonymous += int(anon.group(1)) if anon.group(1) else 1
            continue
        m = _SEGMENT_RE.match(segment)
        if not m:
            raise ValueError(f"can't parse {segment!r} as a card list segment")
        named.extend([_singularize(m.group(2).strip())] * (int(m.group(1)) if m.group(1) else 1))
    return named, anonymous


def _parse_card_list(text: str) -> list[str]:
    """Like `_parse_card_list_with_anonymous`, but every card must be named."""
    named, anonymous = _parse_card_list_with_anonymous(text)
    if anonymous:
        raise ValueError(f"unnamed card in {text!r}")
    return named


def _generic_count(text: str) -> int | None:
    m = _GENERIC_DRAW.match(text.strip())
    return None if not m else int(m.group(1)) if m.group(1) else 1


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
    bought: bool = False  # treasures still in hand can't be played after a buy


@dataclass
class _OppState:
    hand_size: int
    discard: list[str]
    play_area: list[str]


@dataclass
class _PendingReaction:
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
    opp_known_hand: list[str]
    pending_reaction: _PendingReaction | None
    my_open_play_index: int | None  # line of your last Action play, if the log ends on your turn


def _replay_full_state(lines: list[str], my_full_name: str, opp_full_name: str, player_of) -> _Replay:
    """The replay (layer 2). Raises ValueError at anything it can't resolve exactly.
    `player_of(token)` maps a line's first token to a full player name, or None."""
    me = _MyState(hand=[], discard=[], play_area=[])
    opp = _OppState(hand_size=0, discard=[], play_area=[])  # the log's opening draw brings it to 5
    players = (my_full_name, opp_full_name)
    current_turn_player: str | None = None
    # Cards just taken off a deck top (Sentry looks, Bandit reveals) awaiting the line that resolves
    # them; pending_anon counts unnamed ones.
    pending_reveal: dict[str, list[str]] = {p: [] for p in players}
    pending_anon: dict[str, int] = {p: 0 for p in players}
    known_top: dict[str, list[str | None]] = {p: [] for p in players}  # next draw first; None: unnamed
    last_played: dict[str, str | None] = {p: None for p in players}
    opp_known_hand: list[str] = []
    opp_owned: Counter = Counter({"Copper": _STARTING_COPPER, "Estate": _STARTING_ESTATE})
    # Cards an unnamed Harbinger topdeck took from the opponent's discard, settled later (settle_opp_discard).
    opp_discard_unnamed_out = 0
    # Vassal discards its deck-top card, then may play it from the discard: None, "await", or ("discarded", card).
    vassal: dict[str, object] = {p: None for p in players}
    my_pending_picks = 0  # my Throne Rooms whose pick (a free play) hasn't appeared yet
    my_open_play_index: int | None = None
    # The opponent's open turn, while it holds only safe plays and an attack's own gains
    # (anything else disqualifies it), plus their discard at its start.
    current_turn_safe_path: list[Action] = []
    current_turn_self_gains: list[str] = []
    current_turn_disqualified = False
    opp_turn_start_discard: list[str] | None = None
    reaction_pending = False  # from their supported attack until a line shows it resolved against me

    def is_boundary(next_line: str | None) -> bool:
        # The end of the paste isn't one (`_next_turn_line` adds the header after a cleanup draw).
        return next_line is not None and bool(_TURN_LINE.match(next_line) or _GAME_END_LINE.match(next_line))

    def cleanup(player: str) -> None:
        if player == my_full_name:
            me.discard.extend(me.hand + me.play_area)
            me.hand, me.play_area = [], []
            me.actions, me.buys, me.coins, me.phase = 1, 1, 0, "ACTION"
            me.merchant_bonus, me.silver_played, me.bought = 0, False, False
        else:
            opp.discard.extend(opp.play_area)
            opp.play_area = []
            opp.hand_size = 0
            opp_known_hand.clear()

    def take_from_top(player: str, cards: list[str | None]) -> None:
        """Cards leaving `player`'s deck top; a mismatch means the tracking is off, so forget it."""
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

    def settle_opp_discard() -> list[str]:
        """The opponent's discard minus what unnamed Harbinger topdecks took: first cards
        it shows more copies of than they own, then the most recent (which doesn't matter)."""
        discard = list(opp.discard)
        n = opp_discard_unnamed_out
        excess = Counter(discard) + Counter(opp.play_area)
        excess.subtract(opp_owned)
        for card, extra in excess.items():
            while extra > 0 and n > 0 and remove_one(discard, card):
                extra -= 1
                n -= 1
        return discard[:len(discard) - n]

    def take_revealed(player: str, card: str) -> bool:
        if remove_one(pending_reveal[player], card):
            return True
        if pending_anon[player] > 0:
            pending_anon[player] -= 1
            return True
        return False

    def next_line_resolves(i: int, player: str, card: str) -> bool:
        """Whether `player`'s next line trashes, discards or topdecks `card`: a reveal from
        their deck (Bandit), not a Moat shown from hand."""
        for later in lines[i + 1:]:
            if _TURN_LINE.match(later) or _GAME_END_LINE.match(later):
                return False
            if player_of(later.split(" ", 1)[0]) != player:
                continue
            m = _TRASH_LINE.match(later) or _DISCARD_LINE.match(later) or _TOPDECK_LINE.match(later)
            return bool(m) and card in _parse_card_list_with_anonymous(m.group(2))[0]
        return False

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        m = _TURN_LINE.match(line)
        if m:
            current_turn_player = m.group(2)
            current_turn_safe_path, current_turn_self_gains = [], []
            current_turn_disqualified = reaction_pending = False
            for p in players:
                last_played[p], pending_reveal[p], pending_anon[p], vassal[p] = None, [], 0, None
            my_pending_picks, my_open_play_index = 0, None
            me.merchant_bonus, me.silver_played, me.bought = 0, False, False
            if current_turn_player == opp_full_name:
                opp_turn_start_discard = settle_opp_discard()
            continue
        if _GAME_END_LINE.match(line):
            break
        if _STARTS_WITH_LINE.match(line) or _RATING_LINE.match(line):
            continue
        player = player_of(line.split(" ", 1)[0])
        if player is None:
            continue
        mine = player == my_full_name
        on_their_turn = player == opp_full_name and current_turn_player == opp_full_name
        attacked_me = mine and current_turn_player == opp_full_name

        state = vassal[player]
        if state == "await":  # the first discard after Vassal is its deck-top card
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
                vassal[player] = None  # empty deck and discard: nothing discarded
        elif state is not None:  # a play of that card right after comes from the discard
            vassal[player] = None
            m = _PLAY_ACTION_LINE.match(line)
            if m and not m.group(3) and _card_name_from_play_line(m.group(2)) == state[1]:
                card = state[1]
                if not remove_one(me.discard if mine else opp.discard, card):
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
            # The log prints a cleanup shuffle before the cleanup draw, and never logs the cleanup's
            # discard of hand and play area: those cards went into the new deck, so drop them now.
            upcoming_cleanup = (player == current_turn_player and next_line is not None
                                and bool(_DRAW_LINE.match(next_line))
                                and is_boundary(lines[i + 2] if i + 2 < len(lines) else None))
            known_top[player] = []
            if mine:
                me.discard = []
                if upcoming_cleanup:
                    me.play_area, me.hand = [], []
                if current_turn_player == opp_full_name:
                    current_turn_disqualified = True  # Bandit reshuffled my deck: the replay can't follow
            else:
                opp.discard = []
                opp_discard_unnamed_out = 0
                if upcoming_cleanup:
                    opp.play_area = []
            continue

        m = _PLAY_TREASURE_LINE.match(line)
        if m:
            cards = _parse_card_list(m.group(2))
            last_played[player] = cards[-1] if cards else last_played[player]
            if mine:
                for c in cards:
                    if not remove_one(me.hand, c):
                        raise ValueError(f"played {c!r} not found in tracked hand")
                me.play_area.extend(cards)
                me.coins += int(m.group(3))
                me.phase = "BUY"
                me.silver_played = me.silver_played or "Silver" in cards
            else:
                opp.hand_size -= len(cards)
                opp.play_area.extend(cards)
                for c in cards:
                    remove_one(opp_known_hand, c)
                current_turn_disqualified = True  # their turn is past any attack
            continue

        m = _PLAY_ACTION_LINE.match(line)
        if m:
            again = bool(m.group(3))
            card_name = _card_name_from_play_line(m.group(2))
            last_played[player] = card_name
            if mine:
                me.phase = "ACTION"
                if not again:
                    if not remove_one(me.hand, card_name):
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
                    remove_one(opp_known_hand, card_name)
                # A replay ("again") or an unsafe card could ask them a choice the path can't hold.
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
            source = None if is_buy or player != current_turn_player else last_played[player]
            to_hand, to_deck_top = source in _GAIN_TO_HAND_SOURCES, source in _GAIN_TO_DECK_TOP_SOURCES
            cards = _parse_card_list(m.group(2))
            for card in cards:
                if not mine:
                    opp_owned[card] += 1
                if mine and is_buy:
                    me.buys -= 1
                    me.coins -= ALL_CARDS[card].cost
                    me.phase, me.bought = "BUY", True
                if to_hand:
                    if mine:
                        me.hand.append(card)
                    else:
                        opp.hand_size += 1
                elif to_deck_top:
                    put_on_top(player, [card])
                else:
                    (me.discard if mine else opp.discard).append(card)
            if on_their_turn:
                if not is_buy and cards == [_ATTACK_SELF_GAINS.get(source)]:
                    current_turn_self_gains.extend(cards)
                else:
                    current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # e.g. Witch's Curse: resolved
            continue

        m = _REVEALS_HAND_LINE.match(line)
        if m:
            if attacked_me:
                reaction_pending = False  # Bureaucrat found no Victory card
            if not mine:
                try:
                    opp_known_hand[:] = _parse_card_list(m.group(2))
                except ValueError:
                    pass  # informational only
            continue

        m = _REACTS_LINE.match(line)
        if m:
            if _card_name_from_play_line(m.group(2)) != "Moat":
                raise ValueError(f"unexpected reaction: {line!r}")
            if attacked_me:
                reaction_pending = False  # blocked
            if not mine and "Moat" not in opp_known_hand:
                opp_known_hand.append("Moat")
            continue

        m = _LOOKS_AT_LINE.match(line)
        if m:
            text, context = m.group(2), last_played[player]
            if not mine:
                current_turn_disqualified = True
            if context == "Harbinger":
                continue  # it looks through the (tracked) discard; its topdeck reads from there
            count = _generic_count(text)
            if count is not None:
                if mine:
                    raise ValueError("your own look was unnamed -- can't track exact hand from here")
                take_from_top(player, [None] * count)
                if context == "Library":
                    opp.hand_size += count  # Library's look is a draw; a set-aside card is discarded later
                elif context == "Sentry":
                    pending_anon[player] += count
                else:
                    raise ValueError(f"unnamed 'looks at' after {context!r}: {line!r}")
                continue
            cards = _parse_card_list(text)
            take_from_top(player, cards)
            if context == "Library":  # assumed, not yet seen in a real log: your own named Library looks
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
            if player != current_turn_player and revealed == ["Moat"] and not next_line_resolves(i, player, "Moat"):
                if attacked_me:
                    reaction_pending = False  # Moat shown to block: nothing moves
                if not mine and "Moat" not in opp_known_hand:
                    opp_known_hand.append("Moat")
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
                    if not from_reveal and not remove_one(me.hand, card):
                        raise ValueError(f"trashed {card!r} not found in tracked hand")
                else:
                    if not from_reveal:
                        opp.hand_size -= 1
                        remove_one(opp_known_hand, card)
                    opp_owned[card] -= 1
            if not mine:
                current_turn_disqualified = True
            if attacked_me:
                reaction_pending = False  # Bandit's trash chosen
            continue

        m = _DISCARD_LINE.match(line)
        if m:
            named, anonymous = _parse_card_list_with_anonymous(m.group(2))
            for card in named:
                from_reveal = take_revealed(player, card)
                if mine:
                    if not from_reveal and not remove_one(me.hand, card):
                        raise ValueError(f"discarded {card!r} not found in tracked hand")
                    me.discard.append(card)
                else:
                    if not from_reveal:
                        opp.hand_size -= 1
                        remove_one(opp_known_hand, card)
                    opp.discard.append(card)
            if anonymous:
                if mine:
                    raise ValueError("your own discard included unnamed card(s) -- can't track exact hand from here")
                opp.hand_size -= anonymous  # real cards, but unnamed: kept out of the tracked discard
                opp_known_hand.clear()
            if not mine:
                current_turn_disqualified = True  # self-caused (Cellar, Sentry, Poacher)
            if attacked_me:
                reaction_pending = False  # Militia's discard done
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
                elif harbinger_context:  # an unnamed Harbinger topdeck from the discard: settled later
                    if len(opp.discard) - opp_discard_unnamed_out < count:
                        raise ValueError("opponent's unnamed Harbinger topdeck exceeds their tracked discard")
                    opp_discard_unnamed_out += count
                else:
                    opp.hand_size -= count  # Artisan's topdeck from their hand
                    opp_known_hand.clear()
                put_on_top(player, [None] * count)
                continue
            cards = _parse_card_list(text)
            for card in cards:
                if take_revealed(player, card):
                    continue  # Sentry putting back a card it looked at
                if harbinger_context:
                    if not remove_one(me.discard if mine else opp.discard, card):
                        raise ValueError(f"topdecked {card!r} not found in tracked discard")
                elif mine:
                    if not remove_one(me.hand, card):
                        raise ValueError(f"topdecked {card!r} not found in tracked hand")
                else:
                    opp.hand_size -= 1
                    remove_one(opp_known_hand, card)
            put_on_top(player, cards)
            continue

        m = _DRAW_LINE.match(line)
        if m:
            text = m.group(2).strip()
            is_cleanup = player == current_turn_player and is_boundary(next_line)
            count = _generic_count(text)
            if mine and count is not None:
                raise ValueError("your own draw was unnamed -- can't track exact hand from here")
            cards = _parse_card_list(text) if mine or count is None else [None] * count
            if is_cleanup:
                cleanup(player)
            if mine:
                take_from_top(player, cards)
                me.hand.extend(cards)
            else:
                take_from_top(player, [None] * len(cards))
                opp.hand_size += len(cards)
            continue

        gets = [(_GETS_ACTIONS_LINE, "actions"), (_GETS_BUYS_LINE, "buys"), (_GETS_COINS_LINE, "coins")]
        if any(regex.match(line) for regex, _ in gets):
            for regex, attr in gets:
                if (m := regex.match(line)) and mine:
                    setattr(me, attr, getattr(me, attr) + int(m.group(2)))
            continue
        raise ValueError(f"unrecognized log line: {line!r}")  # it could move cards the replay can't follow

    pending_reaction = None
    if (reaction_pending and not current_turn_disqualified
            and current_turn_player == opp_full_name and opp_turn_start_discard is not None):
        # At their turn start my deck top held what Bandit has since revealed, then what's still known there.
        my_top = pending_reveal[my_full_name] + [c for c in known_top[my_full_name] if c is not None]
        pending_reaction = _PendingReaction(current_turn_safe_path, opp_turn_start_discard, my_top,
                                            current_turn_self_gains)
    open_play = my_open_play_index if current_turn_player == my_full_name else None
    opp.discard = settle_opp_discard()
    return _Replay(me, opp, [c for c in known_top[my_full_name] if c is not None], list(known_top[opp_full_name]),
                   opp_known_hand[:max(opp.hand_size, 0)], pending_reaction, open_play)


def _event_for_line(line: str, player_of, my_full_name: str, last_played: dict) -> LogEvent | None:
    player = player_of(line.split(" ", 1)[0])
    if player is None:
        return None
    mine = player == my_full_name
    m = _PLAY_TREASURE_LINE.match(line)
    if m:
        return LogEvent(mine, "treasure", _parse_card_list(m.group(2)))
    m = _PLAY_ACTION_LINE.match(line)
    if m:
        card = _card_name_from_play_line(m.group(2))
        last_played[player] = card
        return LogEvent(mine, "play", [card], again=bool(m.group(3)))
    for kind, regex in (("buy", _BUY_GAIN_LINE), ("gain", _GAIN_LINE)):
        if m := regex.match(line):
            return LogEvent(mine, kind, _parse_card_list(m.group(2)))
    if _REVEALS_HAND_LINE.match(line):
        return LogEvent(mine, "other")
    for kind, regex in (("look", _LOOKS_AT_LINE), ("reveal", _REVEALS_LINE), ("trash", _TRASH_LINE),
                        ("discard", _DISCARD_LINE), ("topdeck", _TOPDECK_LINE), ("draw", _DRAW_LINE)):
        if m := regex.match(line):
            text = m.group(2)
            unnamed = (kind == "look" and last_played.get(player) == "Harbinger") or _generic_count(text)
            cards = [] if unnamed else _parse_card_list_with_anonymous(text)[0]
            return LogEvent(mine, kind, cards, context=last_played.get(player))
    if _SHUFFLE_LINE.match(line):
        return LogEvent(mine, "shuffle")
    if _GETS_ACTIONS_LINE.match(line) or _GETS_BUYS_LINE.match(line) or _GETS_COINS_LINE.match(line):
        return LogEvent(mine, "gets")
    return LogEvent(mine, "other")


def _next_turn_line(lines: list[str], player_names: list[str], player_of) -> str | None:
    """The next `Turn` line if the paste ends on the turn player's cleanup draw: a draw
    no card of theirs owes (a card's draw comes before their next play or buy)."""
    starts = [i for i, line in enumerate(lines) if _TURN_LINE.match(line)]
    if not starts or len(player_names) != 2 or not _DRAW_LINE.match(lines[-1]):
        return None
    turn_player = _TURN_LINE.match(lines[starts[-1]]).group(2)
    owed = False
    for line in lines[starts[-1] + 1:-1]:
        if player_of(line.split(" ", 1)[0]) != turn_player:
            continue
        if _PLAY_TREASURE_LINE.match(line) or _BUY_GAIN_LINE.match(line) or _DRAW_LINE.match(line):
            owed = False
        elif m := _PLAY_ACTION_LINE.match(line):
            owed = _card_name_from_play_line(m.group(2)) in _DRAWING_CARDS
    if owed or player_of(lines[-1].split(" ", 1)[0]) != turn_player:
        return None
    nxt = next(name for name in player_names if name != turn_player)
    return f"Turn {sum(_TURN_LINE.match(lines[i]).group(2) == nxt for i in starts) + 1} - {nxt}"


def _resolve_player_names(lines: list[str], my_name: str) -> list[str]:
    """Full player names from the "name: rating" header, else from the Turn lines plus,
    for a player without a turn yet, their "starts with" abbreviation."""
    names: list[str] = []
    for line in lines:
        if _TURN_LINE.match(line):
            break
        if m := _RATING_LINE.match(line):
            names.append(m.group(1))
    if not names:
        for line in lines:
            m = _TURN_LINE.match(line)
            if m and m.group(2) not in names:
                names.append(m.group(2))
        for line in lines:
            m = _STARTS_WITH_LINE.match(line)
            if m and not any(n[0].lower() == m.group(1)[0].lower() for n in names):
                names.append(my_name if my_name[0].lower() == m.group(1)[0].lower() else m.group(1))
    return names or [my_name, "Lord Rattington"]  # nothing to go on: assume dominion.games' own bot


def parse_dominion_log(text: str, my_name: str, kingdom: list[str], num_players: int = 2,
                       _with_open_play: bool = True) -> ParsedLog:
    """`my_name` is your account name in the log (case-insensitive)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    player_names = _resolve_player_names(lines, my_name)
    my_matches = [n for n in player_names if n.lower() == my_name.lower()]
    if not my_matches:
        raise ValueError(f"{my_name!r} isn't one of the players in this log: {player_names}")
    my_full_name = my_matches[0]
    other_names = [n for n in player_names if n != my_full_name]

    abbrev_to_name: dict[str, str] = {}  # action lines name players by their first letter
    for name in player_names:
        letter = name[0].lower()
        if letter in abbrev_to_name:
            raise ValueError(f"two players start with the same letter ({name!r} and {abbrev_to_name[letter]!r}) -- "
                             f"this log's per-line abbreviations are ambiguous, can't tell them apart automatically")
        abbrev_to_name[letter] = name

    def resolve(abbrev: str) -> str:
        if abbrev.lower() not in abbrev_to_name:
            raise ValueError(f"unrecognized player abbreviation {abbrev!r} (known: {abbrev_to_name})")
        return abbrev_to_name[abbrev.lower()]

    def player_of(token: str) -> str | None:
        return abbrev_to_name.get(token.lower()) if len(token) == 1 else None

    if next_turn := _next_turn_line(lines, player_names, player_of):
        lines.append(next_turn)
    victory_pile = 8 if num_players == 2 else 12
    supply: Counter = Counter({"Copper": 60 - _STARTING_COPPER * num_players, "Silver": 40, "Gold": 30,
                               "Estate": victory_pile, "Duchy": victory_pile, "Province": victory_pile,
                               "Curse": 10 * (num_players - 1)})
    for name in kingdom:
        supply[name] = victory_pile if CardType.VICTORY in ALL_CARDS[name].types else 10
    total_cards = sum(supply.values()) + (_STARTING_COPPER + _STARTING_ESTATE) * num_players
    trash: list[str] = []
    my_total: Counter = Counter({"Copper": _STARTING_COPPER, "Estate": _STARTING_ESTATE})
    turns_taken: Counter = Counter()
    for line in lines:
        if m := _TURN_LINE.match(line):
            turns_taken[m.group(2)] += 1
        elif m := _BUY_GAIN_LINE.match(line) or _GAIN_LINE.match(line):
            mine = resolve(m.group(1)) == my_full_name
            for card in _parse_card_list(m.group(2)):
                supply[card] -= 1
                if mine:
                    my_total[card] += 1
        elif m := _TRASH_LINE.match(line):
            mine = resolve(m.group(1)) == my_full_name
            for card in _parse_card_list(m.group(2)):
                trash.append(card)
                if mine:
                    my_total[card] -= 1
    for card, count in supply.items():
        if count < 0:
            raise ValueError(f"supply for {card} went negative -- a line in this log wasn't parsed as expected")
    for card, count in my_total.items():
        if count < 0:
            raise ValueError(f"my_total for {card} went negative -- a trash line was likely mis-attributed")

    completed = {name: max(count - 1, 0) for name, count in turns_taken.items()}  # minus the turn in progress
    result = ParsedLog(supply=dict(supply), trash=trash, my_total=list(Counter(my_total).elements()),
                       turns_taken=completed, my_turns_taken=completed.get(my_full_name, 0))
    if len(other_names) != 1:
        return result
    try:
        replay = _replay_full_state(lines, my_full_name, other_names[0], player_of)
        opp = replay.opp
        opp_total = total_cards - sum(supply.values()) - len(trash) - sum(my_total.values())  # by elimination
        opp_draw_pile_size = opp_total - opp.hand_size - len(opp.discard) - len(opp.play_area)
        if opp_draw_pile_size < 0:
            raise ValueError("replay produced a negative opponent draw-pile size -- something drifted")
    except ValueError:
        return result
    me = replay.me
    result.my_hand, result.my_discard, result.my_play_area, result.my_phase = me.hand, me.discard, me.play_area, \
        me.phase
    result.my_actions, result.my_buys, result.my_coins = me.actions, me.buys, me.coins
    result.my_merchant_bonus, result.my_silver_played, result.my_bought = me.merchant_bonus, me.silver_played, \
        me.bought
    result.opp_discard, result.opp_play_area, result.opp_hand_size = opp.discard, opp.play_area, opp.hand_size
    result.opp_draw_pile_size = opp_draw_pile_size
    result.my_deck_top = replay.my_deck_top
    result.opp_deck_top = replay.opp_deck_top[:opp_draw_pile_size]
    result.opp_known_hand = replay.opp_known_hand
    if (r := replay.pending_reaction) is not None:
        result.pending_reaction_path, result.pending_reaction_opp_discard = r.path, r.opp_turn_start_discard
        result.pending_reaction_my_deck_top, result.pending_reaction_opp_gains = r.my_deck_top, r.opp_gains
    if _with_open_play and replay.my_open_play_index is not None:
        try:
            result.open_play = _open_play(lines, replay.my_open_play_index, my_name, kingdom, num_players, player_of,
                                          my_full_name)
        except ValueError:
            pass
    return result


def _open_play(lines: list[str], k: int, my_name: str, kingdom: list[str], num_players: int, player_of,
               my_full_name: str) -> OpenPlay | None:
    boundary = parse_dominion_log("\n".join(lines[:k]), my_name, kingdom, num_players, _with_open_play=False)
    if boundary.my_hand is None or boundary.my_phase != "ACTION":
        return None
    last_played: dict = {}
    events = [e for e in (_event_for_line(line, player_of, my_full_name, last_played) for line in lines[k:])
              if e is not None]
    return OpenPlay(card=events[0].cards[0], boundary=boundary, events=events)
