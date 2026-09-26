"""Edge cases for training/log_parser.py and the relay's use of it, plus a
whole-game regression harness: every one of my turn starts in every real
dominion.games log fixture must parse fully *and* reconstruct into a Game.
"""
from collections import Counter

import pytest

import test_log_parser as T
from domibot import Action
from domibot.enums import DecisionKind
from training.log_parser import parse_dominion_log
from training.mcts import materialize
from training.relay import TableState, reconstruct_game, reconstruct_opponent_turn_boundary, replay_open_play

ME = "domibot_v1.4"

# A real rated game (opponent renamed), where frisco-now-felix's Vassal
# discarded a Chapel and then played it from the discard pile on turn 10 --
# the parser counted that Chapel twice (once discarded, once "played from
# hand") and every query from then on failed reconstruct_game's check. Also
# full of Throne Room + Vassal chains (turns 13, 14, 17, 20) and a Vassal
# playing a Vassal.
USER_VASSAL_KINGDOM = ["Poacher", "Vassal", "Throne Room", "Witch", "Bandit", "Chapel",
                       "Laboratory", "Gardens", "Moat", "Market"]
USER_VASSAL_LOG = """Game #183956265, rated.
domibot_v1.4: 40.73
felix: 44.02
Timer: Patient
Card Pool: level 2
f starts with 3 Estates.
f starts with 7 Coppers.
d starts with 3 Estates.
d starts with 7 Coppers.
f shuffles their deck.
d shuffles their deck.
f draws 5 cards.
d draws 4 Coppers and an Estate.
Turn 1 - felix
f plays 4 Coppers. (+$4)
f buys and gains a Poacher.
f draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Silver.
d draws 3 Coppers and 2 Estates.
Turn 2 - felix
f plays 3 Coppers. (+$3)
f buys and gains a Vassal.
f shuffles their deck.
f draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws 4 Coppers and a Silver.
Turn 3 - felix
f plays 4 Coppers. (+$4)
f buys and gains a Throne Room.
f draws 5 cards.
Turn 3 - domibot_v1.4
d plays a Silver and 4 Coppers. (+$6)
d buys and gains a Witch.
d draws 3 Coppers and 2 Estates.
Turn 4 - felix
f plays a Poacher.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays a Vassal.
f gets +$2.
f discards an Estate.
f plays 3 Coppers. (+$3)
f buys and gains a Witch.
f shuffles their deck.
f draws 5 cards.
Turn 4 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws a Copper, 3 Silvers, and an Estate.
Turn 5 - felix
f plays a Throne Room.
f plays a Poacher.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays a Poacher again.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f plays a Vassal.
f gets +$2.
f discards a Copper.
f plays 4 Coppers. (+$4)
f buys and gains a Province.
f shuffles their deck.
f draws 5 cards.
Turn 5 - domibot_v1.4
d plays 3 Silvers and a Copper. (+$7)
d buys and gains a Bandit.
d draws 3 Coppers, an Estate, and a Witch.
Turn 6 - felix
f plays a Poacher.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays 3 Coppers. (+$3)
f buys and gains a Chapel.
f draws 5 cards.
Turn 6 - domibot_v1.4
d plays a Witch.
d draws 2 Coppers.
f gains a Curse.
d plays 5 Coppers. (+$5)
d buys and gains a Witch.
d shuffles their deck.
d draws 2 Coppers, a Silver, an Estate, and a Witch.
Turn 7 - felix
f plays a Throne Room.
f plays 3 Coppers. (+$3)
f buys and gains a Vassal.
f shuffles their deck.
f draws 5 cards.
Turn 7 - domibot_v1.4
d plays a Witch.
d draws a Copper and a Witch.
f gains a Curse.
d plays a Silver and 3 Coppers. (+$5)
d buys and gains a Laboratory.
d draws a Copper, 2 Silvers, and 2 Estates.
Turn 8 - felix
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f plays a Copper. (+$1)
f draws 5 cards.
Turn 8 - domibot_v1.4
d plays 2 Silvers and a Copper. (+$5)
d buys and gains a Laboratory.
d draws a Curse, 3 Coppers, and a Bandit.
Turn 9 - felix
f plays a Chapel.
f trashes a Copper and an Estate.
f draws 5 cards.
Turn 9 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
f shuffles their deck.
f reveals a Copper and a Poacher.
f discards a Copper and a Poacher.
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws a Copper, 2 Estates, a Bandit, and a Laboratory.
Turn 10 - felix
f plays a Vassal.
f gets +$2.
f discards a Chapel.
f plays a Chapel.
f trashes a Curse and 2 Coppers.
f plays a Copper. (+$1)
f buys and gains a Vassal.
f draws 5 cards.
Turn 10 - domibot_v1.4
d plays a Laboratory.
d draws a Curse and a Gold.
d gets +1 Action.
d plays a Bandit.
d gains a Gold.
f reveals a Curse and a Province.
f discards a Curse and a Province.
d plays a Gold and a Copper. (+$4)
d buys and gains a Silver.
d draws 2 Coppers, a Silver, an Estate, and a Laboratory.
Turn 11 - felix
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f shuffles their deck.
f draws 5 cards.
Turn 11 - domibot_v1.4
d plays a Laboratory.
d draws a Copper and a Silver.
d gets +1 Action.
d plays 2 Silvers and 3 Coppers. (+$7)
d buys and gains a Laboratory.
d draws a Copper, 2 Silvers, and 2 Witches.
Turn 12 - felix
f plays a Vassal.
f gets +$2.
f discards a Copper.
f plays 2 Coppers. (+$2)
f buys and gains a Throne Room.
f draws 5 cards.
Turn 12 - domibot_v1.4
d plays a Witch.
d draws 2 Coppers.
f gains a Curse.
d plays 2 Silvers and 3 Coppers. (+$7)
d buys and gains a Gold.
d shuffles their deck.
d draws a Curse, 2 Coppers, a Silver, and a Laboratory.
Turn 13 - felix
f plays a Throne Room.
f plays a Poacher.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays a Poacher again.
f draws a card.
f gets +1 Action.
f gets +$1.
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f plays a Vassal.
f gets +$2.
f shuffles their deck.
f discards a Throne Room.
f plays a Throne Room.
f plays a Vassal.
f gets +$2.
f discards a Copper.
f plays a Vassal again.
f gets +$2.
f discards an Estate.
f plays a Copper. (+$1)
f buys and gains a Province.
f draws 5 cards.
Turn 13 - domibot_v1.4
d plays a Laboratory.
d draws a Gold and a Laboratory.
d gets +1 Action.
d plays a Laboratory.
d draws an Estate and a Laboratory.
d gets +1 Action.
d plays a Laboratory.
d draws a Copper and a Bandit.
d gets +1 Action.
d plays a Bandit.
d gains a Gold.
f shuffles their deck.
f reveals a Copper and a Poacher.
f discards a Copper and a Poacher.
d plays a Gold, a Silver, and 3 Coppers. (+$8)
d buys and gains a Province.
d draws 2 Coppers, a Silver, an Estate, and a Witch.
Turn 14 - felix
f plays a Vassal.
f gets +$2.
f discards a Vassal.
f plays a Vassal.
f gets +$2.
f discards a Throne Room.
f plays a Throne Room.
f plays 2 Coppers. (+$2)
f buys and gains a Laboratory.
f draws 5 cards.
Turn 14 - domibot_v1.4
d plays a Witch.
d draws a Copper and a Silver.
f gains a Curse.
d plays 2 Silvers and 3 Coppers. (+$7)
d buys and gains a Duchy.
d draws a Curse, a Copper, 2 Golds, and a Witch.
Turn 15 - felix
f shuffles their deck.
f draws 5 cards.
Turn 15 - domibot_v1.4
d plays a Witch.
d draws 2 Silvers.
f gains a Curse.
d plays 2 Golds, 2 Silvers, and a Copper. (+$11)
d buys and gains a Province.
d shuffles their deck.
d draws a Curse, a Copper, a Gold, and 2 Estates.
Turn 16 - felix
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f draws 5 cards.
Turn 16 - domibot_v1.4
d plays a Gold and a Copper. (+$4)
d buys and gains a Gardens.
d draws a Copper, a Silver, a Gold, an Estate, and a Duchy.
Turn 17 - felix
f plays a Laboratory.
f draws 2 cards.
f gets +1 Action.
f plays a Throne Room.
f plays a Vassal.
f gets +$2.
f discards a Poacher.
f plays a Poacher.
f draws a card.
f gets +1 Action.
f gets +$1.
f discards an Estate.
f plays a Vassal again.
f gets +$2.
f discards an Estate.
f plays 3 Coppers. (+$3)
f buys and gains a Province.
f shuffles their deck.
f draws 5 cards.
Turn 17 - domibot_v1.4
d plays a Gold, a Silver, and a Copper. (+$6)
d buys and gains a Duchy.
d draws a Silver, a Gold, a Province, a Laboratory, and a Witch.
Turn 18 - felix
f draws 5 cards.
Turn 18 - domibot_v1.4
d plays a Laboratory.
d draws a Silver and a Laboratory.
d gets +1 Action.
d plays a Laboratory.
d draws a Curse and a Gold.
d gets +1 Action.
d plays a Witch.
d draws a Copper and a Witch.
d plays 2 Golds, 2 Silvers, and a Copper. (+$11)
d buys and gains a Province.
d draws 2 Curses, a Copper, a Silver, and a Province.
Turn 19 - felix
f plays a Vassal.
f gets +$2.
f discards a Province.
f plays a Copper. (+$1)
f buys and gains a Vassal.
f draws 5 cards.
Turn 19 - domibot_v1.4
d plays a Silver and a Copper. (+$3)
d buys and gains a Vassal.
d draws 2 Coppers, a Silver, a Bandit, and a Laboratory.
Turn 20 - felix
f plays a Laboratory.
f draws 2 cards.
f gets +1 Action.
f plays a Throne Room.
f plays a Vassal.
f gets +$2.
f discards a Witch.
f plays a Witch.
f draws 2 cards.
f plays a Vassal again.
f gets +$2.
f discards a Poacher.
f plays a Poacher.
f shuffles their deck.
f draws a card.
f gets +1 Action.
f gets +$1.
f discards a Curse.
f plays a Chapel.
f trashes an Estate.
f plays 3 Coppers. (+$3)
f buys and gains a Province.
f draws 5 cards.
Turn 20 - domibot_v1.4
d plays a Laboratory.
d shuffles their deck.
d draws a Copper and a Gold.
d gets +1 Action.
d plays a Bandit.
d gains a Gold.
f reveals an Estate and a Province.
f discards an Estate and a Province.
d plays a Gold, a Silver, and 3 Coppers. (+$8)
d buys and gains a Province.
d draws a Curse, a Copper, 2 Golds, and a Province.
The game has ended."""


def _state(parsed, kingdom) -> TableState:
    return TableState(
        kingdom=kingdom, supply=parsed.supply, trash=parsed.trash,
        my_hand=parsed.my_hand, my_discard=parsed.my_discard, my_play_area=parsed.my_play_area,
        my_total=parsed.my_total, my_actions=parsed.my_actions, my_buys=parsed.my_buys,
        my_coins=parsed.my_coins, my_phase=parsed.my_phase, my_turns_taken=parsed.my_turns_taken,
        opp_discard=parsed.opp_discard, opp_play_area=parsed.opp_play_area,
        opp_hand_size=parsed.opp_hand_size, opp_draw_pile_size=parsed.opp_draw_pile_size,
        my_deck_top=parsed.my_deck_top or [], opp_deck_top=parsed.opp_deck_top or [],
        my_merchant_bonus=parsed.my_merchant_bonus or 0, my_silver_played=bool(parsed.my_silver_played),
    )


def _prefixes(log: str, pred):
    lines = log.strip().splitlines()
    for i, line in enumerate(lines):
        if pred(line):
            yield line, "\n".join(lines[: i + 1])


def _is_boundary(line: str | None) -> bool:
    return line is not None and (line.startswith("Turn ") or line.startswith("The game has ended"))


def _pasteable_prefixes(log: str):
    """Every prefix a real paste could end with: from the first turn on,
    and never cut between an end-of-turn draw (or the shuffle just before
    it) and the next Turn header, which dominion.games prints together."""
    lines = log.strip().splitlines()
    first_turn = next(i for i, line in enumerate(lines) if line.startswith("Turn "))
    for i in range(first_turn, len(lines)):
        after = lines[i + 1:i + 3] + [None, None]
        if _is_boundary(after[0]) or (" draws " in (after[0] or "") and _is_boundary(after[1])):
            continue
        yield lines[i], "\n".join(lines[: i + 1])


_REAL_LOGS = [
    ("REAL", T.REAL_LOG, T.KINGDOM),
    ("WITCH_MIRROR", T.WITCH_MIRROR_LOG, T.WITCH_MIRROR_KINGDOM),
    ("TWO_THRONES_THEN_VASSAL", T.TWO_THRONES_THEN_VASSAL_LOG, T.TWO_THRONES_THEN_VASSAL_KINGDOM),
    ("HARBINGER", T.HARBINGER_LOG, T.HARBINGER_KINGDOM),
    ("USER_VASSAL", USER_VASSAL_LOG, USER_VASSAL_KINGDOM),
]


@pytest.mark.parametrize("name,log,kingdom", _REAL_LOGS, ids=[c[0] for c in _REAL_LOGS])
def test_every_turn_start_of_mine_parses_and_reconstructs(name, log, kingdom):
    for header, prefix in _prefixes(log, lambda line: line.startswith("Turn ") and line.endswith(ME)):
        parsed = parse_dominion_log(prefix, my_name=ME, kingdom=kingdom)
        assert parsed.my_hand is not None, f"{name}: not fully derived at {header!r}"
        assert parsed.opp_hand_size == 5, f"{name}: opponent hand size at {header!r}"
        reconstruct_game(_state(parsed, kingdom), seed=0)


@pytest.mark.parametrize("name,log,kingdom", _REAL_LOGS, ids=[c[0] for c in _REAL_LOGS])
def test_every_line_of_every_real_log_parses_consistently(name, log, kingdom):
    # Stopping the paste after *any* line (mid-turn included) must still
    # derive a state whose counts add up -- the opponent's draw pile never
    # negative, my tracked zones never exceeding what I own.
    for line, prefix in _pasteable_prefixes(log):
        parsed = parse_dominion_log(prefix, my_name=ME, kingdom=kingdom)
        if parsed.my_hand is None:
            continue
        tracked = Counter(parsed.my_hand) + Counter(parsed.my_discard) + Counter(parsed.my_play_area)
        assert not tracked - Counter(parsed.my_total), f"{name}: my zones exceed my_total after {line!r}"
        assert parsed.opp_draw_pile_size >= 0
        reconstruct_game(_state(parsed, kingdom), seed=0)


def test_user_vassal_log_derives_every_line():
    # Nothing in this real game should force the replay to give up.
    for line, prefix in _pasteable_prefixes(USER_VASSAL_LOG):
        parsed = parse_dominion_log(prefix, my_name=ME, kingdom=USER_VASSAL_KINGDOM)
        assert parsed.my_hand is not None, f"gave up after {line!r}"


def test_opponent_vassal_playing_its_discard_is_counted_once():
    # Turn 10 of the real game: Vassal discards a Chapel, which is then
    # played from the discard pile -- one Chapel, not two.
    lines = USER_VASSAL_LOG.splitlines()
    prefix = "\n".join(lines[: lines.index("Turn 10 - domibot_v1.4") + 1])
    parsed = parse_dominion_log(prefix, my_name=ME, kingdom=USER_VASSAL_KINGDOM)
    assert Counter(parsed.opp_discard) == Counter(["Copper", "Poacher", "Vassal", "Vassal", "Chapel", "Copper"])
    assert parsed.opp_hand_size == 5
    assert sorted(parsed.my_hand) == sorted(["Copper", "Estate", "Estate", "Bandit", "Laboratory"])


def test_opponent_sentry_unnamed_look_leaves_hand_size_alone():
    # "looks at 2 cards" / "trashes a Copper" / "topdecks a card": all three
    # are cards off the deck, none from hand. 5 at turn start, Cellar (-1,
    # discard 2, draw 2) -> 4, Sentry (-1, draw 1) -> 4.
    lines = T.REAL_LOG.strip().splitlines()
    prefix = "\n".join(lines[: lines.index("l topdecks a card.") + 1])
    parsed = parse_dominion_log(prefix, my_name=ME, kingdom=T.KINGDOM)
    assert parsed.opp_hand_size == 4
    assert parsed.opp_deck_top == [None]  # something went back on top, unnamed


def _game(buy1: str, buy2: str, hand3: str, opp_buy1: str = "a Silver",
          opp2: str | None = None, header: bool = True) -> str:
    """A minimal game up to my turn 3, where I hold `hand3`; `opp2` is the
    opponent's turn 2 (their cleanup draw included)."""
    opp2 = opp2 or ("f plays 3 Coppers. (+$3)\nf buys and gains a Silver.\n"
                    "f shuffles their deck.\nf draws 5 cards.")
    head = "Game #1, rated.\ndomibot_v1.4: 40\nfelix: 40\n" if header else "Game #1, unrated.\n"
    return head + f"""d starts with 7 Coppers.
d starts with 3 Estates.
f starts with 7 Coppers.
f starts with 3 Estates.
d shuffles their deck.
f shuffles their deck.
d draws 4 Coppers and an Estate.
f draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains {buy1}.
d draws 3 Coppers and 2 Estates.
Turn 1 - felix
f plays 4 Coppers. (+$4)
f buys and gains {opp_buy1}.
f draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains {buy2}.
d shuffles their deck.
d draws {hand3}.
Turn 2 - felix
{opp2}
Turn 3 - domibot_v1.4"""


EDGE_KINGDOM = ["Vassal", "Village", "Throne Room", "Bureaucrat", "Witch", "Moat", "Cellar",
                "Smithy", "Merchant", "Sentry"]
EDGE_KINGDOM_2 = ["Chapel", "Remodel", "Workshop", "Militia", "Bandit", "Market", "Moneylender",
                  "Library", "Harbinger", "Artisan"]


def _parse(log: str, kingdom=EDGE_KINGDOM, name: str = ME):
    return parse_dominion_log(log, my_name=name, kingdom=kingdom)


def test_my_vassal_plays_its_discard_without_spending_an_action():
    log = _game("a Vassal", "a Village", "a Vassal, 3 Coppers, and an Estate") + """
d plays a Vassal.
d gets +$2.
d discards a Village.
d plays a Village.
d draws a Copper.
d gets +2 Actions."""
    parsed = _parse(log)
    assert parsed.my_actions == 1 - 1 + 2
    assert parsed.my_play_area == ["Vassal", "Village"]
    assert parsed.my_discard == []
    assert sorted(parsed.my_hand) == sorted(["Copper"] * 4 + ["Estate"])
    assert parsed.my_coins == 2


def test_throne_room_pick_spends_no_action():
    log = _game("a Throne Room", "a Village", "a Throne Room, a Village, 2 Coppers, and an Estate") + """
d plays a Throne Room.
d plays a Village.
d draws a Copper.
d gets +2 Actions.
d plays a Village again.
d draws a Copper.
d gets +2 Actions."""
    assert _parse(log).my_actions == 1 - 1 + 2 + 2


def test_witch_curse_goes_to_discard_even_after_my_bureaucrat_turn():
    # My last play of turn 3 is Bureaucrat (no treasure after it), whose
    # gains go to the deck top -- the opponent's Witch Curse must not
    # inherit that routing.
    log = _game("a Bureaucrat", "a Village", "a Bureaucrat, a Village, and 3 Estates",
                opp_buy1="a Witch") + """
d plays a Bureaucrat.
d gains a Silver.
f topdecks an Estate."""
    parsed = _parse(log)
    assert parsed.my_deck_top == ["Silver"]
    assert parsed.opp_deck_top == ["Estate"]
    log += """
d draws a Silver and 4 Coppers.
Turn 3 - felix
f plays a Witch.
f draws 2 cards.
d gains a Curse.
f plays 3 Coppers. (+$3)
f buys and gains a Silver.
f draws 5 cards.
Turn 4 - domibot_v1.4"""
    parsed = _parse(log)
    assert "Curse" in parsed.my_discard
    assert parsed.my_deck_top == []
    reconstruct_game(_state(parsed, EDGE_KINGDOM), seed=0)


def test_unnamed_n_cards_discard_counts_against_hand():
    log = _game("a Militia", "a Silver", "a Militia, 3 Coppers, and an Estate") + """
d plays a Militia.
d gets +$2.
f discards 2 cards."""
    assert _parse(log, EDGE_KINGDOM_2).opp_hand_size == 3


def test_log_ending_on_a_mid_turn_draw_keeps_my_hand():
    log = _game("a Smithy", "a Silver", "a Smithy, 3 Coppers, and an Estate") + """
d plays a Smithy.
d draws 2 Coppers and an Estate."""
    parsed = _parse(log)
    assert sorted(parsed.my_hand) == sorted(["Copper"] * 5 + ["Estate"] * 2)
    assert parsed.my_play_area == ["Smithy"]


def test_unrecognized_line_stops_the_replay_but_keeps_layer_one():
    log = _game("a Smithy", "a Silver", "a Smithy, 3 Coppers, and an Estate") + """
d sets aside a Smithy."""
    parsed = _parse(log)
    assert parsed.my_hand is None
    assert parsed.supply["Smithy"] == 9


def test_headerless_game_against_a_human_takes_names_from_turn_lines():
    parsed = _parse(_game("a Smithy", "a Silver", "a Smithy, 3 Coppers, and an Estate", header=False))
    assert parsed.my_hand is not None
    assert parsed.turns_taken["felix"] == 1


def test_turns_taken_matches_account_name_case_insensitively():
    log = _game("a Smithy", "a Silver", "a Smithy, 3 Coppers, and an Estate")
    assert _parse(log, name="DOMIBOT_V1.4").my_turns_taken == 2


def test_merchant_bonus_carries_into_the_buy_phase():
    log = _game("a Merchant", "a Silver", "a Merchant, a Silver, 2 Coppers, and an Estate") + """
d plays a Merchant.
d draws a Copper.
d gets +1 Action."""
    parsed = _parse(log)
    assert parsed.my_merchant_bonus == 1 and parsed.my_silver_played is False
    game = reconstruct_game(_state(parsed, EDGE_KINGDOM), seed=0)
    game.step(Action("END_ACTIONS"))
    assert game.players[0].coins == 2 + 1 + 3  # Silver, Merchant's bonus, 3 Coppers


def test_buy_phase_reconstruction_plays_the_treasures_left_in_hand():
    log = _game("a Smithy", "a Silver", "a Silver, 3 Coppers, and an Estate") + """
d plays a Copper. (+$1)"""
    parsed = _parse(log)
    assert parsed.my_phase == "BUY" and parsed.my_coins == 1
    game = reconstruct_game(_state(parsed, EDGE_KINGDOM), seed=0)
    assert game.players[0].coins == 1 + 2 + 1 + 1


def test_sentry_topdeck_is_placed_on_top_of_my_deck():
    log = _game("a Sentry", "a Silver", "a Sentry, 3 Coppers, and an Estate") + """
d plays a Sentry.
d draws a Copper.
d gets +1 Action.
d looks at a Silver and an Estate.
d trashes an Estate.
d topdecks a Silver."""
    parsed = _parse(log)
    assert parsed.my_deck_top == ["Silver"]
    assert reconstruct_game(_state(parsed, EDGE_KINGDOM), seed=3).players[0].deck[-1] == "Silver"


# --- replay_open_play: a choice of my own card still waiting on me ---

def _open(log: str, kingdom=EDGE_KINGDOM):
    parsed = _parse(log, kingdom)
    assert parsed.open_play is not None
    result = replay_open_play(parsed.open_play, kingdom, parsed.my_hand, seed=0)
    decision = materialize(result.boundary, result.path).pending_decision if result.status == "pending" else None
    return result, decision


def test_open_chapel_is_pending():
    result, decision = _open(_game("a Chapel", "a Silver", "a Chapel, 3 Coppers, and an Estate")
                             + "\nd plays a Chapel.", EDGE_KINGDOM_2)
    assert result.status == "pending"
    assert {Action("TRASH", "Copper"), Action("TRASH", "Estate"), Action("DONE")} <= set(decision.options)


def test_chapel_whose_trash_is_logged_has_resolved():
    result, _ = _open(_game("a Chapel", "a Silver", "a Chapel, 3 Coppers, and an Estate")
                      + "\nd plays a Chapel.\nd trashes an Estate and a Copper.", EDGE_KINGDOM_2)
    assert result.status == "resolved"


def test_open_throne_room_pick_is_pending():
    result, decision = _open(_game("a Throne Room", "a Village", "a Throne Room, a Village, 2 Coppers, and an Estate")
                             + "\nd plays a Throne Room.")
    assert result.status == "pending"
    assert Action("PLAY", "Village") in decision.options


def test_remodel_gain_is_pending_after_the_logged_trash():
    result, decision = _open(_game("a Remodel", "a Silver", "a Remodel, 3 Coppers, and an Estate")
                             + "\nd plays a Remodel.\nd trashes an Estate.", EDGE_KINGDOM_2)
    assert result.status == "pending"
    assert Action("GAIN", "Silver") in decision.options and Action("GAIN", "Market") not in decision.options


def test_vassal_play_or_not_is_pending_after_its_discard():
    result, decision = _open(_game("a Vassal", "a Village", "a Vassal, 3 Coppers, and an Estate")
                             + "\nd plays a Vassal.\nd gets +$2.\nd discards a Village.")
    assert result.status == "pending"
    assert decision.kind == DecisionKind.YES_NO and decision.source_card == "Vassal"


def test_vassal_that_played_its_discard_has_resolved():
    result, _ = _open(_game("a Vassal", "a Village", "a Vassal, 3 Coppers, and an Estate")
                      + "\nd plays a Vassal.\nd gets +$2.\nd discards a Village.\nd plays a Village."
                      + "\nd draws a Copper.\nd gets +2 Actions.")
    assert result.status == "resolved"


def test_sentry_choice_sees_the_cards_the_log_revealed():
    result, decision = _open(_game("a Sentry", "a Silver", "a Sentry, 3 Coppers, and an Estate")
                             + "\nd plays a Sentry.\nd draws a Copper.\nd gets +1 Action."
                             + "\nd looks at a Silver and an Estate.")
    assert result.status == "pending"
    assert set(decision.options) == {Action("TRASH", "Silver"), Action("TRASH", "Estate"), Action("DONE")}


def test_sentry_discard_step_follows_a_logged_trash():
    result, decision = _open(_game("a Sentry", "a Silver", "a Sentry, 3 Coppers, and an Estate")
                             + "\nd plays a Sentry.\nd draws a Copper.\nd gets +1 Action."
                             + "\nd looks at a Silver and an Estate.\nd trashes an Estate.")
    assert result.status == "pending"
    assert Action("DISCARD", "Silver") in decision.options  # trash step closed; now the discard step


def test_workshop_whose_gain_is_logged_has_resolved():
    result, _ = _open(_game("a Workshop", "a Silver", "a Workshop, 3 Coppers, and an Estate")
                      + "\nd plays a Workshop.\nd gains a Silver.", EDGE_KINGDOM_2)
    assert result.status == "resolved"


# --- the opponent's attack, with my reaction not yet logged ---

def _reaction(log: str, kingdom=EDGE_KINGDOM):
    parsed = _parse(log, kingdom)
    assert parsed.pending_reaction_path is not None
    boundary = reconstruct_opponent_turn_boundary(
        _state(parsed, kingdom), parsed.pending_reaction_opp_discard, path=parsed.pending_reaction_path,
        seed=0, my_deck_top=parsed.pending_reaction_my_deck_top, opp_gains=parsed.pending_reaction_opp_gains)
    return parsed, boundary, materialize(boundary, parsed.pending_reaction_path).pending_decision


def _opponent_turn(body: str, buy1: str = "a Silver", draw3: str = "4 Coppers and an Estate",
                   opp_buy1: str = "a Silver") -> str:
    """Ends partway through the opponent's turn 3 -- their first turn after
    a shuffle, so `opp_buy1` can be in their hand -- with `body` being its
    lines so far, while I hold `draw3`. My deck then holds 4 Coppers, 2
    Estates and `buy1`, minus `draw3`."""
    return _game(buy1, "a Silver", "a Silver, 3 Coppers, and an Estate", opp_buy1=opp_buy1) + f"""
d plays a Silver and 3 Coppers. (+$5)
d buys and gains a Silver.
d draws {draw3}.
Turn 3 - felix
{body}"""


def test_bureaucrat_topdeck_is_a_pending_reaction():
    parsed, boundary, decision = _reaction(_opponent_turn("f plays a Bureaucrat.\nf gains a Silver.",
                                                          opp_buy1="a Bureaucrat"))
    assert parsed.pending_reaction_opp_gains == ["Silver"]
    assert boundary.supply["Silver"] == parsed.supply["Silver"] + 1
    assert decision.player == 0 and decision.options == [Action("TOPDECK", "Estate")]


def test_bandit_trash_choice_sees_my_revealed_cards():
    _, _, decision = _reaction(_opponent_turn("f plays a Bandit.\nf gains a Gold.\nd reveals a Silver and an Estate.",
                                              opp_buy1="a Bandit"), EDGE_KINGDOM_2)
    assert decision.player == 0 and decision.options == [Action("TRASH", "Silver")]


def test_moat_reveal_is_a_pending_reaction_to_witch():
    log = _opponent_turn("f plays a Witch.", buy1="a Moat", draw3="a Moat, 3 Coppers, and an Estate",
                         opp_buy1="a Witch")
    _, _, decision = _reaction(log)
    assert decision.player == 0 and decision.kind == DecisionKind.REACT


def test_moat_reveal_to_block_leaves_moat_in_hand():
    log = _opponent_turn("f plays a Witch.\nd reveals a Moat.\nf draws 2 cards.", buy1="a Moat",
                         draw3="a Moat, 3 Coppers, and an Estate", opp_buy1="a Witch")
    parsed = _parse(log)
    assert "Moat" in parsed.my_hand
    assert parsed.pending_reaction_path is None  # blocked: nothing left to decide
    log += """
f plays 3 Coppers. (+$3)
f buys and gains a Silver.
f draws 5 cards.
Turn 4 - domibot_v1.4
d plays a Moat.
d draws a Copper and an Estate."""
    parsed = _parse(log)
    assert sorted(parsed.my_hand) == sorted(["Copper"] * 4 + ["Estate"] * 2)
    reconstruct_game(_state(parsed, EDGE_KINGDOM), seed=0)
