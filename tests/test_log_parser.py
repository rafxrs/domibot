from collections import Counter
from pathlib import Path

import pytest

from domibot import Action
from domibot.models import END_ACTIONS
from training.log_parser import parse_dominion_log
from training.relay import resolve_card_name

FIXTURES = Path(__file__).parent / "fixtures" / "dominion_logs"


def load_log(name: str) -> tuple[str, list[str]]:
    """A saved dominion.games log; its first line is the kingdom in short codes."""
    codes, log = (FIXTURES / name).read_text(encoding="utf-8").split("\n", 1)
    return log, [resolve_card_name(code) for code in codes.split()]


# Real rated games (opponents renamed): the first turns of one; a Witch mirror with Throne Room
# chains and Bandit reveals; two Throne Rooms and a Vassal in one turn; an opponent's Harbinger with
# unnamed cards.
REAL_LOG, KINGDOM = load_log("first_turns_trimmed.txt")
WITCH_MIRROR_LOG, WITCH_MIRROR_KINGDOM = load_log("witch_mirror_bandit_reveals.txt")
TWO_THRONES_THEN_VASSAL_LOG, TWO_THRONES_THEN_VASSAL_KINGDOM = load_log("two_thrones_then_vassal.txt")
HARBINGER_LOG, HARBINGER_KINGDOM = load_log("harbinger_unnamed_topdeck.txt")


def test_parses_real_log_my_total_matches_hand_derivation():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)

    # Counted by hand: 7 Coppers and 3 Estates, plus 10 gains, minus 4 Coppers and 2 Estates trashed.
    from collections import Counter
    total = Counter(parsed.my_total)
    assert sum(total.values()) == 14
    assert total["Copper"] == 3
    assert total["Estate"] == 1
    assert total["Silver"] == 1
    assert total["Gold"] == 2
    assert total["Sentry"] == 2
    assert total["Artisan"] == 2
    assert total["Bandit"] == 1
    assert total["Moneylender"] == 1
    assert total["Smithy"] == 1


def test_parses_real_log_trash_pile():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    from collections import Counter
    trash = Counter(parsed.trash)
    # Counted by hand from the Remodel, Sentry, Moneylender and Bandit trashes.
    assert trash["Estate"] == 4
    assert trash["Silver"] == 4
    assert trash["Copper"] == 5
    assert trash["Remodel"] == 1
    assert sum(trash.values()) == 14


def test_parses_real_log_supply_decremented_for_both_players_buys():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    # Gains (Remodel's too) come from the supply as well as buys.
    assert parsed.supply["Sentry"] < 10
    assert parsed.supply["Bandit"] < 10
    assert parsed.supply["Artisan"] == 10 - 2  # d bought 2 (T8, T10); l never touched Artisan


def test_gardens_supply_starts_at_victory_pile_size_not_ten():
    # Gardens has a Victory pile (8 in a 2-player game), not 10.
    log = "\n".join("d buys and gains a Gardens." for _ in range(8))
    kingdom = ["Gardens", "Smithy", "Remodel", "Bandit", "Moneylender", "Sentry",
               "Cellar", "Artisan", "Village", "Chapel"]
    parsed = parse_dominion_log(log, my_name="domibot_v1.4", kingdom=kingdom)
    assert parsed.supply["Gardens"] == 0


def test_parses_real_log_turns_taken():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    # Completed turns: the turn in progress doesn't count.
    assert parsed.turns_taken["domibot_v1.4"] == 9
    assert parsed.turns_taken["lololololo"] == 10


def test_unknown_player_name_raises():
    with pytest.raises(ValueError, match="isn't one of the players"):
        parse_dominion_log(REAL_LOG, my_name="someone_else", kingdom=KINGDOM)


def test_ambiguous_first_letters_raise():
    log = "dave: 30.0\ndan: 25.0\nTurn 1 - dave\n"
    with pytest.raises(ValueError, match="same letter"):
        parse_dominion_log(log, my_name="dave", kingdom=KINGDOM)


def test_my_hand_still_derived_when_log_ends_mid_opponents_turn():
    # The log ends mid-way through the opponent's turn, which doesn't touch my hand.
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    assert sorted(parsed.my_hand) == sorted(["Copper", "Silver", "Gold", "Estate", "Artisan"])
    assert parsed.my_phase == "ACTION"


FRESH_TURN_LOG = """
Game #1, rated.
Pogonomyrmex: 48.26
domibot_v1.4: 44.22
Timer: Patient
Card Pool: level 1
d starts with 3 Estates.
d starts with 7 Coppers.
P starts with 3 Estates.
P starts with 7 Coppers.
P shuffles their deck.
d shuffles their deck.
P draws 5 cards.
d draws 4 Coppers and an Estate.
Turn 1 - Pogonomyrmex
P plays 5 Coppers. (+$5)
P buys and gains a Market.
P draws 5 cards.
Turn 1 - domibot_v1.4
"""


FRESH_TURN_KINGDOM = KINGDOM[:-1] + ["Market"]  # the log's one buy (Market) must be a real kingdom card


def test_my_hand_derived_when_log_ends_at_a_fresh_turn_for_me():
    parsed = parse_dominion_log(FRESH_TURN_LOG, my_name="domibot_v1.4", kingdom=FRESH_TURN_KINGDOM)
    assert sorted(parsed.my_hand) == sorted(["Copper", "Copper", "Copper", "Copper", "Estate"])
    assert parsed.my_phase == "ACTION"
    assert parsed.my_actions == 1
    assert parsed.my_buys == 1
    assert parsed.my_coins == 0
    assert parsed.my_play_area == []


def test_my_hand_not_derived_when_the_fresh_turn_is_the_opponents():
    parsed = parse_dominion_log(FRESH_TURN_LOG, my_name="Pogonomyrmex", kingdom=FRESH_TURN_KINGDOM)
    assert parsed.my_hand is None


def test_my_hand_not_derived_from_a_generic_draw():
    log = FRESH_TURN_LOG.replace("d draws 4 Coppers and an Estate.", "d draws 5 cards.")
    parsed = parse_dominion_log(log, my_name="domibot_v1.4", kingdom=FRESH_TURN_KINGDOM)
    assert parsed.my_hand is None





def test_witch_mirror_full_game_derives_final_hand():
    parsed = parse_dominion_log(WITCH_MIRROR_LOG, my_name="domibot_v1.4", kingdom=WITCH_MIRROR_KINGDOM)
    # last line: "d draws 2 Curses, a Copper, a Silver, and a Province." at
    # the true end of d's turn 23, immediately followed by game-end.
    assert sorted(parsed.my_hand) == sorted(["Curse", "Curse", "Copper", "Silver", "Province"])
    assert parsed.my_phase == "ACTION"
    assert parsed.my_actions == 1 and parsed.my_buys == 1 and parsed.my_coins == 0


def test_witch_mirror_mid_turn_stop_after_council_room():
    lines = WITCH_MIRROR_LOG.splitlines()
    truncated = "\n".join(lines[: lines.index("d gets +1 Buy.") + 1])  # first occurrence: turn 7
    parsed = parse_dominion_log(truncated, my_name="domibot_v1.4", kingdom=WITCH_MIRROR_KINGDOM)
    # Turn 6's cleanup draw, minus the Council Room played, plus its 4 cards.
    assert sorted(parsed.my_hand) == sorted(
        ["Copper", "Copper", "Silver", "Estate", "Curse", "Silver", "Estate", "Witch"]
    )
    assert parsed.my_phase == "ACTION"
    assert parsed.my_actions == 0  # spent on Council Room, which grants no +actions
    assert parsed.my_buys == 2  # base 1 + Council Room's +1 Buy
    assert parsed.my_coins == 0  # no treasure played yet


def test_witch_mirror_bandit_reveal_does_not_touch_hand():
    lines = WITCH_MIRROR_LOG.splitlines()
    truncated = "\n".join(lines[: lines.index("d discards a Copper.") + 1])
    parsed = parse_dominion_log(truncated, my_name="domibot_v1.4", kingdom=WITCH_MIRROR_KINGDOM)
    # The opponent's Bandit takes from my deck top, never my hand.
    assert sorted(parsed.my_hand) == sorted(["Curse", "Curse", "Silver", "Silver", "Bandit"])
    assert "Copper" in parsed.my_discard
    assert parsed.trash[-1] == "Silver"


def test_witch_mirror_reconstructs_without_error():
    from training.relay import TableState, reconstruct_game

    parsed = parse_dominion_log(WITCH_MIRROR_LOG, my_name="domibot_v1.4", kingdom=WITCH_MIRROR_KINGDOM)
    state = TableState(
        kingdom=WITCH_MIRROR_KINGDOM, supply=parsed.supply, trash=parsed.trash,
        my_hand=parsed.my_hand, my_discard=parsed.my_discard, my_play_area=parsed.my_play_area,
        my_total=parsed.my_total, my_actions=parsed.my_actions, my_buys=parsed.my_buys,
        my_coins=parsed.my_coins, my_phase=parsed.my_phase,
        my_turns_taken=parsed.turns_taken["domibot_v1.4"],
        opp_discard=parsed.opp_discard, opp_play_area=parsed.opp_play_area,
        opp_hand_size=parsed.opp_hand_size, opp_draw_pile_size=parsed.opp_draw_pile_size,
    )
    game = reconstruct_game(state, seed=1)
    assert game.legal_actions() == [END_ACTIONS]


def test_opponent_name_with_a_space_is_recognized():
    # A name with a space once hid the second player entirely.
    log = """Game #183190908, unrated.
domibot_v1.4: 44
Lord Rattington: 40
Timer: Off
Card Pool: level 2
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 2 Coppers and 3 Estates.
L draws 5 cards.
Turn 1 - domibot_v1.4"""
    kingdom = ["Bandit", "Festival", "Library", "Artisan", "Vassal",
               "Bureaucrat", "Moneylender", "Remodel", "Cellar", "Harbinger"]
    parsed = parse_dominion_log(log, my_name="domibot_v1.4", kingdom=kingdom)
    assert sorted(parsed.my_hand) == sorted(["Copper", "Copper", "Estate", "Estate", "Estate"])
    assert parsed.my_phase == "ACTION" and parsed.my_actions == 1
    assert parsed.opp_hand_size == 5 and parsed.opp_draw_pile_size == 5


def test_missing_header_defaults_to_lord_rattington():
    # A trimmed log with no "name: rating" lines at all -- default to you
    # (whatever --account-name is) vs. dominion.games' own bot.
    log = """d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 2 Coppers and 3 Estates.
L draws 5 cards.
Turn 1 - domibot_v1.4"""
    kingdom = ["Bandit", "Festival", "Library", "Artisan", "Vassal",
               "Bureaucrat", "Moneylender", "Remodel", "Cellar", "Harbinger"]
    parsed = parse_dominion_log(log, my_name="domibot_v1.4", kingdom=kingdom)
    assert sorted(parsed.my_hand) == sorted(["Copper", "Copper", "Estate", "Estate", "Estate"])
    assert parsed.opp_hand_size == 5


# My Bureaucrat twice, finding no Victory card ("reveals their hand: ..."); its Silver goes to my deck top.
BUREAUCRAT_LOG = """Game #183190908, unrated.
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 2 Coppers and 3 Estates.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 2 Coppers. (+$2)
d buys and gains a Copper.
d draws 5 Coppers.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Silver.
L draws 5 cards.
Turn 2 - domibot_v1.4
d plays 5 Coppers. (+$5)
d buys and gains a Bandit.
d shuffles their deck.
d draws 4 Coppers and a Bandit.
Turn 2 - Lord Rattington
L plays 3 Coppers. (+$3)
L buys and gains a Silver.
L shuffles their deck.
L draws 5 cards.
Turn 3 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
L reveals 2 Coppers.
L discards 2 Coppers.
d plays 4 Coppers. (+$4)
d buys and gains a Bureaucrat.
d draws 3 Coppers and 2 Estates.
Turn 3 - Lord Rattington
L plays 3 Coppers and 2 Silvers. (+$7)
L buys and gains a Bandit.
L draws 5 cards.
Turn 4 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws 3 Coppers, an Estate, and a Bandit.
Turn 4 - Lord Rattington
L plays 2 Coppers. (+$2)
L shuffles their deck.
L draws 5 cards.
Turn 5 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
L reveals a Copper and an Estate.
L discards a Copper and an Estate.
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d draws 3 Coppers, a Gold, and a Bureaucrat.
Turn 5 - Lord Rattington
L plays a Bandit.
L gains a Gold.
d reveals 2 Coppers.
d discards 2 Coppers.
L plays a Copper and a Silver. (+$3)
L buys and gains a Silver.
L draws 5 cards.
Turn 6 - domibot_v1.4
d plays a Bureaucrat.
d gains a Silver.
L reveals their hand: 5 Coppers.
d plays 3 Coppers and a Gold. (+$6)
d buys and gains a Gold.
d shuffles their deck.
d draws 2 Silvers, 2 Estates, and a Bureaucrat.
Turn 6 - Lord Rattington
L plays 5 Coppers. (+$5)
L buys and gains a Library.
L shuffles their deck.
L draws 5 cards.
Turn 7 - domibot_v1.4
d plays a Bureaucrat.
d gains a Silver.
L reveals their hand: 2 Coppers and 3 Silvers.
d plays 2 Silvers. (+$4)
d buys and gains a Bureaucrat.
d draws 3 Coppers, a Silver, and a Bandit.
Turn 7 - Lord Rattington
L plays 2 Coppers and 3 Silvers. (+$8)
L buys and gains a Province.
L draws 5 cards.
Turn 8 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
L reveals an Estate and a Bandit.
L discards an Estate and a Bandit.
d plays 3 Coppers and a Silver. (+$5)
d buys and gains a Festival.
d draws 2 Coppers, a Silver, a Gold, and an Estate.
Turn 8 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Silver.
L shuffles their deck.
L draws 5 cards.
Turn 9 - domibot_v1.4"""

BUREAUCRAT_KINGDOM = ["Bandit", "Bureaucrat", "Festival", "Library"]

# Two more turns, into the opponent's Library, whose looks are unnamed ("looks at a card.").
LIBRARY_LOG = BUREAUCRAT_LOG + """
d plays 2 Coppers, a Gold, and a Silver. (+$7)
d buys and gains a Gold.
d draws 3 Coppers and 2 Golds.
Turn 9 - Lord Rattington
L plays a Library.
L looks at a card.
L looks at a card.
L looks at a card.
L plays 2 Coppers, a Gold, and 2 Silvers. (+$9)
L buys and gains a Province.
L draws 5 cards.
Turn 10 - domibot_v1.4"""


def test_library_unnamed_looks_at_a_card_does_not_crash_the_replay():
    parsed = parse_dominion_log(LIBRARY_LOG, my_name="domibot_v1.4", kingdom=BUREAUCRAT_KINGDOM)
    assert parsed.my_phase == "ACTION"
    # The log ends after a cleanup; the mid-turn count below is the real check.
    assert parsed.opp_hand_size == 5
    tracked = Counter(parsed.my_hand) + Counter(parsed.my_discard) + Counter(parsed.my_play_area)
    my_total = Counter(parsed.my_total)
    for card, count in tracked.items():
        assert count <= my_total[card], f"{card}: tracked {count} exceeds my_total {my_total[card]}"


# A shuffle during cleanup, logged before the cleanup draw (and before the unlogged discard).
SHUFFLE_CLEANUP_LOG = """Game #183209524, unrated.
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 4 Coppers and an Estate.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Bureaucrat.
d draws 3 Coppers and 2 Estates.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Chapel.
L draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws 3 Coppers, a Silver, and a Bureaucrat.
Turn 2 - Lord Rattington
L plays 3 Coppers. (+$3)
L buys and gains a Silver.
L shuffles their deck.
L draws 5 cards.
Turn 3 - domibot_v1.4"""

SHUFFLE_CLEANUP_KINGDOM = ["Bandit", "Bureaucrat", "Chapel", "Militia", "Laboratory"]


def test_cleanup_shuffle_does_not_resurrect_leftover_hand_into_discard():
    # The unplayed hand went into the new deck with the shuffle; it must not reappear in the discard.
    parsed = parse_dominion_log(SHUFFLE_CLEANUP_LOG, my_name="domibot_v1.4", kingdom=SHUFFLE_CLEANUP_KINGDOM)
    tracked = Counter(parsed.my_hand) + Counter(parsed.my_discard) + Counter(parsed.my_play_area)
    my_total = Counter(parsed.my_total)
    for card, count in tracked.items():
        assert count <= my_total[card], f"{card}: tracked {count} exceeds my_total {my_total[card]}"
    assert parsed.my_discard.count("Estate") == 0  # both got folded into the untracked deck


def test_library_looks_at_a_card_increments_opponent_hand_size():
    lines = LIBRARY_LOG.splitlines()
    cutoff = lines.index("L plays 2 Coppers, a Gold, and 2 Silvers. (+$9)")
    parsed = parse_dominion_log("\n".join(lines[:cutoff]), my_name="domibot_v1.4", kingdom=BUREAUCRAT_KINGDOM)
    # Turn 9 opponent hand started at 5, minus the played Library, plus
    # the 3 drawn-and-examined cards.
    assert parsed.opp_hand_size == 5 - 1 + 3





def test_two_thrones_then_vassal_still_fully_derives():
    parsed = parse_dominion_log(TWO_THRONES_THEN_VASSAL_LOG, my_name="domibot_v1.4",
                                 kingdom=TWO_THRONES_THEN_VASSAL_KINGDOM)
    assert parsed.my_hand is not None
    assert parsed.opp_discard is not None


VASSAL_HAND_SIZE_KINGDOM = ["Vassal", "Village", "Market", "Moneylender", "Chapel", "Witch",
                            "Militia", "Moat", "Council Room", "Festival"]

VASSAL_DECLINE_LOG = """Game #1, unrated.
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 4 Coppers and an Estate.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Silver.
d draws 3 Coppers and 2 Estates.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Vassal.
L draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d draws 3 Coppers, an Estate, and a Silver.
Turn 2 - Lord Rattington
L plays a Vassal.
L gets +$2.
L discards a Copper."""


def test_vassal_reveal_discard_does_not_decrement_opponent_hand_size():
    # Vassal's discard comes off the deck top, not the hand.
    parsed = parse_dominion_log(VASSAL_DECLINE_LOG, my_name="domibot_v1.4",
                                 kingdom=VASSAL_HAND_SIZE_KINGDOM)
    assert parsed.opp_hand_size is not None
    # 5 cards, minus the Vassal played.
    assert parsed.opp_hand_size == 4





def test_opponent_harbinger_with_anonymous_look_and_unnamed_topdeck_still_derives():
    # Harbinger looks through the (tracked) discard, so its partly unnamed lines don't matter.
    parsed = parse_dominion_log(HARBINGER_LOG, my_name="domibot_v1.4", kingdom=HARBINGER_KINGDOM)
    assert parsed.my_hand is not None
    assert parsed.opp_hand_size is not None
    assert parsed.opp_draw_pile_size is not None
    assert parsed.opp_draw_pile_size >= 0


def test_bureaucrat_reveals_hand_fallback_is_a_no_op():
    # "reveals their hand: ..." is informational; nothing moves.
    parsed = parse_dominion_log(BUREAUCRAT_LOG, my_name="domibot_v1.4", kingdom=BUREAUCRAT_KINGDOM)
    assert parsed.my_phase == "ACTION"
    assert parsed.opp_hand_size == 5


def test_bureaucrat_own_gain_is_not_double_counted_via_discard():
    # Bureaucrat's Silver goes to the deck top, so it isn't in the discard.
    parsed = parse_dominion_log(BUREAUCRAT_LOG, my_name="domibot_v1.4", kingdom=BUREAUCRAT_KINGDOM)
    tracked = Counter(parsed.my_hand) + Counter(parsed.my_discard) + Counter(parsed.my_play_area)
    my_total = Counter(parsed.my_total)
    for card, count in tracked.items():
        assert count <= my_total[card], f"{card}: tracked {count} exceeds my_total {my_total[card]}"


# The opponent's Artisan topdecks a card from their hidden hand, unnamed.
ARTISAN_TOPDECK_KINGDOM = ["Throne Room", "Council Room", "Laboratory", "Market", "Artisan",
                           "Cellar", "Moat", "Moneylender", "Poacher", "Remodel"]
ARTISAN_TOPDECK_LOG = """Game #183301566, rated.
L starts with 7 Coppers.
L starts with 3 Estates.
d starts with 7 Coppers.
d starts with 3 Estates.
L shuffles their deck.
d shuffles their deck.
L draws 5 cards.
d draws 4 Coppers and an Estate.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains an Artisan.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Silver.
d draws 4 Coppers and an Estate.
Turn 2 - Lord Rattington
L plays an Artisan.
L gains a Laboratory.
L topdecks a card."""


def test_artisan_unnamed_opponent_topdeck_does_not_crash_the_replay():
    # Stops mid-effect (no cleanup draw yet) so the interesting count isn't
    # washed out by a subsequent full-hand reset.
    parsed = parse_dominion_log(ARTISAN_TOPDECK_LOG, my_name="domibot_v1.4", kingdom=ARTISAN_TOPDECK_KINGDOM)
    assert parsed.my_phase == "ACTION"  # unaffected by the opponent's still-unresolved turn
    # 5 - Artisan played + its gain to hand - the topdeck.
    assert parsed.opp_hand_size == 4


# The opponent's Cellar: "discards 3 other cards and a Copper."
CELLAR_ANONYMOUS_DISCARD_KINGDOM = ARTISAN_TOPDECK_KINGDOM
CELLAR_ANONYMOUS_DISCARD_LOG = """Game #183301566, rated.
L starts with 7 Coppers.
L starts with 3 Estates.
d starts with 7 Coppers.
d starts with 3 Estates.
L shuffles their deck.
d shuffles their deck.
L draws 5 cards.
d draws 4 Coppers and an Estate.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Cellar.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Silver.
d draws 4 Coppers and an Estate.
Turn 2 - Lord Rattington
L plays a Cellar.
L gets +1 Action.
L discards 3 other cards and a Copper."""


def test_cellar_anonymous_discard_does_not_crash_the_replay():
    # Stops right after the discard (no cleanup shuffle+draw yet) so the
    # interesting counts aren't washed out by a subsequent full-hand reset.
    parsed = parse_dominion_log(CELLAR_ANONYMOUS_DISCARD_LOG, my_name="domibot_v1.4",
                                 kingdom=CELLAR_ANONYMOUS_DISCARD_KINGDOM)
    assert parsed.my_phase == "ACTION"
    # Turn 1's cleanup put their Coppers and Cellar there; this turn adds the named Copper.
    assert Counter(parsed.opp_discard) == Counter(["Copper"] * 5 + ["Cellar"])
    # 5 - Cellar - the named Copper - 3 unnamed cards.
    assert parsed.opp_hand_size == 0


# Militia's discard: "discards a card and a Copper." (no "other" for one card)
MILITIA_ANONYMOUS_DISCARD_KINGDOM = ["Militia", "Village", "Moat", "Smithy", "Workshop",
                                     "Chapel", "Bandit", "Council Room", "Festival", "Library"]
MILITIA_ANONYMOUS_DISCARD_LOG = """Game #1, unrated.
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 4 Coppers and an Estate.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Militia.
d draws 2 Coppers, 2 Estates, and a Militia.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Silver.
L draws 5 cards.
Turn 2 - domibot_v1.4
d plays a Militia.
d gets +$2.
L discards a card and a Copper."""


def test_militia_singular_anonymous_discard_does_not_crash_the_replay():
    parsed = parse_dominion_log(MILITIA_ANONYMOUS_DISCARD_LOG, my_name="domibot_v1.4",
                                 kingdom=MILITIA_ANONYMOUS_DISCARD_KINGDOM)
    assert parsed.my_phase == "ACTION"
    assert "Copper" in parsed.opp_discard
    # 5-card hand, minus 1 unnamed and 1 named Copper discarded.
    assert parsed.opp_hand_size == 3


# Throne Room on Bureaucrat: both Silvers go to the deck top across the "again" replay.
THRONE_ROOM_BUREAUCRAT_KINGDOM = ["Throne Room", "Bureaucrat", "Village", "Smithy",
                                  "Market", "Militia", "Witch", "Laboratory", "Festival", "Council Room"]
THRONE_ROOM_BUREAUCRAT_LOG = """Game #1, unrated.
L starts with 7 Coppers.
L starts with 3 Estates.
d starts with 7 Coppers.
d starts with 3 Estates.
L shuffles their deck.
d shuffles their deck.
L draws 5 cards.
d draws 3 Coppers and 2 Estates.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Throne Room.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Bureaucrat.
d shuffles their deck.
d draws 3 Coppers and 2 Estates.
Turn 2 - Lord Rattington
L plays a Throne Room.
L plays a Bureaucrat.
L gains a Silver.
d topdecks an Estate.
L plays a Bureaucrat again.
L gains a Silver.
d topdecks an Estate."""


def test_throne_room_replays_bureaucrat_topdecking_twice():
    parsed = parse_dominion_log(THRONE_ROOM_BUREAUCRAT_LOG, my_name="domibot_v1.4",
                                 kingdom=THRONE_ROOM_BUREAUCRAT_KINGDOM)
    # Both Estates topdecked (untracked -- deck contents are always derived
    # by elimination), neither one incorrectly landing in discard.
    assert parsed.my_hand == ["Copper", "Copper", "Copper"]
    assert parsed.my_discard == []


# --- pending_reaction_path: the opponent's Militia played, my forced
# discard not yet shown -- the relay tool's core new capability. ---

_PENDING_REACTION_KINGDOM = ["Militia", "Village", "Moat", "Smithy", "Workshop",
                             "Chapel", "Council Room", "Throne Room", "Festival", "Library"]


def _pending_reaction_log(opponent_turn_body: str) -> str:
    """A minimal, otherwise-boring game up through 'Turn 2 - Lord
    Rattington', with `opponent_turn_body` (already-indented log lines, no
    trailing newline) appended as the rest of that still-open turn."""
    return f"""Game #1, unrated.
d starts with 7 Coppers.
d starts with 3 Estates.
L starts with 7 Coppers.
L starts with 3 Estates.
d shuffles their deck.
L shuffles their deck.
d draws 3 Coppers and 2 Estates.
L draws 5 cards.
Turn 1 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d draws 3 Coppers and 2 Estates.
Turn 1 - Lord Rattington
L plays 4 Coppers. (+$4)
L buys and gains a Village.
L draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d draws 3 Coppers and 2 Estates.
Turn 2 - Lord Rattington
{opponent_turn_body}"""


def test_pending_reaction_detected_when_militia_discard_not_yet_logged():
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Militia.\nL gets +$2."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path == [Action("PLAY", "Militia")]
    # Turn 1's cleanup put their Coppers and Village there.
    assert Counter(parsed.pending_reaction_opp_discard) == Counter(["Copper"] * 4 + ["Village"])


def test_pending_reaction_includes_an_earlier_safe_play():
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Village.\nL draws a card.\nL gets +1 Action.\n"
                               "L plays a Militia.\nL gets +$2."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path == [Action("PLAY", "Village"), Action("PLAY", "Militia")]


def test_pending_reaction_none_when_i_play_militia_myself():
    # MILITIA_ANONYMOUS_DISCARD_LOG has *domibot_v1.4* play Militia, not the
    # opponent -- there's nothing pending on the opponent to recommend.
    parsed = parse_dominion_log(MILITIA_ANONYMOUS_DISCARD_LOG, my_name="domibot_v1.4",
                                 kingdom=MILITIA_ANONYMOUS_DISCARD_KINGDOM)
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_once_my_discard_is_already_logged():
    # My discard is already in the log: nothing is pending.
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Militia.\nL gets +$2.\nd discards a Copper and an Estate."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_once_opponent_moves_on_past_the_resolved_discard():
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Militia.\nL gets +$2.\nd discards a Copper and an Estate.\n"
                               "L plays 3 Coppers. (+$3)\nL buys and gains a Silver."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_when_i_block_with_moat():
    # Unverified against a real log -- inferred from _REVEALS_LINE's
    # general "X reveals Y." pattern, not observed dominion.games phrasing.
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Militia.\nL gets +$2.\nd reveals a Moat."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_after_throne_room_replayed_militia():
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Throne Room.\nL plays a Militia.\nL gets +$2.\n"
                               "d discards a Copper and an Estate.\nL plays a Militia again.\nL gets +$2."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_after_opponent_plays_chapel_then_militia():
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Chapel.\nL trashes 2 Coppers.\nL plays a Militia.\nL gets +$2."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None


def test_pending_reaction_none_after_opponent_plays_council_room_then_militia():
    # Council Room draws me a card, so a turn with it can't be replayed from its start.
    parsed = parse_dominion_log(
        _pending_reaction_log("L plays a Council Room.\nL gets +1 Buy.\nd draws a card.\n"
                               "L plays a Militia.\nL gets +$2."),
        my_name="domibot_v1.4", kingdom=_PENDING_REACTION_KINGDOM,
    )
    assert parsed.pending_reaction_path is None
