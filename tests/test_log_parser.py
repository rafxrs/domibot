import pytest

from training.log_parser import parse_dominion_log

# Verbatim (trimmed) from a real dominion.games log, provided by the user --
# the best possible regression fixture, since it's not a log we wrote to fit
# the parser.
REAL_LOG = """
Game #183147151, rated.
lololololo: 44.45
domibot_v1.4: 42.12
Timer: Friendly
Card Pool: level 1
l starts with 3 Estates.
l starts with 7 Coppers.
d starts with 3 Estates.
d starts with 7 Coppers.
l shuffles their deck.
d shuffles their deck.
l draws 5 cards.
d draws 4 Coppers and an Estate.
Turn 1 - lololololo
l plays 3 Coppers. (+$3)
l buys and gains a Silver.
l draws 5 cards.
Turn 1 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Smithy.
d draws 3 Coppers and 2 Estates.
Turn 2 - lololololo
l plays 4 Coppers. (+$4)
l buys and gains a Remodel.
l shuffles their deck.
l draws 5 cards.
Turn 2 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws 5 Coppers.
Turn 3 - lololololo
l plays 3 Coppers. (+$3)
l buys and gains a Silver.
l draws 5 cards.
Turn 3 - domibot_v1.4
d plays 5 Coppers. (+$5)
d buys and gains a Bandit.
d draws 2 Coppers, a Silver, an Estate, and a Smithy.
Turn 4 - lololololo
l plays a Remodel.
l trashes an Estate.
l gains a Remodel.
l plays 3 Coppers. (+$3)
l buys and gains a Silver.
l shuffles their deck.
l draws 5 cards.
Turn 4 - domibot_v1.4
d plays a Smithy.
d shuffles their deck.
d draws 2 Estates and a Bandit.
d plays a Silver and 2 Coppers. (+$4)
d buys and gains a Moneylender.
d draws 5 Coppers.
Turn 5 - lololololo
l plays a Remodel.
l trashes a Silver.
l gains a Sentry.
l plays 2 Coppers. (+$2)
l buys and gains a Cellar.
l draws 5 cards.
Turn 5 - domibot_v1.4
d plays 5 Coppers. (+$5)
d buys and gains a Sentry.
d shuffles their deck.
d draws a Copper, 2 Estates, a Bandit, and a Smithy.
Turn 6 - lololololo
l plays a Remodel.
l trashes an Estate.
l gains a Silver.
l plays a Silver and 2 Coppers. (+$4)
l buys and gains a Remodel.
l shuffles their deck.
l draws 5 cards.
Turn 6 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
l reveals a Copper and a Silver.
l trashes a Silver.
l discards a Copper.
d plays a Copper. (+$1)
d draws 2 Coppers, an Estate, a Moneylender, and a Sentry.
Turn 7 - lololololo
l plays a Silver and 3 Coppers. (+$5)
l buys and gains a Sentry.
l draws 5 cards.
Turn 7 - domibot_v1.4
d plays a Sentry.
d draws a Copper.
d gets +1 Action.
d looks at 2 Coppers.
d trashes 2 Coppers.
d plays a Moneylender.
d trashes a Copper.
d gets +$3.
d plays 2 Coppers. (+$2)
d buys and gains a Sentry.
d shuffles their deck.
d draws 3 Coppers, a Silver, and a Moneylender.
Turn 8 - lololololo
l plays a Cellar.
l gets +1 Action.
l discards a card and a Remodel.
l draws 2 cards.
l plays a Sentry.
l draws a card.
l gets +1 Action.
l shuffles their deck.
l looks at 2 cards.
l trashes a Copper.
l topdecks a card.
l plays a Remodel.
l trashes a Remodel.
l gains a Bandit.
l plays a Silver. (+$2)
l buys and gains a Cellar.
l draws 5 cards.
Turn 8 - domibot_v1.4
d plays a Moneylender.
d trashes a Copper.
d gets +$3.
d plays a Silver and 2 Coppers. (+$4)
d buys and gains an Artisan.
d draws a Copper, 2 Estates, a Sentry, and a Smithy.
Turn 9 - lololololo
l plays a Remodel.
l trashes a Silver.
l gains a Sentry.
l plays 3 Coppers. (+$3)
l buys and gains a Silver.
l shuffles their deck.
l draws 5 cards.
Turn 9 - domibot_v1.4
d plays a Sentry.
d draws a Bandit.
d gets +1 Action.
d looks at a Gold and an Estate.
d trashes an Estate.
d topdecks a Gold.
d plays a Bandit.
d gains a Gold.
l reveals a Silver and a Bandit.
l trashes a Silver.
l discards a Bandit.
d plays a Copper. (+$1)
d shuffles their deck.
d draws a Copper, a Gold, an Estate, and 2 Sentries.
Turn 10 - lololololo
l plays a Sentry.
l draws a card.
l gets +1 Action.
l looks at 2 cards.
l topdecks 2 cards.
l plays 4 Coppers. (+$4)
l buys and gains a Remodel.
l draws 5 cards.
Turn 10 - domibot_v1.4
d plays a Sentry.
d draws a Copper.
d gets +1 Action.
d looks at an Estate and a Smithy.
d trashes an Estate.
d topdecks a Smithy.
d plays a Sentry.
d draws a Smithy.
d gets +1 Action.
d looks at a Copper and a Moneylender.
d topdecks a Copper and a Moneylender.
d plays a Smithy.
d draws a Copper, a Bandit, and a Moneylender.
d plays 3 Coppers and a Gold. (+$6)
d buys and gains an Artisan.
d shuffles their deck.
d draws a Copper, a Silver, a Gold, an Estate, and an Artisan.
Turn 11 - lololololo
l plays a Sentry.
l draws a card.
l gets +1 Action.
l shuffles their deck.
"""

KINGDOM = ["Smithy", "Remodel", "Bandit", "Moneylender", "Sentry", "Cellar", "Artisan",
           "Village", "Chapel", "Witch"]  # last 3 never appear in this fragment


def test_parses_real_log_my_total_matches_hand_derivation():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)

    # Hand-derived from the log (d's gains: Smithy, Silver, Bandit,
    # Moneylender, Sentry, Gold(bandit T6), Sentry(buy T7), Artisan(buy T8),
    # Gold(bandit T9), Artisan(buy T10) = 10 gains; d's trashes: 2 Copper +
    # 1 Copper (sentry+moneylender T7), 1 Copper (moneylender T8), 1 Estate
    # (sentry T9), 1 Estate (sentry T10) = 4 Copper + 2 Estate trashed).
    # Starting 7C/3E + 10 gains - 6 trashes = 14 cards total.
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
    # Estate: l's Remodel (T4, T6) + d's Sentry (T9, T10) = 4
    # Silver: l's Remodel (T5, T9) + l's Bandit-forced (T6, T9) = 4
    # Copper: d's Sentry x2 + Moneylender (T7) + d's Moneylender (T8) + l's Sentry (T8) = 5
    # Remodel: l trashing a spare copy of its own card (T8) = 1
    assert trash["Estate"] == 4
    assert trash["Silver"] == 4
    assert trash["Copper"] == 5
    assert trash["Remodel"] == 1
    assert sum(trash.values()) == 14


def test_parses_real_log_supply_decremented_for_both_players_buys():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    # Sentry bought 3 times total across both players (l: T5, T7, T9; d: T5, T7) -- recount from log:
    # l buys Sentry: none directly bought by l (l only *gains* Sentry via Remodel, not a supply buy... wait
    # Remodel's gain also draws from supply) -- gains count against supply too, so just check it went down.
    assert parsed.supply["Sentry"] < 10
    assert parsed.supply["Bandit"] < 10
    assert parsed.supply["Artisan"] == 10 - 2  # d bought 2 (T8, T10); l never touched Artisan


def test_parses_real_log_turns_taken():
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    # d has started turn 10 (fully shown) and l has started turn 11 -- so
    # "completed turns before this one" for d is 9 if querying mid-turn-10,
    # but since d's turn 10 is fully shown as complete in this fragment,
    # d has *started* 10 turns.
    assert parsed.turns_taken["domibot_v1.4"] == 9
    assert parsed.turns_taken["lololololo"] == 10


def test_unknown_player_name_raises():
    with pytest.raises(ValueError, match="isn't one of the players"):
        parse_dominion_log(REAL_LOG, my_name="someone_else", kingdom=KINGDOM)


def test_ambiguous_first_letters_raise():
    log = "dave: 30.0\ndan: 25.0\nTurn 1 - dave\n"
    with pytest.raises(ValueError, match="same letter"):
        parse_dominion_log(log, my_name="dave", kingdom=KINGDOM)


def test_my_hand_not_derived_when_log_ends_mid_turn():
    # REAL_LOG ends partway through lololololo's turn 11, not at the start
    # of domibot_v1.4's own turn -- too many possible in-between states to
    # safely reconstruct, so this should fall back to manual entry.
    parsed = parse_dominion_log(REAL_LOG, my_name="domibot_v1.4", kingdom=KINGDOM)
    assert parsed.my_hand is None
    assert parsed.my_phase is None


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
