import pytest

from domibot.models import END_ACTIONS
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


# A second real, complete (23-turn) dominion.games log, provided by the
# user -- a mutual-Witch mirror match with heavy Throne Room/Village
# chaining on the opponent's side and several Bandit-forced reveals on
# ours. This caught two real bugs during development: the opponent's
# hand_size double-counting their opening draw, and -- more subtly -- a
# shuffle line logged *before* the cleanup draw it's actually part of,
# which (without special handling) let that turn's played cards wrongly
# survive into "known" discard instead of being swept away by the shuffle
# like everything else already there.
WITCH_MIRROR_LOG = """
Game #183186242, rated.
domibot_v1.4: 43.22
apoorvab: 40.69
Timer: Patient
Card Pool: level 1
d starts with 3 Estates.
d starts with 7 Coppers.
a starts with 3 Estates.
a starts with 7 Coppers.
d shuffles their deck.
a shuffles their deck.
d draws 3 Coppers and 2 Estates.
a draws 5 cards.
Turn 1 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d draws 4 Coppers and an Estate.
Turn 1 - apoorvab
a draws 5 cards.
Turn 2 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Moneylender.
d shuffles their deck.
d draws a Copper, a Silver, 2 Estates, and a Moneylender.
Turn 2 - apoorvab
a plays 5 Coppers. (+$5)
a buys and gains a Witch.
a shuffles their deck.
a draws 5 cards.
Turn 3 - domibot_v1.4
d plays a Moneylender.
d trashes a Copper.
d gets +$3.
d plays a Silver. (+$2)
d buys and gains a Witch.
d draws 4 Coppers and an Estate.
Turn 3 - apoorvab
a plays 4 Coppers. (+$4)
a buys and gains a Moneylender.
a draws 5 cards.
Turn 4 - domibot_v1.4
d plays 4 Coppers. (+$4)
d buys and gains a Throne Room.
d shuffles their deck.
d draws 2 Coppers, 2 Estates, and a Moneylender.
Turn 4 - apoorvab
a plays a Witch.
a shuffles their deck.
a draws 2 cards.
d gains a Curse.
a plays 3 Coppers. (+$3)
a buys and gains a Village.
a draws 5 cards.
Turn 5 - domibot_v1.4
d plays a Moneylender.
d trashes a Copper.
d gets +$3.
d plays a Copper. (+$1)
d buys and gains a Silver.
d draws 3 Coppers, a Silver, and a Witch.
Turn 5 - apoorvab
a plays a Moneylender.
a trashes a Copper.
a gets +$3.
a plays 3 Coppers. (+$3)
a buys and gains a Witch.
a shuffles their deck.
a draws 5 cards.
Turn 6 - domibot_v1.4
d plays a Witch.
d draws an Estate and a Throne Room.
a gains a Curse.
d plays a Silver and 3 Coppers. (+$5)
d buys and gains a Council Room.
d shuffles their deck.
d draws 2 Coppers, a Silver, an Estate, and a Council Room.
Turn 6 - apoorvab
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays 4 Coppers. (+$4)
a buys and gains a Throne Room.
a draws 5 cards.
Turn 7 - domibot_v1.4
d plays a Council Room.
d draws a Curse, a Silver, an Estate, and a Witch.
d gets +1 Buy.
a draws a card.
d plays 2 Silvers and 2 Coppers. (+$6)
d buys and gains a Gold.
d draws 3 Coppers, an Estate, and a Throne Room.
Turn 7 - apoorvab
a plays a Village.
a shuffles their deck.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays 4 Coppers. (+$4)
a buys and gains a Throne Room.
a shuffles their deck.
a draws 5 cards.
Turn 8 - domibot_v1.4
d plays 3 Coppers. (+$3)
d buys and gains a Silver.
d shuffles their deck.
d draws a Curse, 2 Silvers, an Estate, and a Moneylender.
Turn 8 - apoorvab
a plays 3 Coppers. (+$3)
a buys and gains a Village.
a draws 5 cards.
Turn 9 - domibot_v1.4
d plays a Moneylender.
d plays 2 Silvers. (+$4)
d buys and gains a Silver.
d draws a Curse, a Copper, a Gold, a Council Room, and a Witch.
Turn 9 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays a Moneylender.
a trashes a Copper.
a gets +$3.
a plays a Copper. (+$1)
a buys and gains a Throne Room.
a shuffles their deck.
a draws 5 cards.
Turn 10 - domibot_v1.4
d plays a Witch.
d draws a Curse and an Estate.
a gains a Curse.
d plays a Copper and a Gold. (+$4)
d buys and gains a Silver.
d draws 3 Coppers, a Silver, and an Estate.
Turn 10 - apoorvab
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays a Witch again.
a draws 2 cards.
d gains a Curse.
a plays 3 Coppers. (+$3)
a buys and gains a Village.
a draws 5 cards.
Turn 11 - domibot_v1.4
d plays a Silver and 3 Coppers. (+$5)
d buys and gains a Bandit.
d shuffles their deck.
d draws a Curse, 2 Coppers, a Silver, and a Throne Room.
Turn 11 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Village.
a draws a card.
a gets +2 Actions.
a shuffles their deck.
a draws 5 cards.
Turn 12 - domibot_v1.4
d plays a Silver and 2 Coppers. (+$4)
d buys and gains a Silver.
d draws 2 Curses, a Silver, a Gold, and a Council Room.
Turn 12 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a draws 5 cards.
Turn 13 - domibot_v1.4
d plays a Council Room.
d draws a Copper, a Silver, an Estate, and a Bandit.
d gets +1 Buy.
a draws a card.
d plays 2 Silvers, a Copper, and a Gold. (+$8)
d buys and gains a Province.
d draws a Curse, a Copper, a Silver, an Estate, and a Witch.
Turn 13 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
d gains a Curse.
a plays a Witch.
a shuffles their deck.
a draws 2 cards.
a plays a Moneylender.
a trashes a Copper.
a gets +$3.
a plays 3 Coppers. (+$3)
a buys and gains a Council Room.
a draws 5 cards.
Turn 14 - domibot_v1.4
d plays a Witch.
d draws a Curse and a Moneylender.
d plays a Silver and a Copper. (+$3)
d buys and gains a Silver.
d draws 2 Curses, a Copper, a Silver, and an Estate.
Turn 14 - apoorvab
a plays a Throne Room.
a plays a Village.
a shuffles their deck.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Council Room.
a draws 4 cards.
a gets +1 Buy.
d shuffles their deck.
d draws a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
a plays a Witch.
a draws 2 cards.
a plays a Throne Room.
a plays a Moneylender.
a trashes a Copper.
a gets +$3.
a plays a Moneylender again.
a trashes a Copper.
a gets +$3.
a plays 2 Coppers. (+$2)
a buys and gains a Bandit.
a buys and gains a Village.
a shuffles their deck.
a draws 5 cards.
Turn 15 - domibot_v1.4
d plays a Throne Room.
d plays a Silver and a Copper. (+$3)
d buys and gains a Silver.
d draws a Curse, a Copper, 2 Silvers, and a Council Room.
Turn 15 - apoorvab
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Throne Room.
a plays a Moneylender.
a trashes a Copper.
a gets +$3.
a plays a Moneylender again.
a trashes a Copper.
a gets +$3.
a buys and gains a Gold.
a draws 5 cards.
Turn 16 - domibot_v1.4
d plays a Council Room.
d draws a Curse, a Silver, an Estate, and a Moneylender.
d gets +1 Buy.
a draws a card.
d plays 3 Silvers and a Copper. (+$7)
d buys and gains a Gold.
d draws 2 Curses, 2 Silvers, and a Bandit.
Turn 16 - apoorvab
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a shuffles their deck.
a draws 2 cards.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Bandit.
a gains a Gold.
d reveals a Copper and a Silver.
d trashes a Silver.
d discards a Copper.
a plays a Bandit again.
a gains a Gold.
d reveals a Curse and a Copper.
d discards a Curse and a Copper.
a plays a Council Room.
a shuffles their deck.
a draws 2 cards.
a gets +1 Buy.
d draws a Gold.
a plays 3 Golds. (+$9)
a buys and gains a Province.
a shuffles their deck.
a draws 5 cards.
Turn 17 - domibot_v1.4
d plays a Bandit.
d gains a Gold.
a reveals a Bandit and a Throne Room.
a discards a Bandit and a Throne Room.
d plays 2 Silvers and a Gold. (+$7)
d buys and gains a Gold.
d draws a Curse, a Copper, an Estate, a Province, and a Witch.
Turn 17 - apoorvab
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Gold. (+$3)
a buys and gains a Village.
a draws 5 cards.
Turn 18 - domibot_v1.4
d plays a Witch.
d shuffles their deck.
d draws a Silver and a Bandit.
d plays a Silver and a Copper. (+$3)
d buys and gains a Silver.
d draws 2 Curses, a Copper, a Silver, and an Estate.
Turn 18 - apoorvab
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
a plays 2 Golds. (+$6)
a buys and gains a Gold.
a shuffles their deck.
a draws 5 cards.
Turn 19 - domibot_v1.4
d plays a Silver and a Copper. (+$3)
d buys and gains a Silver.
d draws a Copper, a Silver, 2 Golds, and a Council Room.
Turn 19 - apoorvab
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Council Room.
a draws 4 cards.
a gets +1 Buy.
d draws a Curse.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Bandit.
a gains a Gold.
d reveals a Curse and a Gold.
d trashes a Gold.
d discards a Curse.
a plays 4 Golds. (+$12)
a buys and gains a Province.
a buys and gains a Throne Room.
a shuffles their deck.
a draws 5 cards.
Turn 20 - domibot_v1.4
d plays a Council Room.
d draws 2 Curses, a Copper, and an Estate.
d gets +1 Buy.
a draws a card.
d plays a Silver, 2 Coppers, and 2 Golds. (+$10)
d buys and gains a Province.
d buys and gains an Estate.
d draws a Curse, a Copper, a Silver, a Moneylender, and a Throne Room.
Turn 20 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Witch.
a draws 2 cards.
a plays a Council Room.
a draws 4 cards.
a gets +1 Buy.
d draws a Silver.
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Bandit.
a gains a Gold.
d reveals a Silver and a Gold.
d trashes a Silver.
d discards a Gold.
a plays 4 Golds. (+$12)
a buys and gains a Province.
a buys and gains a Village.
a draws 5 cards.
Turn 21 - domibot_v1.4
d plays a Throne Room.
d plays a Moneylender.
d trashes a Copper.
d gets +$3.
d plays a Moneylender again.
d plays 2 Silvers. (+$4)
d buys and gains a Gold.
d shuffles their deck.
d draws 3 Silvers, a Gold, and an Estate.
Turn 21 - apoorvab
a plays a Throne Room.
a plays a Village.
a shuffles their deck.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Gold. (+$3)
a buys and gains a Village.
a draws 5 cards.
Turn 22 - domibot_v1.4
d plays 3 Silvers and a Gold. (+$9)
d buys and gains a Province.
d draws a Copper, a Silver, a Gold, a Bandit, and a Council Room.
Turn 22 - apoorvab
a plays a Throne Room.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village again.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Council Room.
a draws 4 cards.
a gets +1 Buy.
d draws a Gold.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Throne Room.
a plays a Witch.
a draws 2 cards.
a plays a Witch again.
a draws 2 cards.
a plays a Bandit.
a gains a Gold.
d reveals a Gold and a Province.
d trashes a Gold.
d discards a Province.
a plays a Village.
a draws a card.
a gets +2 Actions.
a plays a Village.
a shuffles their deck.
a draws a card.
a gets +2 Actions.
a plays 5 Golds. (+$15)
a buys and gains a Province.
a buys and gains a Gold.
a draws 5 cards.
Turn 23 - domibot_v1.4
d plays a Council Room.
d draws a Curse, a Copper, a Silver, and an Estate.
d gets +1 Buy.
a draws a card.
d plays 2 Silvers, 2 Coppers, and 2 Golds. (+$12)
d buys and gains an Estate.
d buys and gains a Province.
d draws 2 Curses, a Copper, a Silver, and a Province.
The game has ended.
"""

WITCH_MIRROR_KINGDOM = [
    "Council Room", "Library", "Mine", "Witch", "Workshop",
    "Moneylender", "Throne Room", "Bandit", "Vassal", "Village",
]


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
    # turn 7 hand was [Copper, Copper, Silver, Estate, Throne Room] (from
    # turn 6's cleanup draw), minus the played Council Room, plus its
    # +4 Cards draw [Curse, Silver, Estate, Witch].
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
    # d's hand at this point is turn 16's cleanup draw, untouched by the
    # opponent's Bandit (which reveals/trashes/discards from *deck top*,
    # never hand) -- the revealed Copper/Silver must not be removed from
    # the tracked hand, and the discarded Copper must land in discard.
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
    # A real dominion.games opponent name ("Lord Rattington") broke player
    # detection entirely: the rating-line regex didn't allow spaces in
    # names, so only the first player was ever found, silently disabling
    # the full replay (and even the baseline supply/trash layer) for every
    # game against a multi-word-named opponent.
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
