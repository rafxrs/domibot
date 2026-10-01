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
from training.relay import (TableState, reconstruct_game, reconstruct_opponent_turn_boundary, replay_open_play,
                            table_state)

ME = "domibot_v1.4"

# A Vassal playing a Chapel from the discard (turn 10), and Throne Room + Vassal chains.
USER_VASSAL_LOG, USER_VASSAL_KINGDOM = T.load_log("vassal_plays_from_discard.txt")


def _state(parsed, kingdom) -> TableState:
    return table_state(parsed, kingdom)


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


_REAL_LOGS = [(path.stem, *T.load_log(path.name)) for path in sorted(T.FIXTURES.glob("*.txt"))]


@pytest.mark.parametrize("name,log,kingdom", _REAL_LOGS, ids=[c[0] for c in _REAL_LOGS])
def test_every_turn_start_of_mine_parses_and_reconstructs(name, log, kingdom):
    for header, prefix in _prefixes(log, lambda line: line.startswith("Turn ") and line.endswith(ME)):
        parsed = parse_dominion_log(prefix, my_name=ME, kingdom=kingdom)
        assert parsed.my_hand is not None, f"{name}: not fully derived at {header!r}"
        assert parsed.opp_hand_size == 5, f"{name}: opponent hand size at {header!r}"
        reconstruct_game(_state(parsed, kingdom), seed=0)


@pytest.mark.parametrize("name,log,kingdom", _REAL_LOGS, ids=[c[0] for c in _REAL_LOGS])
def test_every_line_of_every_real_log_parses_consistently(name, log, kingdom):
    # A paste ending at any line must still add up.
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
    # Sentry's look, trash and topdeck are all off the deck: 5 - Cellar - Sentry + its draw.
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
    # My Bureaucrat's deck-top routing must not apply to the opponent's Curse.
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


# --- Moat reactions ("G reacts with a Moat.") and reshuffles mid-card, from
# the real game in fixtures/dominion_logs/moat_reaction_sentry_reshuffles.txt ---

MOAT_LOG, MOAT_KINGDOM = T.load_log("moat_reaction_sentry_reshuffles.txt")


def _cut_after(log: str, line: str) -> str:
    lines = log.splitlines()
    return "\n".join(lines[: lines.index(line) + 1])


def test_opponent_moat_reaction_is_known_to_be_in_their_hand():
    parsed = _parse(_cut_after(MOAT_LOG, "d draws 2 Poachers."), MOAT_KINGDOM)
    assert parsed.my_hand is not None
    assert parsed.opp_known_hand == ["Moat"]
    assert parsed.my_actions == 0 and parsed.my_play_area == ["Sentry", "Sentry", "Witch"]
    game = reconstruct_game(_state(parsed, MOAT_KINGDOM), seed=0)
    assert "Moat" in game.players[1].hand
    # My Witch resolved (they blocked it): a plain phase decision is next.
    assert replay_open_play(parsed.open_play, MOAT_KINGDOM, parsed.my_hand, seed=0).status == "resolved"


def test_my_own_moat_reaction_resolves_the_attack():
    log = _opponent_turn("f plays a Witch.\nd reacts with a Moat.", buy1="a Moat",
                         draw3="a Moat, 3 Coppers, and an Estate", opp_buy1="a Witch")
    parsed = _parse(log)
    assert parsed.pending_reaction_path is None
    assert "Moat" in parsed.my_hand


def test_sentry_whose_draw_reshuffles_my_deck_is_replayed():
    # Sentry's draw and looks come from the reshuffled discard; the replay must reveal those.
    log = _cut_after(MOAT_LOG, "d looks at an Estate and a Sentry.")
    result, decision = _open(log, MOAT_KINGDOM)
    assert result.status == "pending"
    assert set(decision.options) == {Action("TRASH", "Estate"), Action("TRASH", "Sentry"), Action("DONE")}

