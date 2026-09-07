import pytest

from examples.domibot_relay import format_cards, parse_cards, parse_kingdom, prompt_multiline


def test_parse_kingdom_accepts_space_separated_codes_no_commas():
    kingdom = parse_kingdom("POA CR CHA MIL MIN MOA MLR VAS VIL TR")
    assert kingdom == [
        "Poacher", "Council Room", "Chapel", "Militia", "Mine",
        "Moat", "Moneylender", "Vassal", "Village", "Throne Room",
    ]


def test_parse_kingdom_still_tolerates_commas():
    assert parse_kingdom("POA, CR, CHA") == ["Poacher", "Council Room", "Chapel"]


def test_parse_kingdom_rejects_unknown_code():
    with pytest.raises(ValueError, match="not a recognized card"):
        parse_kingdom("POA XYZ CHA")


def test_parse_cards_accepts_abbreviation_with_count_suffix():
    assert parse_cards("BANx2, marx1") == ["Bandit", "Bandit", "Market"]


def test_parse_cards_empty_string_is_empty_zone():
    assert parse_cards("") == []


def test_format_cards_round_trips_through_parse_cards():
    cards = ["Copper", "Copper", "Copper", "Estate"]
    assert parse_cards(format_cards(cards)) == cards


def test_parse_cards_accepts_leading_count_space_separated():
    assert parse_cards("3 copper 2 estate") == ["Copper", "Copper", "Copper", "Estate", "Estate"]


def test_parse_cards_accepts_leading_count_with_trailing_x():
    assert parse_cards("3x copper") == ["Copper", "Copper", "Copper"]


def test_parse_cards_accepts_bare_name_no_count():
    assert parse_cards("copper") == ["Copper"]


def test_parse_cards_recovers_two_word_full_names_with_leading_count():
    assert parse_cards("2 Council Room") == ["Council Room", "Council Room"]


def test_parse_cards_mixes_leading_and_trailing_counts_freely():
    assert parse_cards("POAx2 CR 3 copper") == ["Poacher", "Poacher", "Council Room", "Copper", "Copper", "Copper"]


def test_parse_cards_rejects_unrecognized_token():
    with pytest.raises(ValueError, match="not a recognized card"):
        parse_cards("3 xyz")


def test_recommend_auto_skips_action_phase_with_no_playable_cards(capsys):
    import torch

    from domibot import Game
    from examples.domibot_relay import recommend
    from training.network import DomibotNet
    from training.relay import TableState

    kingdom = ["Bandit", "Festival", "Library", "Artisan", "Vassal",
               "Bureaucrat", "Moneylender", "Remodel", "Cellar", "Harbinger"]
    supply = dict(Game(kingdom, num_players=2, seed=0).supply)
    state = TableState(
        kingdom=kingdom, supply=supply, trash=[],
        my_hand=["Copper", "Copper", "Estate", "Estate", "Estate"],
        my_discard=[], my_play_area=[],
        my_total=["Copper"] * 7 + ["Estate"] * 3,
        my_phase="ACTION", my_actions=1, my_buys=1, my_coins=0, my_turns_taken=0,
        opp_discard=[], opp_play_area=[], opp_hand_size=5, opp_draw_pile_size=5,
    )
    net = DomibotNet()
    net.eval()
    recommend(state, net, simulations=10, device=torch.device("cpu"))

    out = capsys.readouterr().out
    assert "auto-ending your action phase" in out
    assert "recommended: END_ACTIONS" not in out


def test_prompt_multiline_ignores_spurious_leading_blank_line(monkeypatch):
    # Pasting a large block into a Windows console commonly delivers a
    # spurious empty first line before the real content (no bracketed-
    # paste support) -- without skipping it, the old "blank line ends the
    # paste" logic terminated immediately with nothing captured, and every
    # subsequent already-buffered log line got fed one-at-a-time into
    # whatever prompt came next instead.
    fed = iter(["", "Turn 1 - domibot_v1.4", "d plays 3 Coppers. (+$3)", ""])
    monkeypatch.setattr("builtins.input", lambda: next(fed))
    result = prompt_multiline("Paste something")
    assert result == "Turn 1 - domibot_v1.4\nd plays 3 Coppers. (+$3)"


def test_prompt_multiline_still_ends_on_first_real_blank_line(monkeypatch):
    fed = iter(["line one", "line two", "", "should never be consumed"])
    monkeypatch.setattr("builtins.input", lambda: next(fed))
    result = prompt_multiline("Paste something")
    assert result == "line one\nline two"
