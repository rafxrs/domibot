import pytest

from examples.domibot_relay import format_cards, parse_cards, parse_kingdom


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
