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
    from training.relay import TableState, reconstruct_game

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
    recommend(reconstruct_game(state), net, simulations=10, device=torch.device("cpu"))

    out = capsys.readouterr().out
    assert "auto-ending your action phase" in out
    assert "recommended: END_ACTIONS" not in out


def test_a_choice_of_nothing_also_recommends_the_next_move(capsys, monkeypatch):
    # dominion.games logs no line for Chapel trashing nothing, so the move after it is shown too.
    import torch

    import examples.domibot_relay as cli
    from domibot import Action, Game
    from domibot.models import DONE
    from training.network import DomibotNet
    from training.relay import TableState, reconstruct_game

    kingdom = ["Chapel", "Festival", "Library", "Artisan", "Vassal", "Bureaucrat", "Moneylender", "Remodel",
               "Cellar", "Harbinger"]
    supply = {**Game(kingdom, num_players=2, seed=0).supply, "Chapel": 9}
    state = TableState(kingdom=kingdom, supply=supply, trash=[], my_discard=[],
                       my_hand=["Chapel", "Copper", "Copper", "Copper", "Estate"],
                       my_total=["Copper"] * 7 + ["Estate"] * 3 + ["Chapel"])
    boundary, net = reconstruct_game(state, seed=0), DomibotNet().eval()
    for pick, moves_shown in ((DONE, 2), (Action("TRASH", "Estate"), 1)):
        monkeypatch.setattr(cli, "select_action", lambda root, temperature, pick=pick:
                            pick if pick in root.legal_actions else root.legal_actions[-1])
        cli.recommend_choice(boundary, [Action("PLAY", "Chapel")], net, simulations=4, device=torch.device("cpu"))
        assert capsys.readouterr().out.count("==> recommended:") == moves_shown


def test_a_log_gaining_a_card_outside_the_kingdom_asks_for_the_kingdom_again(monkeypatch):
    import test_log_parser as T
    from examples.domibot_relay import try_parse_log

    monkeypatch.setattr("builtins.input", lambda _prompt: " ".join(T.KINGDOM))
    parsed, kingdom = try_parse_log(T.REAL_LOG, ["Market"] + T.KINGDOM[1:], "domibot_v1.4")
    assert kingdom == T.KINGDOM and parsed.my_hand is not None


def test_prompt_multiline_ignores_spurious_leading_blank_line(monkeypatch):
    # A Windows console paste can start with a spurious blank line.
    fed = iter(["", "Turn 1 - domibot_v1.4", "d plays 3 Coppers. (+$3)", ""])
    monkeypatch.setattr("builtins.input", lambda: next(fed))
    result = prompt_multiline("Paste something")
    assert result == "Turn 1 - domibot_v1.4\nd plays 3 Coppers. (+$3)"


def test_prompt_multiline_still_ends_on_first_real_blank_line(monkeypatch):
    fed = iter(["line one", "line two", "", "should never be consumed"])
    monkeypatch.setattr("builtins.input", lambda: next(fed))
    result = prompt_multiline("Paste something")
    assert result == "line one\nline two"


def test_recommend_decides_a_penultimate_province_by_playouts(capsys, monkeypatch):
    import torch

    import examples.domibot_relay as cli
    import training.agents as agents
    from test_search_agent import _buy_phase
    from training.network import DomibotNet

    monkeypatch.setattr(agents, "PLAYOUT_MOVES", 20)
    monkeypatch.setattr(cli, "PROVINCE_PLAYOUTS", 2)
    cli.recommend(_buy_phase(2, opp_extra=["Estate"]), DomibotNet().eval(), simulations=4, device=torch.device("cpu"))
    assert "playouts" in capsys.readouterr().out
