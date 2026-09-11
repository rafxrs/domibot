import pytest

from domibot import Action, Game, KINGDOM_CARDS
from domibot.enums import DecisionKind, Phase
from training.mcts import materialize, run_mcts, select_action
from training.relay import (
    CARD_ABBREVIATIONS,
    TableState,
    reconstruct_game,
    reconstruct_opponent_turn_boundary,
    resolve_card_name,
)


def _tiny_kingdom() -> list[str]:
    return list(KINGDOM_CARDS)[:10]


def _fresh_supply() -> dict[str, int]:
    """The real starting supply for a fresh 2-player game -- read off an
    actual Game rather than hand-derived, so the test can't drift from
    whatever setup rules game.py actually implements."""
    return dict(Game(_tiny_kingdom(), num_players=2, seed=0).supply)


def _turn_one_state(**overrides) -> TableState:
    """Both players still on their opening hand, nothing bought yet."""
    defaults = dict(
        kingdom=_tiny_kingdom(),
        supply=_fresh_supply(),
        trash=[],
        my_hand=["Copper", "Copper", "Copper", "Estate", "Estate"],
        my_discard=[],
        my_total=["Copper"] * 7 + ["Estate"] * 3,
        my_actions=1,
        my_buys=1,
        my_coins=0,
        my_phase="ACTION",
        my_turns_taken=0,
        opp_discard=[],
        opp_hand_size=5,
        opp_draw_pile_size=5,
    )
    defaults.update(overrides)
    return TableState(**defaults)


def test_reconstruct_game_matches_reported_state():
    state = _turn_one_state()
    game = reconstruct_game(state, seed=1)

    me = game.players[0]
    assert sorted(me.hand) == sorted(state.my_hand)
    assert me.actions == 1 and me.buys == 1 and me.coins == 0
    assert game.phase == Phase.ACTION
    assert game.current_player == 0
    assert game.pending_decision is None

    # my deck should be exactly what's left of my_total after hand/discard/play_area
    assert sorted(me.deck + me.hand + me.discard + me.play_area) == sorted(state.my_total)


def test_reconstruct_game_gives_opponent_the_reported_hand_and_deck_sizes():
    state = _turn_one_state()
    game = reconstruct_game(state, seed=2)
    opp = game.players[1]
    assert len(opp.hand) == state.opp_hand_size
    assert len(opp.deck) == state.opp_draw_pile_size
    # a fresh game: opponent's total ownership must also be exactly 7 Copper + 3 Estate
    assert sorted(opp.hand + opp.deck + opp.discard + opp.play_area) == sorted(["Copper"] * 7 + ["Estate"] * 3)


def test_reconstruct_game_legal_actions_reflect_reported_hand_and_phase():
    state = _turn_one_state(my_phase="BUY", my_coins=5, my_actions=0)
    game = reconstruct_game(state, seed=3)
    legal = game.legal_actions()
    from domibot.models import Action
    assert Action("BUY", "Estate") in legal  # cost 2, affordable
    assert all(a.verb in ("BUY", "END_BUY") for a in legal)


def test_reconstruct_game_rejects_my_total_missing_hand_cards():
    # my_hand has a Gold that my_total (still just the starting 7C/3E) doesn't account for at all
    state = _turn_one_state(my_hand=["Copper", "Copper", "Copper", "Estate", "Gold"])
    with pytest.raises(ValueError, match="my_total"):
        reconstruct_game(state, seed=4)


def test_reconstruct_game_rejects_inconsistent_opponent_sizes():
    state = _turn_one_state(opp_hand_size=4, opp_draw_pile_size=5)  # should be 5 and 5
    with pytest.raises(ValueError, match="opp_hand_size"):
        reconstruct_game(state, seed=5)


def test_reconstruct_game_rejects_overclaimed_opponent_discard():
    # opponent can't have discarded a Gold on turn 1 -- their total ownership
    # (by elimination) is only 7 Copper + 3 Estate at this point.
    state = _turn_one_state(opp_discard=["Gold"])
    with pytest.raises(ValueError):
        reconstruct_game(state, seed=6)


def test_domibot_agent_can_act_on_a_reconstructed_game():
    from training.agents import DomibotAgent
    from training.network import DomibotNet

    net = DomibotNet()
    net.eval()
    agent = DomibotAgent(net, num_simulations=5)

    state = _turn_one_state(my_phase="BUY", my_coins=5, my_actions=0)
    game = reconstruct_game(state, seed=7)
    action = agent.act(game)
    assert action in game.legal_actions()


def test_card_abbreviations_cover_every_kingdom_card_exactly_once():
    covered = sorted(CARD_ABBREVIATIONS.values())
    assert covered == sorted(KINGDOM_CARDS)


def test_card_abbreviations_are_all_distinct():
    codes = list(CARD_ABBREVIATIONS.keys())
    assert len(codes) == len(set(codes))


def test_resolve_card_name_expands_abbreviations_case_insensitively():
    assert resolve_card_name("POA") == "Poacher"
    assert resolve_card_name("poa") == "Poacher"
    assert resolve_card_name("Poa") == "Poacher"
    assert resolve_card_name("CR") == "Council Room"


def test_resolve_card_name_passes_through_full_names_and_is_case_insensitive():
    assert resolve_card_name("Copper") == "Copper"
    assert resolve_card_name("copper") == "Copper"
    assert resolve_card_name("Council Room") == "Council Room"


def test_resolve_card_name_rejects_unknown_tokens():
    with pytest.raises(ValueError, match="not a recognized card"):
        resolve_card_name("XYZ")


# --- reconstruct_opponent_turn_boundary ---

def _militia_kingdom() -> list[str]:
    return ["Militia", "Moat", "Village", "Smithy", "Workshop",
            "Chapel", "Bandit", "Council Room", "Festival", "Library"]


def _militia_supply(militias_bought: int = 0, moats_bought: int = 0) -> dict[str, int]:
    supply = dict(Game(_militia_kingdom(), num_players=2, seed=0).supply)
    supply["Militia"] -= militias_bought
    supply["Moat"] -= moats_bought
    return supply


def _militia_state(**overrides) -> TableState:
    """A turn-one state where the opponent owns exactly one Militia (one
    fewer in the supply than fresh, attributed to them by elimination).
    opp_hand_size/opp_draw_pile_size here only need to sum to the
    opponent's true total (11); reconstruct_opponent_turn_boundary ignores
    their split and always reconstructs a true 5-card turn-start hand, then
    forces every card named in `path` into it -- so tests below pass Militia's
    play as `path` rather than relying on a lucky seed."""
    defaults = dict(
        kingdom=_militia_kingdom(), supply=_militia_supply(militias_bought=1),
        opp_hand_size=11, opp_draw_pile_size=0,
    )
    defaults.update(overrides)
    return _turn_one_state(**defaults)


def test_reconstruct_opponent_turn_boundary_positions_opponent_to_act():
    state = _turn_one_state(kingdom=_militia_kingdom(), supply=_militia_supply())
    game = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], seed=1)

    assert game.current_player == 1
    assert game.phase == Phase.ACTION
    assert game.pending_decision is None
    assert len(game.players[1].hand) == 5
    assert game.players[1].play_area == []


def test_militia_forced_discard_materializes_for_me():
    state = _militia_state(my_hand=["Copper"] * 4 + ["Estate"])
    path = [Action("PLAY", "Militia")]
    boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], path=path, seed=2)

    game = materialize(boundary, path)

    assert game.pending_decision is not None
    assert game.pending_decision.kind == DecisionKind.SELECT_CARD
    assert game.pending_decision.player == 0


def test_militia_forced_discard_forces_a_known_play_into_the_opponents_hand():
    # regression: reconstruct_opponent_turn_boundary must not leave it to
    # chance whether a card the log says the opponent just played actually
    # lands in their randomly-reconstructed hand -- try every seed in a
    # wide range; without the forcing fix, some of them would produce an
    # "illegal action" in materialize().
    state = _militia_state(my_hand=["Copper"] * 4 + ["Estate"])
    path = [Action("PLAY", "Militia")]
    for seed in range(20):
        boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], path=path, seed=seed)
        game = materialize(boundary, path)
        assert game.pending_decision is not None


def test_militia_is_a_no_op_when_my_hand_has_three_or_fewer_cards():
    state = _militia_state(my_hand=["Copper", "Copper", "Estate"])
    path = [Action("PLAY", "Militia")]
    boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], path=path, seed=0)

    game = materialize(boundary, path)

    assert game.pending_decision is None


def test_militia_offers_moat_reveal_when_i_hold_a_moat():
    state = _militia_state(
        my_hand=["Moat", "Copper", "Copper", "Estate", "Estate"],
        my_total=["Copper"] * 7 + ["Estate"] * 3 + ["Moat"],
        supply=_militia_supply(militias_bought=1, moats_bought=1),
    )
    path = [Action("PLAY", "Militia")]
    boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], path=path, seed=0)

    game = materialize(boundary, path)

    assert game.pending_decision is not None
    assert game.pending_decision.kind == DecisionKind.REACT
    assert game.pending_decision.player == 0


def test_domibot_agent_can_search_from_an_opponent_turn_boundary():
    from training.network import DomibotNet

    net = DomibotNet()
    net.eval()

    state = _militia_state()
    path = [Action("PLAY", "Militia")]
    boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard=[], path=path, seed=0)

    root = run_mcts(boundary, net, num_simulations=5, path=path)
    action = select_action(root, temperature=0.0)

    assert action in materialize(boundary, path).legal_actions()
