import pytest

from domibot import Game, KINGDOM_CARDS
from domibot.enums import Phase
from training.relay import TableState, reconstruct_game


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
