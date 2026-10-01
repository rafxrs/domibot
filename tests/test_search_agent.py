from collections import Counter

import torch

import training.agents as agents
from domibot import Game, KINGDOM_CARDS, Phase
from training.agents import PROVINCE, DeterminizedSearchAgent, PPOAgent
from training.network import DomibotNet


def _midgame(seed: int = 0) -> Game:
    import random

    game = Game(list(KINGDOM_CARDS)[:10], num_players=2, seed=seed)
    rng = random.Random(seed)
    while game.turn_number < 6 or game.pending_decision is not None or len(game.legal_actions()) < 2:
        game.step(rng.choice(game.legal_actions()))
    return game


def test_search_sees_only_what_the_player_could_know(monkeypatch):
    game = _midgame()
    before = game.clone()
    me, opp = game.current_decider(), 1 - game.current_decider()
    seen = {}

    def fake_run_mcts_batch(worlds, network, sims, **kwargs):
        seen["world"] = worlds[0]
        raise StopIteration  # stop before searching; only the world handed in matters here

    monkeypatch.setattr(agents, "run_mcts_batch", fake_run_mcts_batch)
    agent = DeterminizedSearchAgent(DomibotNet(), num_simulations=8, device=torch.device("cpu"), seed=1)
    try:
        agent.act(game)
    except StopIteration:
        pass
    world = seen["world"]
    # Public zones and my own hand are exactly as they are...
    assert world.players[me].hand == game.players[me].hand
    for i in (me, opp):
        assert world.players[i].discard == game.players[i].discard
        assert world.players[i].play_area == game.players[i].play_area
    assert world.supply == game.supply and world.trash == game.trash
    # ...hidden ones keep their sizes and contents but not their order/split.
    assert Counter(world.players[me].deck) == Counter(game.players[me].deck)
    assert len(world.players[opp].hand) == len(game.players[opp].hand)
    assert (Counter(world.players[opp].hand + world.players[opp].deck)
            == Counter(game.players[opp].hand + game.players[opp].deck))
    # The real game is untouched.
    assert game.players[opp].hand == before.players[opp].hand
    assert game.players[me].deck == before.players[me].deck


def test_search_agent_plays_a_legal_move():
    game = _midgame(seed=3)
    for determinizations in (1, 4):
        agent = DeterminizedSearchAgent(DomibotNet(), num_simulations=8, device=torch.device("cpu"), seed=2,
                                        determinizations=determinizations)
        assert agent.act(game) in game.legal_actions()


def _buy_phase(provinces: int, my_extra=(), opp_extra=()) -> Game:
    """My Buy phase with $8, `provinces` Provinces left, and extra Victory cards to set the score."""
    game = Game(list(KINGDOM_CARDS)[:10], num_players=2, seed=0)
    game.phase, game.supply["Province"] = Phase.BUY, provinces
    game.players[0].coins = 8
    game.players[0].discard += list(my_extra)
    game.players[1].discard += list(opp_extra)
    return game


def test_penultimate_province_is_the_second_to_last_while_not_ahead():
    assert agents.penultimate_province(_buy_phase(2, opp_extra=["Estate"]))
    assert agents.penultimate_province(_buy_phase(2))
    assert not agents.penultimate_province(_buy_phase(2, my_extra=["Estate"]))
    assert not agents.penultimate_province(_buy_phase(3, opp_extra=["Estate"]))


def test_playouts_score_a_move_that_ends_the_game():
    agent = PPOAgent(DomibotNet(), torch.device("cpu"), seed=0)
    assert agent.playout_win_rates(_buy_phase(1, opp_extra=["Duchy"]), [PROVINCE], 4) == [1.0]
    assert agent.playout_win_rates(_buy_phase(1, opp_extra=["Duchy"] * 3), [PROVINCE], 4) == [0.0]


def test_province_check_decides_the_spot(monkeypatch):
    monkeypatch.setattr(agents, "PLAYOUT_MOVES", 20)
    game = _buy_phase(2, opp_extra=["Estate"])
    agent = PPOAgent(DomibotNet(), torch.device("cpu"), province_playouts=2, seed=0)
    rates = agent.province_check(game)
    assert PROVINCE in rates and len(rates) == 2
    assert agent.act(game) in rates
