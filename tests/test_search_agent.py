from collections import Counter

import torch

import training.agents as agents
from domibot import Game, KINGDOM_CARDS
from training.agents import DeterminizedSearchAgent
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
