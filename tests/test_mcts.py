import pytest

from domibot import Action, Game, KINGDOM_CARDS
from domibot.models import END_ACTIONS
from training.agents import DomibotAgent
from training.env import terminal_value
from training.evaluate import play_game
from training.heuristics import heuristic_reaction
from training.mcts import materialize, run_mcts, run_mcts_batch, select_action, visit_distribution
from training.network import DomibotNet

KINGDOM = list(KINGDOM_CARDS)[:10]
ATTACKS = ["Militia", "Chapel", "Village", "Moat", "Smithy", "Workshop", "Bandit", "Council Room", "Festival", "Library"]


def _net():
    net = DomibotNet()
    net.eval()
    return net


def test_clone_is_independent_and_refused_mid_effect():
    game = Game(KINGDOM, num_players=2, seed=1)
    clone = game.clone()
    clone.players[0].hand.append("MUTATED")
    assert "MUTATED" not in game.players[0].hand and clone.rng is not game.rng
    game.players[0].hand = ["Cellar", "Copper", "Copper"]
    game.step(Action("PLAY", "Cellar"))
    with pytest.raises(RuntimeError):
        game.clone()


def test_terminal_value_is_margin_based():
    game = Game(KINGDOM, num_players=2, seed=1)
    for p in game.players:
        p.hand, p.discard, p.play_area, p.set_aside = [], [], [], []
    game.players[0].deck, game.players[1].deck = ["Province"] * 3, ["Province"]
    small = terminal_value(game, 0)
    assert 0 < small < 1 and terminal_value(game, 1) == -small
    game.players[0].deck = ["Province"] * 6
    assert terminal_value(game, 0) > small
    game.players[0].deck = ["Province"]
    assert terminal_value(game, 0) == 0.0


def test_search_visits_only_legal_actions_and_leaves_the_game_untouched():
    game = Game(KINGDOM, num_players=2, seed=2)
    game.step(END_ACTIONS)
    root = run_mcts(game, _net(), num_simulations=30)
    assert set(root.N) == set(game.legal_actions()) and sum(root.N.values()) == 30
    assert abs(sum(visit_distribution(root).values()) - 1.0) < 1e-6
    assert game.legal_actions() == root.game.legal_actions()
    assert select_action(root, temperature=0.0) in game.legal_actions()


def test_batched_search_runs_each_game_independently():
    games = [Game(KINGDOM, num_players=2, seed=s) for s in (10, 11, 12)]
    roots = run_mcts_batch(games, _net(), num_simulations=20)
    for game, root in zip(games, roots):
        assert set(root.N) == set(game.legal_actions()) and sum(root.N.values()) == 20


def test_materialize_matches_direct_stepping_through_a_reshuffle():
    kingdom = ["Sentry", "Village", "Moat", "Smithy", "Militia", "Workshop", "Bandit", "Council Room", "Festival",
               "Library"]
    game = Game(kingdom, num_players=2, seed=7)
    game.players[0].hand = ["Sentry", "Copper", "Copper"]
    game.players[0].deck = ["Silver"]  # Sentry looks at 2: forces a reshuffle mid-effect
    game.players[0].discard = ["Gold", "Estate", "Copper"]
    boundary = game.clone()
    path = [Action("PLAY", "Sentry")]
    game.step(path[0])
    while game.pending_decision is not None:
        opts = game.legal_actions()
        path.append(next((a for a in opts if a.card is None), opts[0]))
        game.step(path[-1])
    replayed = materialize(boundary, path)
    for a, b in zip(game.players, replayed.players):
        assert (a.hand, a.deck, a.discard, a.play_area) == (b.hand, b.deck, b.discard, b.play_area)
    assert game.trash == replayed.trash and game.supply == replayed.supply
    assert game.rng.getstate() == replayed.rng.getstate()


def test_search_of_a_card_choice_covers_its_options():
    game = Game(ATTACKS, num_players=2, seed=0)
    game.players[0].hand = ["Chapel", "Gold", "Silver", "Estate", "Copper"]
    path = [Action("PLAY", "Chapel")]
    root = run_mcts(game, _net(), num_simulations=30, path=path)
    assert set(root.N) == set(materialize(game, path).legal_actions()) and sum(root.N.values()) == 30
    assert game.pending_decision is None


def test_an_attacks_choice_belongs_to_the_attacked_player():
    game = Game(ATTACKS, num_players=2, seed=0)
    game.players[0].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    game.players[1].hand = ["Estate", "Copper", "Copper", "Smithy", "Village"]
    assert run_mcts(game, _net(), num_simulations=6, path=[Action("PLAY", "Militia")]).decider == 1


def test_domibot_agent_plays_legal_games_and_searches_card_choices():
    net = _net()
    agent = DomibotAgent(net, num_simulations=8, temperature=1.0)  # sampling keeps an untrained net from stalling
    game = play_game(agent, agent, KINGDOM, seed=5)
    assert set(game.winners()) <= set(game.get_scores())

    game = Game(ATTACKS, num_players=2, seed=0)
    game.players[0].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    agents = [DomibotAgent(net, num_simulations=4, temperature=1.0) for _ in range(2)]
    for agent in agents:
        agent.act(game)  # caches the boundary before the attack
    game.step(Action("PLAY", "Militia"))
    for _ in range(20):
        if game.is_game_over():
            break
        game.step(agents[game.current_decider()].act(game))


def test_domibot_agent_can_leave_card_choices_to_the_heuristic():
    game = Game(ATTACKS, num_players=2, seed=0)
    game.players[0].hand = ["Chapel", "Gold", "Silver", "Estate", "Copper"]
    game.step(Action("PLAY", "Chapel"))
    agent = DomibotAgent(_net(), num_simulations=4, search_sub_decisions=False)
    assert agent.act(game) == heuristic_reaction(game)
