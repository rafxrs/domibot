from domibot import Action, Game
from domibot.models import END_ACTIONS, END_BUY
from training.heuristics import heuristic_reaction

KINGDOM = ["Chapel", "Village", "Moat", "Smithy", "Militia", "Workshop", "Bandit", "Council Room", "Festival",
           "Library"]


def _play_chapel(hand: list[str]) -> Game:
    """Play Chapel from `hand`, answering every choice with the heuristic."""
    game = Game(KINGDOM, num_players=2, seed=0)
    game.players[0].hand = hand
    game.step(Action("PLAY", "Chapel"))
    while game.pending_decision is not None:
        game.step(heuristic_reaction(game))
    return game


def test_gains_take_the_most_expensive_card_never_a_curse():
    game = Game(KINGDOM, num_players=2, seed=0)
    game.players[0].hand = ["Workshop", "Copper", "Copper", "Copper", "Estate"]
    game.step(Action("PLAY", "Workshop"))
    chosen = heuristic_reaction(game)
    assert chosen.card != "Curse" and game.cards[chosen.card].cost == 4


def test_a_forced_discard_gives_up_the_worst_card():
    game = Game(KINGDOM, num_players=2, seed=0)
    game.step(END_ACTIONS)
    game.step(END_BUY)
    game.players[1].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    game.players[0].hand = ["Estate", "Copper", "Copper", "Smithy", "Village"]
    game.step(Action("PLAY", "Militia"))
    assert heuristic_reaction(game).card == "Estate"


def test_an_optional_trash_stops_once_no_junk_is_left():
    game = _play_chapel(["Chapel", "Gold", "Silver", "Estate", "Copper"])
    assert sorted(game.players[0].hand) == ["Gold", "Silver"] and sorted(game.trash) == ["Copper", "Estate"]
    game = _play_chapel(["Chapel", "Gold", "Silver", "Village", "Smithy"])
    assert sorted(game.players[0].hand) == ["Gold", "Silver", "Smithy", "Village"] and game.trash == []
