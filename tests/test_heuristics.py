from domibot import Action, Game
from domibot.models import END_ACTIONS, END_BUY

from training.heuristics import heuristic_reaction


def test_gain_decisions_never_pick_curse_over_a_better_option():
    # Regression: heuristic_reaction used to resolve every SELECT_CARD
    # decision (trash/discard/topdeck *and* gain) with the same "give up
    # your worst card" ranking, where Curse ranks best -- so a GAIN
    # decision (Workshop, Artisan, ...) would always take a free Curse
    # over a Silver, Village, Smithy, or anything else on offer.
    kingdom = ["Workshop", "Village", "Moat", "Smithy", "Militia",
               "Chapel", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.players[0].hand = ["Workshop", "Copper", "Copper", "Copper", "Estate"]
    game.step(Action("PLAY", "Workshop"))

    assert game.pending_decision is not None
    chosen = heuristic_reaction(game)

    assert chosen.card != "Curse"
    assert game.cards[chosen.card].cost == 4  # the most expensive card on offer


def test_trash_and_discard_decisions_still_give_up_the_worst_card():
    # The GAIN-specific branch must not change existing give-up-your-worst
    # behavior for the (much more common) trash/discard/topdeck case.
    kingdom = ["Militia", "Village", "Moat", "Smithy", "Workshop",
               "Chapel", "Bandit", "Council Room", "Festival", "Library"]
    game = Game(kingdom, num_players=2, seed=0)
    game.step(END_ACTIONS)  # player 0's turn: nothing to play here
    game.step(END_BUY)
    assert game.current_decider() == 1

    game.players[1].hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    game.players[0].hand = ["Estate", "Copper", "Copper", "Smithy", "Village"]
    game.step(Action("PLAY", "Militia"))

    assert game.pending_decision is not None
    chosen = heuristic_reaction(game)

    assert chosen.card == "Estate"  # worse than Copper, and no Curse in hand
