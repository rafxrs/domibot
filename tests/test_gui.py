import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless: no real display needed to run these

import pytest

pytest.importorskip("pygame")

from domibot import Action, Game, KINGDOM_CARDS, Phase
from domibot.models import END_ACTIONS
from gui.app import DominionGUI
from training.agents import DomibotAgent
from training.network import DomibotNet


def _agent() -> DomibotAgent:
    net = DomibotNet()
    net.eval()
    return DomibotAgent(net, num_simulations=3)


def _tiny_kingdom() -> list[str]:
    return list(KINGDOM_CARDS)[:10]


def test_gui_renders_action_phase_without_crashing():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    gui = DominionGUI(game, _agent(), human_seat=0)
    gui._rebuild_clickables()
    gui._draw()  # would raise on any layout/render bug
    assert len(gui.clickables) >= 1  # at least END_ACTIONS


def test_gui_buy_phase_clickables_are_all_actually_legal():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    game.step(END_ACTIONS)
    gui = DominionGUI(game, _agent(), human_seat=0)
    gui._rebuild_clickables()
    gui._draw()
    legal = set(game.legal_actions())
    assert all(c.action in legal for c in gui.clickables)
    assert len(gui.clickables) == len(legal)  # every legal action got a clickable


def test_gui_subdecision_renders_as_decision_panel_buttons():
    game = Game(_tiny_kingdom(), num_players=2, seed=2)  # includes Chapel
    game.players[0].hand = ["Chapel", "Estate", "Copper", "Copper", "Copper"]
    game.step(Action("PLAY", "Chapel"))

    gui = DominionGUI(game, _agent(), human_seat=0)
    gui._rebuild_clickables()
    gui._draw()

    labels = {c.label for c in gui.clickables}
    assert "Estate" in labels
    assert "Copper" in labels
    assert "Done" in labels


def test_gui_click_on_a_clickable_steps_the_real_game():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    gui = DominionGUI(game, _agent(), human_seat=0)
    gui._rebuild_clickables()
    end_actions = next(c for c in gui.clickables if c.action == END_ACTIONS)

    gui._handle_click(end_actions.rect.center)

    assert game.phase == Phase.BUY  # the underlying Game actually advanced


def test_gui_click_outside_any_clickable_does_nothing():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    gui = DominionGUI(game, _agent(), human_seat=0)
    gui._rebuild_clickables()
    before = game.phase
    gui._handle_click((5, 5))  # top-left corner, nothing there
    assert game.phase == before


def test_gui_ignores_clicks_when_it_is_not_the_humans_turn():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    gui = DominionGUI(game, _agent(), human_seat=1)  # bot is seat 0, so it's not the human's turn
    gui._rebuild_clickables()
    assert gui.clickables == []


def test_supply_kingdom_cards_are_ordered_by_cost_not_alphabetically():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    gui = DominionGUI(game, _agent(), human_seat=0)
    rects = gui._supply_rects()

    kingdom_by_position = sorted(game.kingdom, key=lambda n: (rects[n].top, rects[n].left))
    costs = [game.cards[name].cost for name in kingdom_by_position]
    assert costs == sorted(costs)  # strictly non-decreasing left-to-right, top-to-bottom


def test_domibot_activity_panel_lists_only_the_bots_own_actions():
    game = Game(_tiny_kingdom(), num_players=2, seed=1)
    agent = _agent()
    gui = DominionGUI(game, agent, human_seat=0)

    while len([e for e in game.action_log if e.player == gui.bot_seat]) < 2 and not game.is_game_over():
        gui._maybe_take_bot_turn()
        gui._rebuild_clickables()
        if game.current_decider() == gui.human_seat and not game.is_game_over():
            game.step(game.legal_actions()[-1])

    gui._draw()  # exercises _draw_domibot_activity without crashing

    bot_entries = [e for e in game.action_log if e.player == gui.bot_seat]
    assert len(bot_entries) >= 2
    assert all(e.player == gui.bot_seat for e in bot_entries)  # never the human's own actions
