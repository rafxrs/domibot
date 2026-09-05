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
