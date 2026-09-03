import json
import random

from domibot import Game, KINGDOM_CARDS
from domibot.models import END_ACTIONS, END_BUY


def play_a_few_turns(seed: int = 0) -> Game:
    rng = random.Random(seed)
    kingdom = rng.sample(list(KINGDOM_CARDS), 10)
    game = Game(kingdom, num_players=2, seed=seed)
    for _ in range(30):
        if game.is_game_over():
            break
        game.step(rng.choice(game.legal_actions()))
    return game


def test_action_log_numbers_turns_per_player():
    # Player 0's 2nd turn should log as "turn 2", regardless of how many
    # turns player 1 has taken in between.
    game = Game(list(KINGDOM_CARDS)[:10], num_players=2, seed=1)
    for _ in range(3):  # P0 turn 1, P1 turn 1, P0 turn 2 (started)
        game.step(END_ACTIONS)
        game.step(END_BUY)

    assert game.action_log[0] == (1, 0, END_ACTIONS)  # P0 turn 1
    assert game.action_log[2] == (1, 1, END_ACTIONS)  # P1 turn 1
    assert game.action_log[4] == (2, 0, END_ACTIONS)  # P0 turn 2


def test_save_log_text(tmp_path):
    game = play_a_few_turns()
    out = tmp_path / "nested" / "game.log"
    game.save_log(out)  # also exercises parent-dir auto-creation

    text = out.read_text(encoding="utf-8")
    assert f"kingdom: {', '.join(sorted(game.kingdom))}" in text
    assert f"seed: {game.seed}" in text
    for entry in game.action_log:
        assert f"turn {entry.turn:>3}  P{entry.player}  {entry.action}" in text


def test_save_log_json_round_trips_actions(tmp_path):
    game = play_a_few_turns(seed=2)
    out = tmp_path / "game.json"
    game.save_log(out, fmt="json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["seed"] == 2
    assert data["num_players"] == 2
    assert len(data["actions"]) == len(game.action_log)
    for recorded, entry in zip(data["actions"], game.action_log):
        assert recorded["turn"] == entry.turn
        assert recorded["player"] == entry.player
        assert recorded["verb"] == entry.action.verb
        assert recorded["card"] == entry.action.card


def test_save_log_rejects_unknown_format(tmp_path):
    game = play_a_few_turns(seed=3)
    try:
        game.save_log(tmp_path / "x.log", fmt="xml")
        assert False, "expected ValueError"
    except ValueError:
        pass
