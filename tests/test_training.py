import random

import numpy as np

from domibot import Game, KINGDOM_CARDS
from training import encoding
from training.agents import BigMoneyAgent, RandomAgent
from training.env import DominionEnv
from training.evaluate import play_game, play_match


def test_action_vocab_covers_every_legal_action_across_random_games():
    """Sweep every kingdom card through random play and check every action
    the engine ever offers is in the fixed ACTION_VOCAB (i.e. index_to_action
    round-trips for anything legal_actions() can produce)."""
    rng = random.Random(0)
    all_kingdom = list(KINGDOM_CARDS)
    remaining = list(all_kingdom)
    game_idx = 0
    while remaining:
        chunk = remaining[:10]
        remaining = remaining[10:]
        if len(chunk) < 10:
            filler = [c for c in all_kingdom if c not in chunk]
            chunk = chunk + rng.sample(filler, 10 - len(chunk))
        game = Game(chunk, num_players=rng.choice([2, 3, 4]), seed=game_idx)
        steps = 0
        while not game.is_game_over():
            for a in game.legal_actions():
                assert a in encoding.ACTION_INDEX, f"{a!r} missing from ACTION_VOCAB"
            game.step(rng.choice(game.legal_actions()))
            steps += 1
            assert steps < 50_000
        game_idx += 1


def test_encode_observation_shape_and_dtype():
    game = Game(list(KINGDOM_CARDS)[:10], num_players=3, seed=1)
    for player_idx in range(3):
        obs = encoding.encode_observation(game, player_idx)
        assert obs.shape == (encoding.OBS_DIM,)
        assert obs.dtype == np.float32


def test_legal_action_mask_matches_legal_actions():
    game = Game(list(KINGDOM_CARDS)[:10], num_players=2, seed=2)
    mask = encoding.legal_action_mask(game)
    assert mask.dtype == bool
    legal = set(game.legal_actions())
    for i, action in enumerate(encoding.ACTION_VOCAB):
        assert mask[i] == (action in legal)


def test_env_random_episode_terminates_with_consistent_reward():
    env = DominionEnv(num_players=2)
    obs, info = env.reset(seed=7)
    rng = random.Random(7)
    terminated = truncated = False
    reward = 0.0
    steps = 0
    while not (terminated or truncated):
        legal_idx = np.flatnonzero(obs["action_mask"])
        assert len(legal_idx) > 0
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal_idx)))
        steps += 1
        assert steps < 50_000
    assert terminated
    assert reward in (-1.0, 0.0, 1.0)
    assert set(info["winners"]) <= set(info["scores"])


def test_env_rejects_illegal_action_index():
    env = DominionEnv(num_players=2)
    obs, _ = env.reset(seed=3)
    illegal_idx = int(np.flatnonzero(~obs["action_mask"])[0])
    try:
        env.step(illegal_idx)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_big_money_beats_random_decisively():
    result = play_match(BigMoneyAgent(), RandomAgent(seed=11), n_games=20, seed=11)
    assert result["agent_a_wins"] >= 18  # BigMoney should win nearly every game


def test_play_game_runs_kingdom_to_completion():
    kingdom = list(KINGDOM_CARDS)[:10]
    game = play_game(BigMoneyAgent(), BigMoneyAgent(), kingdom, seed=9)
    assert game.is_game_over()
