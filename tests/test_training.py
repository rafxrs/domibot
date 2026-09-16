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


# --- observation must disambiguate what a sub-decision is actually about ---

def test_source_card_distinguishes_same_kind_decisions():
    # Chapel ("dump junk") and Remodel ("give up your best card to upgrade
    # it") both raise SELECT_CARD with TRASH options over the same hand. If
    # the observation can't tell them apart, no policy can answer both --
    # the action mask is applied to the output, never seen as input.
    from domibot import Action, Game

    kingdom = ["Chapel", "Remodel", "Cellar", "Militia", "Village",
               "Smithy", "Market", "Moat", "Workshop", "Festival"]

    def obs_after_playing(card):
        game = Game(kingdom, num_players=2, seed=4)
        p = game.players[0]
        p.hand = ["Estate", "Copper", "Gold", "Estate", card]
        p.discard = ["Remodel" if card == "Chapel" else "Chapel"]  # owns both
        p.deck = ["Copper", "Copper"]
        game.step(Action("PLAY", card))
        return game, encoding.encode_observation(game, game.current_decider())

    chapel_game, chapel_obs = obs_after_playing("Chapel")
    remodel_game, remodel_obs = obs_after_playing("Remodel")

    assert chapel_game.pending_decision.source_card == "Chapel"
    assert remodel_game.pending_decision.source_card == "Remodel"
    assert not np.array_equal(chapel_obs, remodel_obs)


def test_set_aside_cards_are_visible_in_the_observation():
    # Sentry's two revealed cards live in set_aside while the trash/discard/
    # reorder decisions are pending -- they are literally what's being
    # decided about, so they have to be in the observation.
    from domibot import Action, Game

    game = Game(["Sentry", "Chapel", "Remodel", "Cellar", "Militia",
                 "Village", "Smithy", "Market", "Moat", "Festival"], num_players=2, seed=4)
    p = game.players[0]
    p.hand = ["Sentry", "Copper", "Copper", "Estate", "Estate"]
    p.deck = ["Gold", "Curse", "Copper", "Copper"]
    game.step(Action("PLAY", "Sentry"))

    assert len(p.set_aside) == 2
    obs = encoding.encode_observation(game, game.current_decider())
    block = obs[encoding.NUM_CARDS * 4:encoding.NUM_CARDS * 5]
    for card in p.set_aside:
        assert block[encoding.CARD_INDEX[card]] >= 1


def test_no_source_card_encoded_at_a_plain_phase_action():
    from domibot import Game

    game = Game(list(KINGDOM_CARDS)[:10], num_players=2, seed=1)
    assert game.pending_decision is None
    obs = encoding.encode_observation(game, 0)
    assert not obs[encoding.NUM_CARDS * 5:encoding.NUM_CARDS * 6].any()
