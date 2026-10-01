"""A Gym-style wrapper around `domibot.Game`, plus the shared reward functions.

`step` always acts for whoever must decide now (the active player, or an
opponent answering an attack), and each observation is from the next
decider's point of view, with a mask over `encoding.ACTION_VOCAB`.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np

from domibot import Game, KINGDOM_CARDS

from . import encoding

MAX_MOVES = 1000  # decision cap for a training game
MARGIN_SCALE = 10.0  # a 10-VP margin is worth tanh(1) ~ 0.76

Observation = dict  # {"observation": np.ndarray, "action_mask": np.ndarray[bool]}


def random_kingdom(rng: random.Random, home: tuple[str, ...] | list[str] = ()) -> list[str]:
    """`home` plus random kingdom cards up to 10."""
    rest = [c for c in KINGDOM_CARDS if c not in home]
    return list(home) + rng.sample(rest, 10 - len(home))


def terminal_value(game: Game, perspective: int) -> float:
    """tanh of `perspective`'s score minus the other players' mean, over MARGIN_SCALE."""
    scores = game.get_scores()
    others = [s for p, s in scores.items() if p != perspective]
    return math.tanh((scores[perspective] - sum(others) / len(others)) / MARGIN_SCALE)


def win_weighted_value(game: Game, perspective: int, win_weight: float = 0.8) -> float:
    """`win_weight` x win/loss (+1/-1, 0 for a tie) plus the rest as `terminal_value`."""
    winners = game.winners()
    result = 0.0 if len(winners) != 1 else (1.0 if winners[0] == perspective else -1.0)
    return win_weight * result + (1.0 - win_weight) * terminal_value(game, perspective)


class DominionEnv:
    def __init__(self, num_players: int = 2, max_steps: int = 100_000,
                 reward_fn: Optional[Callable[[Game, int], float]] = None, full_obs: bool = False):
        """`reward_fn(game, player)` replaces the +1/-1/0 terminal reward; `full_obs`
        emits `encoding.encode_full_observation` instead of the base encoding."""
        self.num_players = num_players
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.full_obs = full_obs
        self.game: Optional[Game] = None
        self._steps = 0

    def reset(self, kingdom: Optional[list[str]] = None, seed: Optional[int] = None) -> tuple[Observation, dict]:
        kingdom = kingdom or random_kingdom(random.Random(seed))
        self.game = Game(kingdom, num_players=self.num_players, seed=seed)
        self._steps = 0
        return self._observe(), {"kingdom": kingdom}

    def step(self, action_index: int) -> tuple[Observation, float, bool, bool, dict]:
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        action = encoding.index_to_action(action_index)
        legal = self.game.legal_actions()
        if action not in legal:
            raise ValueError(f"action {action!r} is not legal right now (legal: {legal})")
        actor = self.game.current_decider()
        self.game.step(action)
        self._steps += 1
        terminated = self.game.is_game_over()
        truncated = not terminated and self._steps >= self.max_steps
        info = {"scores": self.game.get_scores(), "winners": self.game.winners()} if terminated else {}
        return self._observe(), self._reward_for(actor) if terminated else 0.0, terminated, truncated, info

    def _reward_for(self, player: int) -> float:
        if self.reward_fn is not None:
            return self.reward_fn(self.game, player)
        winners = self.game.winners()
        return 0.0 if len(winners) != 1 else (1.0 if winners[0] == player else -1.0)

    def _observe(self) -> Observation:
        over = self.game.is_game_over()
        player = self.game.current_player if over else self.game.current_decider()
        encode = encoding.encode_full_observation if self.full_obs else encoding.encode_observation
        mask = np.zeros(encoding.NUM_ACTIONS, dtype=bool) if over else encoding.legal_action_mask(self.game)
        return {"observation": encode(self.game, player), "action_mask": mask}
