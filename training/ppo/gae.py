"""Generalized Advantage Estimation, run along each player's own decisions in a game."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from domibot import Game

from ..env import terminal_value


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int  # index into encoding.ACTION_VOCAB
    log_prob: float  # under the policy that acted
    value: float  # V(obs) at collection time
    decider: int
    turn_number: int
    explore: bool = False  # a steering plan's buy, not the policy's: trains no policy
    # Filled in by compute_gae.
    reward: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0  # the value target: advantage + value


def compute_gae(transitions: list[Transition], final_game: Game, game_over: bool, gamma: float = 1.0,
                lam: float = 0.95, reward_fn: Callable[[Game, int], float] = terminal_value) -> None:
    """Fill in each transition's reward, advantage and return, in place.

    The reward is 0 except at each player's last decision of a finished game.
    A game cut off at the move cap (`game_over=False`) gets no reward, and its
    last decision bootstraps from its own value.
    """
    by_decider: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(transitions):
        by_decider[t.decider].append(i)
    for decider, idxs in by_decider.items():
        values = [transitions[i].value for i in idxs]
        rewards = [0.0] * len(idxs)
        if game_over:
            rewards[-1] = float(reward_fn(final_game, decider))
        next_values = values[1:] + [0.0 if game_over else values[-1]]
        running = 0.0
        for t in range(len(idxs) - 1, -1, -1):
            running = rewards[t] + gamma * next_values[t] - values[t] + gamma * lam * running
            tr = transitions[idxs[t]]
            tr.reward, tr.advantage, tr.return_ = rewards[t], running, running + values[t]
