"""Generate self-play games with MCTS + a network, producing training
examples for every phase-action decision: the encoded state, the legal
mask, the MCTS visit-count distribution (the policy target), and — filled
in once the game ends — the actual outcome from that decision's perspective
(the value target).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from domibot import Game, KINGDOM_CARDS

from . import encoding
from .heuristics import advance_to_next_phase_action
from .mcts import run_mcts, select_action, terminal_value, visit_distribution

DEFAULT_C_PUCT = 1.5
DEFAULT_TEMPERATURE_MOVES = 15  # phase-action decisions before switching to near-greedy play
DEFAULT_MAX_MOVES = 400  # safety cap; real games finish well under this
DEFAULT_ACTION_BIAS = 0.2  # see mcts._apply_action_continuation_bias


@dataclass
class Example:
    obs: np.ndarray
    mask: np.ndarray
    policy_target: np.ndarray  # length NUM_ACTIONS, mass only on that state's legal actions
    decider: int
    value_target: float = field(default=0.0)


def play_self_play_game(
    network: torch.nn.Module,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    seed: int | None = None,
) -> list[Example]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if kingdom is None:
        kingdom = py_rng.sample(list(KINGDOM_CARDS), 10)
    game = Game(kingdom, num_players=num_players, seed=seed)

    examples: list[Example] = []
    move_number = 0
    while not game.is_game_over() and move_number < max_moves:
        decider = game.current_decider()
        root = run_mcts(
            game, network, num_simulations, c_puct=c_puct, add_noise=True, action_bias=action_bias, rng=np_rng
        )
        dist = visit_distribution(root)

        policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
        for action, prob in dist.items():
            policy_target[encoding.ACTION_INDEX[action]] = prob

        examples.append(Example(
            obs=encoding.encode_observation(game, decider),
            mask=encoding.legal_action_mask(game),
            policy_target=policy_target,
            decider=decider,
        ))

        temperature = 1.0 if move_number < temperature_moves else 0.0
        action = select_action(root, temperature, rng=np_rng)
        advance_to_next_phase_action(game, action)
        move_number += 1

    game_over = game.is_game_over()
    for ex in examples:
        # a margin-based value from each example's own decider's perspective;
        # 0.0 for the rare case of hitting max_moves without a real result
        ex.value_target = terminal_value(game, ex.decider) if game_over else 0.0
    return examples


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.examples: list[Example] = []

    def add_game(self, examples: list[Example]) -> None:
        self.examples.extend(examples)
        overflow = len(self.examples) - self.capacity
        if overflow > 0:
            del self.examples[:overflow]

    def sample(self, batch_size: int, rng: random.Random) -> list[Example]:
        return rng.sample(self.examples, min(batch_size, len(self.examples)))

    def __len__(self) -> int:
        return len(self.examples)
