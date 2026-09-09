"""Generate self-play games with MCTS + a network, producing training
examples for every decision -- phase actions *and* card-effect
sub-decisions (Chapel's trash choices, Militia's forced discard, ...) --
searched uniformly: the encoded state, the legal mask, the MCTS
visit-count distribution (the policy target), and — filled in once the
game ends — the actual outcome from that decision's perspective (the value
target).

`search_sub_decisions=False` (an ablation/fallback toggle, off by default)
resolves sub-decisions with the fixed, non-learned heuristic instead
(`training.heuristics.heuristic_reaction`) -- the behavior every version
before this one used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS

from . import encoding
from .heuristics import advance_to_next_phase_action
from .mcts import materialize, run_mcts, run_mcts_batch, select_action, terminal_value, visit_distribution

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
    search_sub_decisions: bool = True,
    sub_decision_simulations: int | None = None,
) -> list[Example]:
    if not search_sub_decisions:
        return _play_self_play_game_phase_actions_only(
            network, num_simulations, num_players, kingdom, c_puct,
            temperature_moves, max_moves, action_bias, seed,
        )
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if kingdom is None:
        kingdom = py_rng.sample(list(KINGDOM_CARDS), 10)
    game = Game(kingdom, num_players=num_players, seed=seed)

    boundary: Game = game
    path: list[Action] = []
    examples: list[Example] = []
    move_number = 0
    while not (not path and boundary.is_game_over()) and move_number < max_moves:
        sims = num_simulations if not path else (sub_decision_simulations or num_simulations)
        root = run_mcts(
            boundary, network, sims, c_puct=c_puct, add_noise=True,
            action_bias=action_bias, rng=np_rng, path=path,
        )
        decider = root.decider
        dist = visit_distribution(root)

        policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
        for action, prob in dist.items():
            policy_target[encoding.ACTION_INDEX[action]] = prob

        examples.append(Example(
            obs=encoding.encode_observation(root.game, decider),
            mask=encoding.legal_action_mask(root.game),
            policy_target=policy_target,
            decider=decider,
        ))

        temperature = 1.0 if move_number < temperature_moves else 0.0
        action = select_action(root, temperature, rng=np_rng)
        # Reuse the child search already materialized for `action` (present
        # whenever it was actually visited, i.e. whenever N[action] > 0 --
        # true here since select_action never picks an unvisited action).
        # Never step `root.game`/`root.boundary` directly: for a root that's
        # itself a fresh boundary, `_make_node` aliases boundary=game to
        # avoid a redundant clone, so mutating one would corrupt the other.
        child = root.children.get(action)
        child_game = child.game if child is not None else materialize(root.boundary, root.path + [action])
        if child_game.pending_decision is None:
            boundary, path = child_game, []
        else:
            boundary, path = root.boundary, root.path + [action]
        move_number += 1

    final_game = boundary if not path else materialize(boundary, path)
    game_over = not path and final_game.is_game_over()
    for ex in examples:
        # a margin-based value from each example's own decider's perspective;
        # 0.0 for the rare case of hitting max_moves without a real result
        ex.value_target = terminal_value(final_game, ex.decider) if game_over else 0.0
    return examples


def _play_self_play_game_phase_actions_only(
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
    """The original phase-actions-only self-play loop, kept verbatim as the
    `search_sub_decisions=False` fallback: every sub-decision between one
    phase-action and the next is resolved instantly by the fixed heuristic
    (`advance_to_next_phase_action`) rather than searched, so it never
    produces a training example."""
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


def play_self_play_games_batch(
    network: torch.nn.Module,
    num_games: int,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    device: torch.device | None = None,
    seed: int | None = None,
    search_sub_decisions: bool = True,
    sub_decision_simulations: int | None = None,
) -> list[list[Example]]:
    """Root-parallel version of `play_self_play_game`: plays `num_games`
    independent games side by side, one decision at a time, so every
    decision's MCTS search across all still-active games shares a single
    batched network call via `run_mcts_batch` instead of each game paying
    for its own batch-size-1 forward passes. Produces the exact same
    per-game example sequences `play_self_play_game` would (same move
    selection, same temperature schedule, same value-target backfill) --
    this only changes how the network is called, not what gets played."""
    if not search_sub_decisions:
        return _play_self_play_games_batch_phase_actions_only(
            network, num_games, num_simulations, num_players, kingdom, c_puct,
            temperature_moves, max_moves, action_bias, device, seed,
        )
    master_rng = random.Random(seed)
    boundaries: list[Game] = []
    paths: list[list[Action]] = []
    np_rngs: list[np.random.Generator] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        game_kingdom = kingdom if kingdom is not None else random.Random(g_seed).sample(list(KINGDOM_CARDS), 10)
        boundaries.append(Game(game_kingdom, num_players=num_players, seed=g_seed))
        paths.append([])
        np_rngs.append(np.random.default_rng(g_seed))

    examples_per_game: list[list[Example]] = [[] for _ in range(num_games)]
    move_numbers = [0] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        sims = [num_simulations if not paths[i] else (sub_decision_simulations or num_simulations) for i in idxs]
        # run_mcts_batch takes one uniform num_simulations today (per-root
        # budgets are a Stage 2 extension); until then, run each distinct
        # sim count in its own batched call -- almost always one call
        # (uniform sims) since sub_decision_simulations defaults to None.
        roots: list = [None] * len(idxs)
        for sim_count in sorted(set(sims)):
            group = [pos for pos, s in enumerate(sims) if s == sim_count]
            group_roots = run_mcts_batch(
                [boundaries[idxs[pos]] for pos in group], network, sim_count, c_puct=c_puct,
                add_noise=True, action_bias=action_bias, device=device,
                rngs=[np_rngs[idxs[pos]] for pos in group], paths=[paths[idxs[pos]] for pos in group],
            )
            for pos, root in zip(group, group_roots):
                roots[pos] = root

        for i, root in zip(idxs, roots):
            decider = root.decider
            dist = visit_distribution(root)

            policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
            for action, prob in dist.items():
                policy_target[encoding.ACTION_INDEX[action]] = prob

            examples_per_game[i].append(Example(
                obs=encoding.encode_observation(root.game, decider),
                mask=encoding.legal_action_mask(root.game),
                policy_target=policy_target,
                decider=decider,
            ))

            temperature = 1.0 if move_numbers[i] < temperature_moves else 0.0
            action = select_action(root, temperature, rng=np_rngs[i])
            child = root.children.get(action)
            child_game = child.game if child is not None else materialize(root.boundary, root.path + [action])
            if child_game.pending_decision is None:
                boundaries[i], paths[i] = child_game, []
            else:
                boundaries[i], paths[i] = root.boundary, root.path + [action]
            move_numbers[i] += 1
            terminal = not paths[i] and boundaries[i].is_game_over()
            if terminal or move_numbers[i] >= max_moves:
                active[i] = False

    for i in range(num_games):
        final_game = boundaries[i] if not paths[i] else materialize(boundaries[i], paths[i])
        game_over = not paths[i] and final_game.is_game_over()
        for ex in examples_per_game[i]:
            ex.value_target = terminal_value(final_game, ex.decider) if game_over else 0.0
    return examples_per_game


def _play_self_play_games_batch_phase_actions_only(
    network: torch.nn.Module,
    num_games: int,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[list[Example]]:
    """The original phase-actions-only batched self-play loop, kept
    verbatim as the `search_sub_decisions=False` fallback -- see
    `_play_self_play_game_phase_actions_only`."""
    master_rng = random.Random(seed)
    games: list[Game] = []
    np_rngs: list[np.random.Generator] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        if kingdom is None:
            game_kingdom = random.Random(g_seed).sample(list(KINGDOM_CARDS), 10)
        else:
            game_kingdom = kingdom
        games.append(Game(game_kingdom, num_players=num_players, seed=g_seed))
        np_rngs.append(np.random.default_rng(g_seed))

    examples_per_game: list[list[Example]] = [[] for _ in range(num_games)]
    move_numbers = [0] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        roots = run_mcts_batch(
            [games[i] for i in idxs],
            network,
            num_simulations,
            c_puct=c_puct,
            add_noise=True,
            action_bias=action_bias,
            device=device,
            rngs=[np_rngs[i] for i in idxs],
        )

        for i, root in zip(idxs, roots):
            game = games[i]
            decider = game.current_decider()
            dist = visit_distribution(root)

            policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
            for action, prob in dist.items():
                policy_target[encoding.ACTION_INDEX[action]] = prob

            examples_per_game[i].append(Example(
                obs=encoding.encode_observation(game, decider),
                mask=encoding.legal_action_mask(game),
                policy_target=policy_target,
                decider=decider,
            ))

            temperature = 1.0 if move_numbers[i] < temperature_moves else 0.0
            action = select_action(root, temperature, rng=np_rngs[i])
            advance_to_next_phase_action(game, action)
            move_numbers[i] += 1
            if game.is_game_over() or move_numbers[i] >= max_moves:
                active[i] = False

    for i, game in enumerate(games):
        game_over = game.is_game_over()
        for ex in examples_per_game[i]:
            ex.value_target = terminal_value(game, ex.decider) if game_over else 0.0
    return examples_per_game


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
