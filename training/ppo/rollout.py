"""Batched PPO self-play rollout collection: `num_games` `DominionEnv`
instances stepped side by side, sharing one batched network forward pass
per round -- the same root-parallel idea `self_play.play_self_play_games_batch`
used for MCTS (`mcts.run_mcts_batch`'s docstring), minus the tree. PPO
never explores a hypothetical future, only ever advances the one real
env, so there's no `boundary`/`path`/`materialize` machinery to worry
about here at all: `DominionEnv.step` is always safe to call directly,
every decision (phase action or card-effect sub-decision alike) included.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch

from ..env import DominionEnv
from ..mcts import terminal_value
from ..self_play import DEFAULT_MAX_MOVES, _sample_kingdom
from .gae import Transition, compute_gae


def collect_rollouts(
    network: torch.nn.Module,
    num_games: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    device: Optional[torch.device] = None,
    seed: int | None = None,
    min_sub_decision_cards: int = 0,
    reward_fn: Callable = terminal_value,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> list[list[Transition]]:
    """Plays `num_games` independent self-play games to completion (or
    `max_moves`), and returns one list of `Transition`s per game with
    `.advantage`/`.return_` already filled in by `compute_gae` -- the same
    "generate, then backfill, in one call" shape
    `self_play.play_self_play_games_batch` already uses, so this drops
    into a PPO training loop the same way that dropped into `train.py`.

    Every decision (phase action *and* card-effect sub-decision, per the
    same policy the MCTS lineage validated) produces one `Transition`,
    sampled from the current policy with legal-action masking (standard
    invalid-action masking: illegal logits set to -inf before the
    categorical distribution, mirroring `mcts.evaluate_node`'s numpy
    version of the same mask)."""
    if device is None:
        device = next(network.parameters()).device
    master_rng = random.Random(seed)

    envs: list[DominionEnv] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        game_kingdom = kingdom if kingdom is not None else \
            _sample_kingdom(random.Random(g_seed), min_sub_decision_cards)
        env = DominionEnv(num_players=num_players, max_steps=max_moves, reward_fn=reward_fn)
        obs, _info = env.reset(kingdom=game_kingdom, seed=g_seed)
        envs.append(env)
        obs_list.append(obs["observation"])
        mask_list.append(obs["action_mask"])

    transitions_per_game: list[list[Transition]] = [[] for _ in range(num_games)]
    game_over = [False] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        _step_group(idxs, network, envs, obs_list, mask_list, transitions_per_game,
                    active, game_over, device, record=True)

    for i in range(num_games):
        compute_gae(transitions_per_game[i], envs[i].game, game_over[i],
                    gamma=gamma, lam=lam, reward_fn=reward_fn)
    return transitions_per_game


def _step_group(
    idxs: list[int],
    network: torch.nn.Module,
    envs: list[DominionEnv],
    obs_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    transitions_per_game: list[list[Transition]],
    active: list[bool],
    game_over: list[bool],
    device: torch.device,
    record: bool,
) -> None:
    """Runs one batched forward pass of `network` over `idxs`' current
    observations, samples a masked action for each, and steps that game's
    env -- recording a `Transition` only if `record` (False for a frozen
    cross-play opponent's decisions, which advance the game but must never
    become a training example, per `collect_cross_play_rollouts`)."""
    if not idxs:
        return
    obs_t = torch.from_numpy(np.stack([obs_list[i] for i in idxs])).to(device)
    mask_t = torch.from_numpy(np.stack([mask_list[i] for i in idxs])).to(device)
    with torch.no_grad():
        logits, values = network(obs_t)
    masked_logits = logits.masked_fill(~mask_t, -1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    for pos, i in enumerate(idxs):
        env = envs[i]
        decider = env.game.current_decider()
        action_idx = int(actions[pos].item())
        if record:
            transitions_per_game[i].append(Transition(
                obs=obs_list[i],
                mask=mask_list[i],
                action=action_idx,
                log_prob=float(log_probs[pos].item()),
                value=float(values[pos].item()),
                decider=decider,
                turn_number=env.game.players[decider].turns_taken,
            ))
        obs, _reward, terminated, truncated, _info = env.step(action_idx)
        if terminated or truncated:
            active[i] = False
            game_over[i] = terminated
        else:
            obs_list[i] = obs["observation"]
            mask_list[i] = obs["action_mask"]


def collect_cross_play_rollouts(
    network: torch.nn.Module,
    opponent_network: torch.nn.Module,
    num_games: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    device: Optional[torch.device] = None,
    seed: int | None = None,
    min_sub_decision_cards: int = 0,
    reward_fn: Callable = terminal_value,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> list[list[Transition]]:
    """Like `collect_rollouts`, but each game has one randomly assigned
    seat (`current_seat`) decided by `network` -- the network being
    trained, whose decisions produce `Transition`s exactly as usual --
    while every other seat is decided by `opponent_network`, a frozen
    snapshot from earlier training: its decisions advance the game
    realistically but never produce a `Transition`, so the PPO update
    never pulls the current network back toward an older policy.

    Ports `self_play.play_cross_play_games`'s rationale to PPO: pure
    self-play optimizes toward "beat the version of myself I'm currently
    playing against," which can make a partially-executed complex
    strategy look like a regression against the current population even
    when a well-executed version of it would win. Facing a genuinely
    different, historical opponent some of the time breaks that
    self-reinforcement. Unlike the MCTS version, there's no boundary/path
    ambiguity to compromise on here -- PPO steps the one real env
    directly, so `current_decider() == current_seat[i]` unambiguously
    picks the right network for every decision, sub-decisions (e.g. a
    forced reaction) included."""
    if device is None:
        device = next(network.parameters()).device
    master_rng = random.Random(seed)

    envs: list[DominionEnv] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    current_seat: list[int] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        game_kingdom = kingdom if kingdom is not None else \
            _sample_kingdom(random.Random(g_seed), min_sub_decision_cards)
        env = DominionEnv(num_players=num_players, max_steps=max_moves, reward_fn=reward_fn)
        obs, _info = env.reset(kingdom=game_kingdom, seed=g_seed)
        envs.append(env)
        obs_list.append(obs["observation"])
        mask_list.append(obs["action_mask"])
        current_seat.append(master_rng.randrange(num_players))

    transitions_per_game: list[list[Transition]] = [[] for _ in range(num_games)]
    game_over = [False] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        own_idxs = [i for i in idxs if envs[i].game.current_decider() == current_seat[i]]
        opp_idxs = [i for i in idxs if envs[i].game.current_decider() != current_seat[i]]
        _step_group(own_idxs, network, envs, obs_list, mask_list, transitions_per_game,
                    active, game_over, device, record=True)
        _step_group(opp_idxs, opponent_network, envs, obs_list, mask_list, transitions_per_game,
                    active, game_over, device, record=False)

    for i in range(num_games):
        compute_gae(transitions_per_game[i], envs[i].game, game_over[i],
                    gamma=gamma, lam=lam, reward_fn=reward_fn)
    return transitions_per_game
