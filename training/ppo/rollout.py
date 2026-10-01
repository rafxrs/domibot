"""Batched rollouts: many games stepped side by side, one forward pass per round."""
from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch

from .. import encoding
from ..env import MAX_MOVES, DominionEnv, random_kingdom, terminal_value
from .explore import ExploreConfig, SteeredPlayer, focus_seat, make_steered, steer
from .gae import Transition, compute_gae


class GameBatch:
    """Games stepped side by side; records a `Transition` for each decision it trains on."""

    def __init__(self, boards: list[tuple[list[str], int]], max_moves: int, reward_fn: Callable, full_obs: bool,
                 num_players: int = 2):
        self.envs: list[DominionEnv] = []
        self.obs: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        for kingdom, seed in boards:
            env = DominionEnv(num_players=num_players, max_steps=max_moves, reward_fn=reward_fn, full_obs=full_obs)
            obs, _info = env.reset(kingdom=kingdom, seed=seed)
            self.envs.append(env)
            self.obs.append(obs["observation"])
            self.masks.append(obs["action_mask"])
        n = len(self.envs)
        self.transitions: list[list[Transition]] = [[] for _ in range(n)]
        self.active = [True] * n
        self.game_over = [False] * n
        self.reward_fn = reward_fn

    def running(self) -> list[int]:
        return [i for i, a in enumerate(self.active) if a]

    def step_network(self, idxs: list[int], network: torch.nn.Module, device: torch.device, record: bool = True,
                     steered: Optional[list[list[SteeredPlayer]]] = None) -> None:
        """One batched forward pass over `idxs`, sampling each game's move from the
        masked policy. A steered player's buys are its plan's (`explore=True`)."""
        if not idxs:
            return
        obs_t = torch.from_numpy(np.stack([self.obs[i] for i in idxs])).to(device)
        mask_t = torch.from_numpy(np.stack([self.masks[i] for i in idxs])).to(device)
        with torch.no_grad():
            logits, values = network(obs_t)
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask_t, -1e9))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        for pos, i in enumerate(idxs):
            game = self.envs[i].game
            decider = game.current_decider()
            action_idx, log_prob = int(actions[pos].item()), float(log_probs[pos].item())
            planned = steer(game, steered[i]) if steered is not None else None
            if planned is not None:
                action_idx = encoding.action_to_index(planned)
                log_prob = float(dist.logits[pos, action_idx].item())
            if record:
                self.transitions[i].append(Transition(
                    obs=self.obs[i], mask=self.masks[i], action=action_idx, log_prob=log_prob,
                    value=float(values[pos].item()), decider=decider, turn_number=game.players[decider].turns_taken,
                    explore=planned is not None))
            self._advance(i, action_idx)

    def step_agent(self, idxs: list[int], agent) -> None:
        """One `agent.act` move per game, recorded nowhere."""
        for i in idxs:
            self._advance(i, encoding.action_to_index(agent.act(self.envs[i].game)))

    def _advance(self, i: int, action_idx: int) -> None:
        obs, _reward, terminated, truncated, _info = self.envs[i].step(action_idx)
        if terminated or truncated:
            self.active[i] = False
            self.game_over[i] = terminated
        else:
            self.obs[i] = obs["observation"]
            self.masks[i] = obs["action_mask"]

    def finish(self, gamma: float, lam: float) -> list[list[Transition]]:
        for i, env in enumerate(self.envs):
            compute_gae(self.transitions[i], env.game, self.game_over[i], gamma=gamma, lam=lam,
                        reward_fn=self.reward_fn)
        return self.transitions


def collect_rollouts(network: torch.nn.Module, num_games: int, num_players: int = 2, kingdom: list[str] | None = None,
                     max_moves: int = MAX_MOVES, device: Optional[torch.device] = None, seed: int | None = None,
                     reward_fn: Callable = terminal_value, gamma: float = 1.0, lam: float = 0.95,
                     full_obs: bool | None = None, explore: ExploreConfig | None = None,
                     stats: dict | None = None) -> list[list[Transition]]:
    """Self-play games, one `Transition` list per game with GAE filled in.

    `full_obs` defaults to whether `network` reads the extras. `explore` steers
    players' buys in a share of the games (see `explore.py`); `stats`, if given,
    gets how often the steered player of interest won ("won" of "games").
    """
    device = device or next(network.parameters()).device
    full_obs = bool(getattr(network, "extra_dim", 0)) if full_obs is None else full_obs
    master_rng = random.Random(seed)
    explore_rng = random.Random(master_rng.randrange(2**31))
    boards, plans = [], []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        game_kingdom = kingdom if kingdom is not None else random_kingdom(random.Random(g_seed))
        steered = make_steered(game_kingdom, num_players, explore, explore_rng) if explore else []
        boards.append((steered[0].board if steered and steered[0].board else game_kingdom, g_seed))
        plans.append(steered)
    batch = GameBatch(boards, max_moves, reward_fn, full_obs, num_players)
    while idxs := batch.running():
        batch.step_network(idxs, network, device, steered=plans)
    transitions = batch.finish(gamma, lam)
    if stats is not None:
        for i, steered in enumerate(plans):
            seat = focus_seat(steered)
            if seat is not None and batch.game_over[i]:
                stats["games"] = stats.get("games", 0) + 1
                stats["won"] = stats.get("won", 0) + (batch.envs[i].game.winners() == [seat])
    return transitions
