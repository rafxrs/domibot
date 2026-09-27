"""A league of fixed opponents for PPO: part of every iteration's games are
played against past checkpoints, other networks, and scripted strategies
instead of the current network, with opponents picked by prioritized
fictitious self-play (PFSP).

Why: continuing domibot2.2 for 2000 more iterations of pure self-play, and
training a 5x larger network, both landed at exactly 2.2's strength
(training/README.md, Phase 2). Self-play only ever asks "how do I beat my
current self?", so once the network is its own best response it stops
moving -- while it still loses ~23% of games to Big Money + a terminal.
The earlier opponent pool (`--opponent-pool-*`) only drew this run's own
recent snapshots, too close to the current network to change that. This
league mixes in genuinely different play: checkpoints from across the
training history, a different architecture, the MCTS lineage's network,
and scripted Big Money strategies.

PFSP (as in AlphaStar): opponent i is drawn with weight (1 - p_i)^power,
where p_i is the learner's recent score against it (wins + half of ties,
Laplace-smoothed, decayed every iteration so it tracks the current
learner), mixed with a uniform floor so easy opponents still show up.
Only the learner's own seat produces training data; opponents' moves just
advance the game (same rule as `rollout.collect_cross_play_rollouts`).
Network opponents play their own sampled policy; scripted ones their
`act(game)`.
"""
from __future__ import annotations

import copy
import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .. import encoding
from ..agents import BigMoneyAgent, BigMoneyTerminalAgent
from ..env import DominionEnv
from ..mcts import terminal_value
from ..network import DomibotNet
from ..self_play import DEFAULT_MAX_MOVES, _sample_kingdom
from .gae import Transition, compute_gae
from .rollout import _step_group

SCRIPTED_OPPONENTS = {"bigmoney": BigMoneyAgent, "bigmoney_terminal": BigMoneyTerminalAgent}


@dataclass(eq=False)  # compared and hashed by identity: two opponents can share a network
class Opponent:
    name: str
    network: Optional[torch.nn.Module] = None  # a frozen network, or
    agent: object = None  # a scripted Agent: act(game) -> Action
    snapshot: bool = False  # a copy of the learner taken during this run
    # The learner's recent results against it: wins + half of ties, and
    # games, both decayed every iteration.
    score: float = 0.0
    games: float = 0.0

    def learner_score(self) -> float:
        return (self.score + 1.0) / (self.games + 2.0)  # 0.5 before any games


class League:
    def __init__(self, opponents: list[Opponent], hard_power: float = 2.0, uniform_mix: float = 0.2,
                 decay: float = 0.95, max_snapshots: int = 4):
        if not opponents:
            raise ValueError("a league needs at least one opponent")
        self.opponents = list(opponents)
        self.hard_power = hard_power
        self.uniform_mix = uniform_mix
        self.decay = decay
        self.max_snapshots = max_snapshots

    def weights(self) -> list[float]:
        hard = [(1.0 - o.learner_score()) ** self.hard_power for o in self.opponents]
        total = sum(hard) or 1.0
        n = len(self.opponents)
        return [(1 - self.uniform_mix) * h / total + self.uniform_mix / n for h in hard]

    def sample(self, k: int, rng: random.Random) -> list[Opponent]:
        return rng.choices(self.opponents, weights=self.weights(), k=k)

    def record(self, opponent: Opponent, results: list[float]) -> None:
        opponent.score += sum(results)
        opponent.games += len(results)

    def end_iteration(self) -> None:
        for o in self.opponents:
            o.score *= self.decay
            o.games *= self.decay

    def add_snapshot(self, network: torch.nn.Module, name: str) -> None:
        frozen = copy.deepcopy(network)
        frozen.eval()
        self.opponents.append(Opponent(name, network=frozen, snapshot=True))
        snapshots = [o for o in self.opponents if o.snapshot]
        for old in snapshots[: max(len(snapshots) - self.max_snapshots, 0)]:
            self.opponents.remove(old)

    def summary(self) -> str:
        rows = sorted(zip(self.weights(), self.opponents), key=lambda r: -r[0])
        return "  ".join(f"{o.name}:{o.learner_score():.0%}/w{w:.2f}" for w, o in rows)


def load_league(checkpoints: list[str], scripted: list[str], device: torch.device, **kwargs) -> League:
    """`checkpoints` are paths or glob patterns of DomibotNet checkpoints
    (any size, with or without the public-feature inputs); `scripted` are
    keys of SCRIPTED_OPPONENTS."""
    opponents: list[Opponent] = []
    for pattern in checkpoints:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"league checkpoint not found: {path}")
            net = DomibotNet.load(path, map_location=device).to(device)
            net.eval()
            opponents.append(Opponent(Path(path).stem, network=net))
    for key in scripted:
        opponents.append(Opponent(key, agent=SCRIPTED_OPPONENTS[key]()))
    return League(opponents, **kwargs)


def _step_scripted(idxs: list[int], agent, envs: list[DominionEnv], obs_list: list[np.ndarray],
                   mask_list: list[np.ndarray], active: list[bool], game_over: list[bool]) -> None:
    """`_step_group` for a scripted agent: one `act(game)` per game."""
    for i in idxs:
        env = envs[i]
        action_idx = encoding.action_to_index(agent.act(env.game))
        obs, _reward, terminated, truncated, _info = env.step(action_idx)
        if terminated or truncated:
            active[i] = False
            game_over[i] = terminated
        else:
            obs_list[i] = obs["observation"]
            mask_list[i] = obs["action_mask"]


def collect_league_rollouts(
    network: torch.nn.Module,
    opponents: list[Opponent],
    max_moves: int = DEFAULT_MAX_MOVES,
    device: Optional[torch.device] = None,
    seed: int | None = None,
    min_sub_decision_cards: int = 0,
    reward_fn: Callable = terminal_value,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[list[list[Transition]], list[float]]:
    """One 2-player game per entry of `opponents`, the learner (`network`)
    in a random seat. Returns each game's learner-only `Transition`s (GAE
    filled in) and the learner's result in it: 1 win, 0.5 tie (or a game
    cut off at `max_moves`), 0 loss."""
    if device is None:
        device = next(network.parameters()).device
    master_rng = random.Random(seed)
    # Every network reads a prefix of the full encoding (DomibotNet.forward),
    # so one encoding serves learner and opponents alike.
    full_obs = any(getattr(net, "extra_dim", 0) for net in [network] + [o.network for o in opponents if o.network])

    envs: list[DominionEnv] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    seat: list[int] = []
    for _ in opponents:
        g_seed = master_rng.randrange(2**31)
        env = DominionEnv(num_players=2, max_steps=max_moves, reward_fn=reward_fn, full_obs=full_obs)
        obs, _info = env.reset(kingdom=_sample_kingdom(random.Random(g_seed), min_sub_decision_cards), seed=g_seed)
        envs.append(env)
        obs_list.append(obs["observation"])
        mask_list.append(obs["action_mask"])
        seat.append(master_rng.randrange(2))

    n = len(opponents)
    transitions_per_game: list[list[Transition]] = [[] for _ in range(n)]
    game_over = [False] * n
    active = [True] * n
    while any(active):
        idxs = [i for i in range(n) if active[i]]
        own = [i for i in idxs if envs[i].game.current_decider() == seat[i]]
        theirs = [i for i in idxs if envs[i].game.current_decider() != seat[i]]
        _step_group(own, network, envs, obs_list, mask_list, transitions_per_game, active, game_over, device,
                    record=True)
        by_opponent: dict[int, list[int]] = {}
        for i in theirs:
            by_opponent.setdefault(id(opponents[i]), []).append(i)
        for group in by_opponent.values():
            opponent = opponents[group[0]]
            if opponent.network is not None:
                _step_group(group, opponent.network, envs, obs_list, mask_list, transitions_per_game, active,
                            game_over, device, record=False)
            else:
                _step_scripted(group, opponent.agent, envs, obs_list, mask_list, active, game_over)

    results: list[float] = []
    for i in range(n):
        compute_gae(transitions_per_game[i], envs[i].game, game_over[i], gamma=gamma, lam=lam, reward_fn=reward_fn)
        winners = envs[i].game.winners() if game_over[i] else []
        results.append(0.5 if not game_over[i] or len(winners) != 1 else float(winners[0] == seat[i]))
    return transitions_per_game, results
