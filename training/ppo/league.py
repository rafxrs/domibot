"""An opponent league for PPO, sampled by prioritized fictitious self-play (PFSP).

Part of each iteration's games are played against fixed opponents (past
checkpoints, other networks, scripted bots) instead of the learner itself.
Opponent i is drawn with weight (1 - learner's recent score vs it) ** power,
mixed with a uniform floor. Only the learner's seat produces training data.
"""
from __future__ import annotations

import copy
import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from ..env import MAX_MOVES, random_kingdom, terminal_value
from ..network import DomibotNet
from ..strategy_bots import SCRIPTED
from .gae import Transition
from .rollout import GameBatch


@dataclass(eq=False)  # hashed by identity: two opponents can share a network
class Opponent:
    name: str
    network: Optional[torch.nn.Module] = None  # a frozen network, or
    agent: object = None  # a scripted agent
    snapshot: bool = False  # a copy of the learner taken during this run
    home: tuple[str, ...] = ()  # cards every kingdom it plays on must hold
    score: float = 0.0  # the learner's recent wins + half its ties against it, decayed
    games: float = 0.0

    def learner_score(self) -> float:
        return (self.score + 1.0) / (self.games + 2.0)


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
        return [(1 - self.uniform_mix) * h / total + self.uniform_mix / len(hard) for h in hard]

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
        for old in snapshots[:max(len(snapshots) - self.max_snapshots, 0)]:
            self.opponents.remove(old)

    def summary(self) -> str:
        rows = sorted(zip(self.weights(), self.opponents), key=lambda r: -r[0])
        return "  ".join(f"{o.name}:{o.learner_score():.0%}/w{w:.2f}" for w, o in rows)


def load_league(checkpoints: list[str], scripted: list[str], device: torch.device, **kwargs) -> League:
    """`checkpoints`: paths or glob patterns; `scripted`: keys of `strategy_bots.SCRIPTED`."""
    opponents = []
    for pattern in checkpoints:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not Path(path).exists():
                raise FileNotFoundError(f"league checkpoint not found: {path}")
            net = DomibotNet.load(path, map_location=device).to(device)
            net.eval()
            opponents.append(Opponent(Path(path).stem, network=net))
    for key in scripted:
        cls = SCRIPTED[key]
        opponents.append(Opponent(key, agent=cls(), home=tuple(getattr(cls, "HOME", ()))))
    return League(opponents, **kwargs)


def collect_league_rollouts(network: torch.nn.Module, opponents: list[Opponent], max_moves: int = MAX_MOVES,
                            device: Optional[torch.device] = None, seed: int | None = None,
                            reward_fn: Callable = terminal_value, gamma: float = 1.0,
                            lam: float = 0.95) -> tuple[list[list[Transition]], list[float]]:
    """One game per entry of `opponents`, the learner in a random seat, on a kingdom
    holding the opponent's `home` cards. Returns the learner's transitions per game
    and its results (1 win, 0.5 tie or cut off, 0 loss)."""
    device = device or next(network.parameters()).device
    master_rng = random.Random(seed)
    full_obs = any(getattr(net, "extra_dim", 0) for net in [network] + [o.network for o in opponents if o.network])
    boards, seat = [], []
    for opponent in opponents:
        g_seed = master_rng.randrange(2**31)
        boards.append((random_kingdom(random.Random(g_seed), opponent.home), g_seed))
        seat.append(master_rng.randrange(2))
    batch = GameBatch(boards, max_moves, reward_fn, full_obs)
    while idxs := batch.running():
        mine = [i for i in idxs if batch.envs[i].game.current_decider() == seat[i]]
        theirs = [i for i in idxs if batch.envs[i].game.current_decider() != seat[i]]
        batch.step_network(mine, network, device)
        by_opponent: dict[int, list[int]] = {}
        for i in theirs:
            by_opponent.setdefault(id(opponents[i]), []).append(i)
        for group in by_opponent.values():
            opponent = opponents[group[0]]
            if opponent.network is not None:
                batch.step_network(group, opponent.network, device, record=False)
            else:
                batch.step_agent(group, opponent.agent)
    transitions = batch.finish(gamma, lam)
    results = []
    for i, env in enumerate(batch.envs):
        winners = env.game.winners() if batch.game_over[i] else []
        results.append(0.5 if len(winners) != 1 else float(winners[0] == seat[i]))
    return transitions, results
