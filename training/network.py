"""The policy/value network: given an encoded observation, predicts a
distribution over the 206-action vocabulary (policy) and the expected game
outcome from the deciding player's perspective (value, in [-1, 1]).

Small residual MLP — the observation is already an engineered fixed-size
feature vector (see encoding.py), not raw pixels/a board grid, so there's
no reason for anything convolutional. At this size the network itself is
not the compute bottleneck of self-play; the Python game engine driving
MCTS simulations is. See train.py's docstring for what that means for GPU
usage.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoding

HIDDEN_DIM = 256
NUM_RESIDUAL_BLOCKS = 4


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class DomibotNet(nn.Module):
    def __init__(self, obs_dim: int = encoding.OBS_DIM, num_actions: int = encoding.NUM_ACTIONS,
                 hidden_dim: int = HIDDEN_DIM, num_blocks: int = NUM_RESIDUAL_BLOCKS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.input = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(_ResidualBlock(hidden_dim) for _ in range(num_blocks))
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (batch, obs_dim). Returns (policy_logits: (batch, num_actions),
        value: (batch,)), both raw — masking/softmax happens in the caller,
        since only the caller knows which actions are legal right now."""
        h = F.relu(self.input(obs))
        for block in self.blocks:
            h = block(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "num_actions": self.num_actions}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> "DomibotNet":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        net = cls(obs_dim=checkpoint["obs_dim"], num_actions=checkpoint["num_actions"])
        net.load_state_dict(checkpoint["state_dict"])
        return net
