"""`DomibotNet`: a residual MLP from an encoded observation to (policy logits, value).

Input blocks beyond the base encoding (public extras, own zones) feed the first
hidden layer through zero-initialized projections, so adding one to a trained
network leaves its outputs unchanged until trained. The size and inputs are
saved in each checkpoint.
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
    """Pre-norm residual block; the residual stream itself is left unclamped."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class DomibotNet(nn.Module):
    def __init__(self, obs_dim: int = encoding.OBS_DIM, num_actions: int = encoding.NUM_ACTIONS,
                 hidden_dim: int = HIDDEN_DIM, num_blocks: int = NUM_RESIDUAL_BLOCKS, extra_dim: int = 0,
                 zones_dim: int = 0):
        super().__init__()
        if zones_dim and not extra_dim:
            raise ValueError("the own-zone inputs follow the public extras, so they need the extras too")
        self.obs_dim, self.num_actions, self.hidden_dim, self.num_blocks = obs_dim, num_actions, hidden_dim, num_blocks
        self.extra_dim, self.zones_dim = extra_dim, zones_dim
        self.input_norm = nn.LayerNorm(obs_dim)  # raw pile counts and 0/1 one-hots differ ~46x in scale
        self.input = nn.Linear(obs_dim, hidden_dim)
        if extra_dim:
            self.extra_norm, self.extra_input = nn.LayerNorm(extra_dim), self._zero_linear(extra_dim, hidden_dim)
        if zones_dim:
            self.zones_norm, self.zones_input = nn.LayerNorm(zones_dim), self._zero_linear(zones_dim, hidden_dim)
        self.blocks = nn.ModuleList(_ResidualBlock(hidden_dim) for _ in range(num_blocks))
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                        nn.Linear(hidden_dim // 2, 1), nn.Tanh())

    @staticmethod
    def _zero_linear(n_in: int, n_out: int) -> nn.Linear:
        layer = nn.Linear(n_in, n_out)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (batch, >= the dims this network reads). Returns raw (policy logits, value);
        masking is the caller's job."""
        x = self.input(self.input_norm(obs[:, :self.obs_dim]))
        if self.extra_dim:
            x = x + self.extra_input(self.extra_norm(obs[:, self.obs_dim:self.obs_dim + self.extra_dim]))
        if self.zones_dim:
            start = self.obs_dim + self.extra_dim
            x = x + self.zones_input(self.zones_norm(obs[:, start:start + self.zones_dim]))
        h = F.relu(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def _with(self, prefix: str, **dims) -> "DomibotNet":
        net = DomibotNet(obs_dim=self.obs_dim, num_actions=self.num_actions, hidden_dim=self.hidden_dim,
                         num_blocks=self.num_blocks, **{"extra_dim": self.extra_dim, "zones_dim": self.zones_dim, **dims})
        missing, unexpected = net.load_state_dict(self.state_dict(), strict=False)
        assert not unexpected and all(k.startswith(prefix) for k in missing)
        return net.to(next(self.parameters()).device)

    def with_extra_inputs(self, extra_dim: int = encoding.EXTRA_DIM) -> "DomibotNet":
        """A copy that also reads the public extras, with unchanged outputs until trained."""
        if self.extra_dim:
            raise ValueError("network already has extra inputs")
        return self._with("extra_", extra_dim=extra_dim)

    def with_zone_inputs(self, zones_dim: int = encoding.ZONES_DIM) -> "DomibotNet":
        """A copy that also reads the own-zone inputs, with unchanged outputs until trained."""
        if self.zones_dim:
            raise ValueError("network already has own-zone inputs")
        return self._with("zones_", zones_dim=zones_dim)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "num_actions": self.num_actions,
                    "hidden_dim": self.hidden_dim, "num_blocks": self.num_blocks, "extra_dim": self.extra_dim,
                    "zones_dim": self.zones_dim}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> "DomibotNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        net = cls(obs_dim=ckpt["obs_dim"], num_actions=ckpt["num_actions"],
                  hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),  # older checkpoints are all 256 x 4
                  num_blocks=ckpt.get("num_blocks", NUM_RESIDUAL_BLOCKS),
                  extra_dim=ckpt.get("extra_dim", 0), zones_dim=ckpt.get("zones_dim", 0))
        net.load_state_dict(ckpt["state_dict"])
        return net
