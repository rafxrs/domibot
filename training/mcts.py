"""PUCT search (as in AlphaZero) over every decision, card-effect choices included.

Card effects are suspended generators that `Game.clone()` can't copy, so a node's
position is a `(boundary, path)` pair: the nearest ancestor with no effect
pending, plus the choices made since. `materialize` replays the path on a clone;
every random draw goes through `game.rng`, so the replay is exact. Each node's
values are from its decider's point of view, negated only where the decider
changes (a turn can hold many decisions, and an attack can make the opponent
decide mid-turn).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from domibot import Action, Game

from . import encoding
from .env import terminal_value

C_PUCT = 1.5


class MCTSNode:
    def __init__(self, game: Game, boundary: Game, path: list[Action]):
        self.game, self.boundary, self.path = game, boundary, path
        self.is_terminal = game.is_game_over()
        self.decider: Optional[int] = None if self.is_terminal else game.current_decider()
        self.legal_actions: list[Action] = [] if self.is_terminal else game.legal_actions()
        self.children: dict[Action, MCTSNode] = {}
        self.P: dict[Action, float] = {}
        self.N: dict[Action, int] = {a: 0 for a in self.legal_actions}
        self.W: dict[Action, float] = {a: 0.0 for a in self.legal_actions}


def materialize(boundary: Game, path: list[Action]) -> Game:
    """A clone of `boundary` (no effect pending) with `path` replayed on it."""
    game = boundary.clone()
    for a in path:
        game.step(a)
    return game


def _make_node(boundary: Game, path: list[Action]) -> MCTSNode:
    game = materialize(boundary, path)
    if game.pending_decision is None:  # back on a boundary: start a fresh path from here
        return MCTSNode(game, game, [])
    return MCTSNode(game, boundary, path)


def _expand(nodes: list[MCTSNode], network: torch.nn.Module, device: torch.device) -> list[float]:
    """One batched forward pass: sets each node's priors and returns its value."""
    obs = np.stack([encoding.encode_for(network, n.game, n.decider) for n in nodes])
    with torch.no_grad():
        logits, values = network(torch.from_numpy(obs).to(device))
    for node, row in zip(nodes, logits.cpu().numpy()):
        row = np.where(encoding.legal_action_mask(node.game), row, -1e9)
        probs = np.exp(row - row.max())
        probs = probs / probs.sum()
        node.P = {a: float(probs[encoding.ACTION_INDEX[a]]) for a in node.legal_actions}
    return [float(v) for v in values.cpu().numpy()]


def _backup(path: list[tuple[MCTSNode, Action, MCTSNode]], leaf_value: Optional[float]) -> None:
    """Back a leaf's value up `path`; `leaf_value=None` means the leaf is terminal."""
    if leaf_value is None:
        node, action, leaf = path[-1]
        v = terminal_value(leaf.game, node.decider)
        node.N[action] += 1
        node.W[action] += v
        decider, path = node.decider, path[:-1]
    else:
        v, decider = leaf_value, path[-1][2].decider
    for node, action, _child in reversed(path):
        v = v if decider == node.decider else -v
        node.N[action] += 1
        node.W[action] += v
        decider = node.decider


def _puct_select(node: MCTSNode, c_puct: float) -> Action:
    sqrt_total = math.sqrt(sum(node.N.values()) + 1e-8)
    return max(node.legal_actions, key=lambda a: (node.W[a] / node.N[a] if node.N[a] else 0.0)
               + c_puct * node.P[a] * sqrt_total / (1 + node.N[a]))


def run_mcts_batch(root_games: list[Game], network: torch.nn.Module, num_simulations: int, c_puct: float = C_PUCT,
                   device: Optional[torch.device] = None,
                   paths: Optional[list[list[Action]]] = None) -> list[MCTSNode]:
    """Independent searches side by side, sharing one forward pass per simulation round.

    Each root game must be a boundary (no effect pending); search a mid-effect
    position by passing the choices made since its boundary in `paths`.
    """
    if any(g.pending_decision is not None or g.is_game_over() for g in root_games):
        raise ValueError("root games must be non-terminal boundaries; pass mid-effect choices via `paths`")
    if not root_games:
        return []
    device = device or next(network.parameters()).device
    roots = [_make_node(g, p) for g, p in zip(root_games, paths or [[] for _ in root_games])]
    if any(r.is_terminal for r in roots):
        raise ValueError("cannot search a finished game")
    _expand(roots, network, device)
    for _ in range(num_simulations):
        pending = []
        for node in roots:
            path = []
            while True:
                action = _puct_select(node, c_puct)
                child = node.children.get(action)
                new = child is None
                if new:
                    child = node.children[action] = _make_node(node.boundary, node.path + [action])
                path.append((node, action, child))
                if child.is_terminal:
                    _backup(path, None)
                    break
                if new:
                    pending.append(path)
                    break
                node = child
        if pending:
            for path, value in zip(pending, _expand([p[-1][2] for p in pending], network, device)):
                _backup(path, value)
    return roots


def run_mcts(root_game: Game, network: torch.nn.Module, num_simulations: int, c_puct: float = C_PUCT,
             device: Optional[torch.device] = None, path: Optional[list[Action]] = None) -> MCTSNode:
    """Search the position `path` reaches from `root_game`; returns the root node."""
    return run_mcts_batch([root_game], network, num_simulations, c_puct, device, [path or []])[0]


def visit_distribution(root: MCTSNode) -> dict[Action, float]:
    total = sum(root.N.values())
    if total == 0:
        return {a: 1.0 / len(root.legal_actions) for a in root.legal_actions}
    return {a: root.N[a] / total for a in root.legal_actions}


def select_action(root: MCTSNode, temperature: float, rng: Optional[np.random.Generator] = None) -> Action:
    """The most-visited action, or with `temperature` > 0 a sample weighted by visits ** (1/temperature)."""
    actions = root.legal_actions
    counts = np.array([root.N[a] for a in actions], dtype=np.float64)
    if temperature <= 1e-3:
        return actions[int(np.argmax(counts))]
    weights = counts ** (1.0 / temperature)
    return actions[(rng or np.random.default_rng()).choice(len(actions), p=weights / weights.sum())]
