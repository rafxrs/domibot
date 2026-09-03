"""PUCT-style MCTS (as in AlphaZero) over Dominion's phase-action decisions only: what to play, what to buy, when to end a phase.

Everything a card effect forces on a player mid-resolution (Chapel's trash
choices, Militia's forced discard, a Moat reveal, ...) is resolved
immediately by the fixed heuristic in heuristics.py rather than searched.
Two reasons, one of them a hard constraint:

1. (Hard constraint) Card effects are implemented as Python generators
   (see effects.py) that capture a live reference to the `Game` object
   they were created against. `Game.clone()` refuses to clone while one is
   suspended (`pending_gen is not None`) — cloning it anyway would leave
   the resumed generator silently mutating the wrong object. MCTS nodes
   are only ever created at points where cloning is safe, i.e. exactly the
   phase-action decision points.
2. (Scope) The strategic weight of a Dominion turn is overwhelmingly in
   what to play and what to buy; treating every discard/trash micro-choice
   as its own searched decision would blow up branching factor for
   comparatively little gain in a first version. A future version could
   extend search to specific high-value sub-decisions if it turns out to
   matter.

Each MCTS node therefore corresponds to a `Game` state at a phase-action
boundary. Expanding an untried action clones the current node's game,
applies the action, and fast-forwards through any forced sub-decisions
(`advance_to_next_phase_action`) to land on the next such boundary — that
result, not the raw one-decision-later state, is the child.

Backup: every node's stored Q/N/W is from the perspective of whoever
decides at that node (`node.decider`). Dominion turns often keep the same
player deciding across many consecutive phase-action nodes (e.g. an entire
Action phase), so — unlike strictly-alternating 2-player games — a value is
only negated when propagating across an actual change of decider between a
node and its child, not on every single edge.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from domibot import Action, Game

from . import encoding
from .heuristics import advance_to_next_phase_action

C_PUCT = 1.5


class MCTSNode:
    def __init__(self, game: Game):
        self.game = game
        self.is_terminal = game.is_game_over()
        self.decider: Optional[int] = None if self.is_terminal else game.current_decider()
        self.legal_actions: list[Action] = [] if self.is_terminal else game.legal_actions()
        self.children: dict[Action, "MCTSNode"] = {}
        self.P: dict[Action, float] = {}
        self.N: dict[Action, int] = {a: 0 for a in self.legal_actions}
        self.W: dict[Action, float] = {a: 0.0 for a in self.legal_actions}
        self.expanded = False


def _terminal_value(game: Game, perspective: int) -> float:
    winners = game.winners()
    if len(winners) != 1:
        return 0.0
    return 1.0 if winners[0] == perspective else -1.0


def evaluate_node(node: MCTSNode, network: torch.nn.Module, device: torch.device) -> float:
    """One network forward pass on `node.game` from `node.decider`'s point
    of view: sets `node.P` (masked, normalized priors over legal actions)
    and marks the node expanded. Returns the value estimate, also from
    `node.decider`'s perspective. Only valid on a non-terminal node."""
    obs = encoding.encode_observation(node.game, node.decider)
    mask = encoding.legal_action_mask(node.game)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = network(obs_t)

    logits = policy_logits[0].cpu().numpy()
    logits = np.where(mask, logits, -1e9)
    logits = logits - logits.max()
    probs = np.exp(logits)
    total = probs.sum()
    probs = probs / total

    node.P = {a: float(probs[encoding.ACTION_INDEX[a]]) for a in node.legal_actions}
    node.expanded = True
    return float(value.item())


def _puct_select(node: MCTSNode, c_puct: float) -> Action:
    sqrt_total = math.sqrt(sum(node.N.values()) + 1e-8)
    best_action, best_score = None, -float("inf")
    for a in node.legal_actions:
        q = node.W[a] / node.N[a] if node.N[a] > 0 else 0.0
        u = c_puct * node.P[a] * sqrt_total / (1 + node.N[a])
        score = q + u
        if score > best_score:
            best_score, best_action = score, a
    return best_action


def _create_child(node: MCTSNode, action: Action) -> MCTSNode:
    clone = node.game.clone()
    advance_to_next_phase_action(clone, action)
    return MCTSNode(clone)


def _simulate(node: MCTSNode, network: torch.nn.Module, device: torch.device, c_puct: float) -> float:
    if not node.expanded:
        return evaluate_node(node, network, device)

    action = _puct_select(node, c_puct)
    child = node.children.get(action)
    is_new_child = child is None
    if is_new_child:
        child = _create_child(node, action)
        node.children[action] = child

    if child.is_terminal:
        value_for_node = _terminal_value(child.game, node.decider)
    else:
        child_value = evaluate_node(child, network, device) if is_new_child else _simulate(child, network, device, c_puct)
        value_for_node = child_value if child.decider == node.decider else -child_value

    node.N[action] += 1
    node.W[action] += value_for_node
    return value_for_node


def run_mcts(
    root_game: Game,
    network: torch.nn.Module,
    num_simulations: int,
    c_puct: float = C_PUCT,
    device: Optional[torch.device] = None,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> MCTSNode:
    """Runs `num_simulations` simulations from a clone of `root_game`
    (never mutates `root_game` itself) and returns the root node; its `.N`
    is the visit-count distribution used both to pick a move and as the
    policy training target. `root_game` must be at a phase-action boundary
    (`pending_decision is None`) and not already over.

    `add_noise` mixes Dirichlet noise into the root priors (standard
    AlphaZero self-play exploration) — leave off for evaluation/play."""
    if root_game.pending_decision is not None or root_game.is_game_over():
        raise ValueError("run_mcts requires a non-terminal phase-action decision point")
    if device is None:
        device = next(network.parameters()).device
    rng = rng or np.random.default_rng()

    root = MCTSNode(root_game.clone())
    evaluate_node(root, network, device)
    if add_noise and root.legal_actions:
        noise = rng.dirichlet([dirichlet_alpha] * len(root.legal_actions))
        for a, n in zip(root.legal_actions, noise):
            root.P[a] = (1 - dirichlet_epsilon) * root.P[a] + dirichlet_epsilon * float(n)

    for _ in range(num_simulations):
        _simulate(root, network, device, c_puct)
    return root


def visit_distribution(root: MCTSNode) -> dict[Action, float]:
    total = sum(root.N.values())
    if total == 0:
        return {a: 1.0 / len(root.legal_actions) for a in root.legal_actions}
    return {a: root.N[a] / total for a in root.legal_actions}


def select_action(root: MCTSNode, temperature: float, rng: Optional[np.random.Generator] = None) -> Action:
    """temperature <= 0 picks the most-visited action; otherwise samples
    proportionally to visit_count ** (1/temperature) (standard AlphaZero
    self-play move selection: high temperature early for exploration, ~0
    later for strength)."""
    rng = rng or np.random.default_rng()
    actions = root.legal_actions
    counts = np.array([root.N[a] for a in actions], dtype=np.float64)
    if temperature <= 1e-3:
        return actions[int(np.argmax(counts))]
    weights = counts ** (1.0 / temperature)
    probs = weights / weights.sum()
    return actions[rng.choice(len(actions), p=probs)]
