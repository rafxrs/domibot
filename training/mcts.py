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
from domibot.models import END_ACTIONS

from . import encoding
from .heuristics import advance_to_next_phase_action

C_PUCT = 1.5

# Squashes a final-score margin (this player's score minus the average of
# everyone else's) into (-1, 1) for use as a value target/terminal backup.
# 20 VP is roughly "a solid, clear win" by the games seen so far (a few
# Provinces' worth of margin) without being an extreme blowout, so it sits
# at tanh(1) ~= 0.76 rather than saturating near +-1 immediately -- a bare
# win and a blowout should still be distinguishable in the training signal.
MARGIN_SCALE = 20.0


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


def terminal_value(game: Game, perspective: int) -> float:
    """Margin-based outcome for a finished game, from `perspective`'s point
    of view: `perspective`'s score minus the average of every other
    player's, squashed through tanh. A 1-point nail-biter and a 40-point
    blowout both used to train identically as "+1" under plain win/loss;
    this keeps that distinction, which matters when the whole point is to
    learn that some strategies (e.g. an actual engine) win *more
    decisively* than others, not just win."""
    scores = game.get_scores()
    others = [s for p, s in scores.items() if p != perspective]
    margin = scores[perspective] - (sum(others) / len(others))
    return math.tanh(margin / MARGIN_SCALE)


def _apply_action_continuation_bias(node: "MCTSNode", strength: float) -> None:
    """Self-play-only nudge: shift a fraction of END_ACTIONS' prior mass
    onto playing another Action card, whenever the player still has actions
    to spend and an Action card in hand to spend them on (exactly the
    situation where `PLAY(...)` options and `END_ACTIONS` are both legal --
    `PLAY` never appears in the Buy phase since treasures auto-play).

    Without this, a multi-card 'engine' turn needs an independent lucky
    Dirichlet-noise roll at *every* step to ever get tried, so its
    probability collapses fast with chain length. Unlike Dirichlet noise
    (root only), this applies at every node reached during search, so a
    multi-step chain actually gets attempted often enough during self-play
    for the value network to learn whether it pays off, instead of it being
    a rare accident that never accumulates real evidence either way."""
    play_actions = [a for a in node.legal_actions if a.verb == "PLAY"]
    if not play_actions or END_ACTIONS not in node.P:
        return
    shift = node.P[END_ACTIONS] * strength
    node.P[END_ACTIONS] -= shift
    bonus = shift / len(play_actions)
    for a in play_actions:
        node.P[a] += bonus


def evaluate_node(
    node: MCTSNode, network: torch.nn.Module, device: torch.device, action_bias: float = 0.0
) -> float:
    """One network forward pass on `node.game` from `node.decider`'s point
    of view: sets `node.P` (masked, normalized priors over legal actions)
    and marks the node expanded. Returns the value estimate, also from
    `node.decider`'s perspective. Only valid on a non-terminal node.

    `action_bias` > 0 applies `_apply_action_continuation_bias`; leave at 0
    for evaluation/play so what you're measuring is the network's own
    judgment, not an artificially nudged one."""
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
    if action_bias > 0:
        _apply_action_continuation_bias(node, action_bias)
    node.expanded = True
    return float(value.item())


def evaluate_nodes_batch(
    nodes: list[MCTSNode], network: torch.nn.Module, device: torch.device, action_bias: float = 0.0
) -> list[float]:
    """Batched `evaluate_node`: one forward pass for all given nodes instead
    of one pass per node. Sets `.P`/`.expanded` on each node exactly like
    `evaluate_node` and returns their value estimates in the same order.
    This is the whole point of root-parallel self-play: instead of playing
    one game at a time and paying a batch-size-1 forward pass per
    simulation, many independent games' trees are advanced in lockstep so
    each simulation round costs one batch-size-G forward pass."""
    if not nodes:
        return []
    obs_batch = np.stack([encoding.encode_observation(n.game, n.decider) for n in nodes])
    obs_t = torch.from_numpy(obs_batch).to(device)
    with torch.no_grad():
        policy_logits, values = network(obs_t)
    policy_logits = policy_logits.cpu().numpy()
    values = values.cpu().numpy()

    for i, node in enumerate(nodes):
        mask = encoding.legal_action_mask(node.game)
        logits = np.where(mask, policy_logits[i], -1e9)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        node.P = {a: float(probs[encoding.ACTION_INDEX[a]]) for a in node.legal_actions}
        if action_bias > 0:
            _apply_action_continuation_bias(node, action_bias)
        node.expanded = True
    return [float(v) for v in values]


def _backup(path: list[tuple[MCTSNode, Action, MCTSNode]], leaf_is_terminal: bool, leaf_value: float) -> None:
    """Iterative equivalent of `_simulate`'s recursive backup, applied to a
    full root-to-leaf path collected during selection. `leaf_value` is the
    terminal margin (if `leaf_is_terminal`) or the network's value estimate
    for the leaf, from the leaf's own perspective (the leaf's `.decider` for
    a non-terminal leaf; terminal nodes have no decider, so the terminal
    case instead computes its value directly from the path's last real
    decider -- see the `terminal_value` call below). Each step up negates
    the running value exactly when the decider changes between a node and
    its child, matching `_simulate`'s `value_for_node = ... if child.decider
    == node.decider else -...` rule level by level."""
    if leaf_is_terminal:
        parent_node, parent_action, terminal_child = path[-1]
        v = terminal_value(terminal_child.game, parent_node.decider)
        parent_node.N[parent_action] += 1
        parent_node.W[parent_action] += v
        current_decider = parent_node.decider
        rest = path[:-1]
    else:
        v = leaf_value
        current_decider = path[-1][2].decider
        rest = path

    for node, action, child in reversed(rest):
        v = v if current_decider == node.decider else -v
        node.N[action] += 1
        node.W[action] += v
        current_decider = node.decider


def run_mcts_batch(
    root_games: list[Game],
    network: torch.nn.Module,
    num_simulations: int,
    c_puct: float = C_PUCT,
    device: Optional[torch.device] = None,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    action_bias: float = 0.0,
    rngs: Optional[list[np.random.Generator]] = None,
) -> list[MCTSNode]:
    """Root-parallel version of `run_mcts`: runs independent searches over
    `len(root_games)` games side by side, one simulation round at a time, so
    that every round's leaf evaluations across all of them are combined into
    a single batched network forward pass (see `evaluate_nodes_batch`)
    instead of one call per game per simulation. Each game's tree is
    otherwise completely independent -- this is not tree parallelism within
    one search, just sharing the GPU call across many simultaneous searches.

    Returns one root `MCTSNode` per game, in the same order as
    `root_games`, each with the same `.N`/`.P` semantics as `run_mcts`'s
    single root."""
    for g in root_games:
        if g.pending_decision is not None or g.is_game_over():
            raise ValueError("run_mcts_batch requires non-terminal phase-action decision points")
    if not root_games:
        return []
    if device is None:
        device = next(network.parameters()).device
    if rngs is None:
        rngs = [np.random.default_rng() for _ in root_games]

    roots = [MCTSNode(g.clone()) for g in root_games]
    evaluate_nodes_batch(roots, network, device, action_bias)
    if add_noise:
        for root, rng in zip(roots, rngs):
            if root.legal_actions:
                noise = rng.dirichlet([dirichlet_alpha] * len(root.legal_actions))
                for a, n in zip(root.legal_actions, noise):
                    root.P[a] = (1 - dirichlet_epsilon) * root.P[a] + dirichlet_epsilon * float(n)

    for _ in range(num_simulations):
        pending: list[tuple[list[tuple[MCTSNode, Action, MCTSNode]], MCTSNode]] = []
        for root in roots:
            node = root
            path: list[tuple[MCTSNode, Action, MCTSNode]] = []
            while True:
                action = _puct_select(node, c_puct)
                child = node.children.get(action)
                is_new_child = child is None
                if is_new_child:
                    child = _create_child(node, action)
                    node.children[action] = child
                path.append((node, action, child))
                if child.is_terminal:
                    _backup(path, leaf_is_terminal=True, leaf_value=0.0)
                    break
                if is_new_child:
                    pending.append((path, child))
                    break
                node = child

        if pending:
            leaf_nodes = [leaf for _, leaf in pending]
            values = evaluate_nodes_batch(leaf_nodes, network, device, action_bias)
            for (path, _leaf), value in zip(pending, values):
                _backup(path, leaf_is_terminal=False, leaf_value=value)

    return roots


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


def _simulate(
    node: MCTSNode, network: torch.nn.Module, device: torch.device, c_puct: float, action_bias: float = 0.0
) -> float:
    if not node.expanded:
        return evaluate_node(node, network, device, action_bias)

    action = _puct_select(node, c_puct)
    child = node.children.get(action)
    is_new_child = child is None
    if is_new_child:
        child = _create_child(node, action)
        node.children[action] = child

    if child.is_terminal:
        value_for_node = terminal_value(child.game, node.decider)
    else:
        child_value = (
            evaluate_node(child, network, device, action_bias)
            if is_new_child
            else _simulate(child, network, device, c_puct, action_bias)
        )
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
    action_bias: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> MCTSNode:
    """Runs `num_simulations` simulations from a clone of `root_game`
    (never mutates `root_game` itself) and returns the root node; its `.N`
    is the visit-count distribution used both to pick a move and as the
    policy training target. `root_game` must be at a phase-action boundary
    (`pending_decision is None`) and not already over.

    `add_noise` mixes Dirichlet noise into the root priors (standard
    AlphaZero self-play exploration). `action_bias` (see
    `_apply_action_continuation_bias`) nudges every node in the tree, not
    just the root, toward continuing to play Action cards. Leave both at
    their defaults (off) for evaluation/play."""
    if root_game.pending_decision is not None or root_game.is_game_over():
        raise ValueError("run_mcts requires a non-terminal phase-action decision point")
    if device is None:
        device = next(network.parameters()).device
    rng = rng or np.random.default_rng()

    root = MCTSNode(root_game.clone())
    evaluate_node(root, network, device, action_bias)
    if add_noise and root.legal_actions:
        noise = rng.dirichlet([dirichlet_alpha] * len(root.legal_actions))
        for a, n in zip(root.legal_actions, noise):
            root.P[a] = (1 - dirichlet_epsilon) * root.P[a] + dirichlet_epsilon * float(n)

    for _ in range(num_simulations):
        _simulate(root, network, device, c_puct, action_bias)
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
