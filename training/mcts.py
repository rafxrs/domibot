"""PUCT-style MCTS (as in AlphaZero) over every Dominion decision -- phase
actions (what to play, what to buy, when to end a phase) *and* card-effect
sub-decisions (Chapel's trash choices, Militia's forced discard, Sentry's
trash/discard/reorder, ...), searched and backed up uniformly.

A node's position is represented as a `(boundary, path)` pair rather than a
raw `Game`: `boundary` is the nearest ancestor `Game` sitting at a true
phase-action boundary (`pending_decision is None`, always safely clonable),
and `path` is the ordered list of sub-decision `Action`s taken since that
boundary. `materialize()` reconstructs the actual position on demand by
cloning `boundary` and replaying `path` via `Game.step()`.

This sidesteps a real hard constraint without touching `Game`/`effects.py`
at all: card effects are Python generators (see effects.py) that capture a
live reference to the `Game` object they were created against, so
`Game.clone()` refuses to clone while one is suspended (`pending_gen is not
None`) -- cloning it anyway would leave the resumed generator silently
mutating the wrong object. But `Game.rng`'s state is fully, deterministically
captured by `clone()` (`getstate()`/`setstate()`), and every random draw
inside any card effect goes exclusively through `game.rng` -- so replaying
the same ordered action sequence against a fresh clone of the boundary
always reaches bit-identical state to the original, uncloned position. A
node therefore never needs to clone anything mid-effect; only `boundary`
(always a real, safe clone) ever gets cloned, and `path` stays short since
it resets to `[]` the moment a node lands back on a boundary -- bounded by a
single effect's depth (e.g. Sentry's 3 stages), not the whole game's length.

Backup: every node's stored Q/N/W is from the perspective of whoever
decides at that node (`node.decider`). Dominion turns often keep the same
player deciding across many consecutive nodes (e.g. an entire Action phase,
or a multi-stage sub-decision), and a sub-decision can belong to a different
player than the one whose turn it is (Militia's forced discard, Bureaucrat's
forced topdeck) -- so, unlike strictly-alternating 2-player games, a value
is only negated when propagating across an actual change of decider between
a node and its child, not on every single edge.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

from domibot import Action, Game
from domibot.models import END_ACTIONS

from . import encoding

C_PUCT = 1.5

# Squashes a final-score margin (this player's score minus the average of
# everyone else's) into (-1, 1) for use as a value target/terminal backup.
# A bare win and a blowout should still be distinguishable, so this stays
# well short of an immediate +-1 saturation -- but it was 20.0, and
# measured margins between competent players are median 3 VP / mean 4.8,
# which 20.0 squashed into |target| ~ 0.15. That wasted almost the whole
# output range, shrank MSE (and so the value head's share of an unweighted
# joint loss) by ~4x, and left Q too small to compete with PUCT's
# exploration term. At 10.0 the median lands near 0.29 with only ~8% of
# games saturating past 0.9.
MARGIN_SCALE = 10.0


class MCTSNode:
    def __init__(self, game: Game, boundary: Game, path: list[Action]):
        self.game = game
        # Nearest ancestor safely-clonable Game (boundary=game, path=[] for
        # a node that's itself at a boundary) and the sub-decision actions
        # since it -- see module docstring. `_create_child` clones
        # `boundary`, never `game` (which may be mid-effect).
        self.boundary = boundary
        self.path = path
        self.is_terminal = game.is_game_over()
        self.decider: Optional[int] = None if self.is_terminal else game.current_decider()
        self.legal_actions: list[Action] = [] if self.is_terminal else game.legal_actions()
        self.children: dict[Action, "MCTSNode"] = {}
        self.P: dict[Action, float] = {}
        self.N: dict[Action, int] = {a: 0 for a in self.legal_actions}
        self.W: dict[Action, float] = {a: 0.0 for a in self.legal_actions}
        self.expanded = False


def materialize(boundary: Game, path: list[Action]) -> Game:
    """Reconstruct the position reached from `boundary` (a safely-clonable
    Game, i.e. `pending_gen is None`) by replaying `path`. Never mutates
    `boundary`. Deterministic -- see module docstring."""
    game = boundary.clone()
    for a in path:
        game.step(a)
    return game


def redeal_hidden_info(game: Game, from_player: int, rng: random.Random) -> Game:
    """A clone of `game` with every *other* player's hand+deck contents
    reshuffled and re-split (sizes preserved), leaving `from_player`'s own
    hand/deck and every public zone (discard, play_area, set_aside, supply,
    trash) exactly as they truly are. Samples one plausible world consistent
    with what `from_player` can actually observe, instead of the one true
    (but to `from_player`, unknown) deal -- see `run_mcts_ensemble`.

    Only meaningful at a real phase-action boundary (`from_player`'s own
    hand must not itself be mid-effect); does not attempt to keep a
    subsequent `path` replay consistent with the redeal (a path that
    consumes specific cards from a redealt player's hand could then find
    them missing) -- callers must only use this where `path == []`."""
    clone = game.clone()
    for i, player in enumerate(clone.players):
        if i == from_player:
            continue
        pool = player.hand + player.deck
        rng.shuffle(pool)
        hand_size = len(player.hand)
        player.hand = pool[:hand_size]
        player.deck = pool[hand_size:]
    # Game.clone() copies the exact rng state, so without this every redeal
    # of the same boundary would see identical future chance events (draws,
    # reshuffles) despite having different hidden hands.
    clone.rng.seed(rng.getrandbits(64))
    return clone


def _make_node(boundary: Game, path: list[Action]) -> MCTSNode:
    """Build the node for the position reached from `boundary` by replaying
    `path`. If that position is itself a fresh boundary (including game
    over, which can only be detected at a boundary), the node becomes its
    own boundary with an empty path, so path length never grows past a
    single effect's sub-decision depth."""
    game = materialize(boundary, path)
    if game.pending_decision is None:
        return MCTSNode(game, boundary=game, path=[])
    return MCTSNode(game, boundary=boundary, path=path)


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
    paths: Optional[list[list[Action]]] = None,
) -> list[MCTSNode]:
    """Root-parallel version of `run_mcts`: runs independent searches over
    `len(root_games)` games side by side, one simulation round at a time, so
    that every round's leaf evaluations across all of them are combined into
    a single batched network forward pass (see `evaluate_nodes_batch`)
    instead of one call per game per simulation. Each game's tree is
    otherwise completely independent -- this is not tree parallelism within
    one search, just sharing the GPU call across many simultaneous searches.

    Each `root_games[i]` must be a safely-clonable phase-action boundary;
    pass sub-decision actions taken since it via `paths[i]` (default `[]`
    for every root, identical to today's behavior) rather than handing in a
    mid-effect Game directly -- see `run_mcts`.

    Returns one root `MCTSNode` per game, in the same order as
    `root_games`, each with the same `.N`/`.P` semantics as `run_mcts`'s
    single root."""
    for g in root_games:
        if g.pending_decision is not None or g.is_game_over():
            raise ValueError("run_mcts_batch requires root_games to be non-terminal phase-action "
                              "boundaries; pass sub-decision actions via `paths`, not by handing in "
                              "a mid-effect Game")
    if not root_games:
        return []
    if device is None:
        device = next(network.parameters()).device
    if rngs is None:
        rngs = [np.random.default_rng() for _ in root_games]
    paths = paths or [[] for _ in root_games]

    roots = [_make_node(g, p) for g, p in zip(root_games, paths)]
    for root in roots:
        if root.is_terminal:
            raise ValueError("run_mcts_batch requires non-terminal decision points")
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
    return _make_node(node.boundary, node.path + [action])


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
    path: Optional[list[Action]] = None,
) -> MCTSNode:
    """Runs `num_simulations` simulations from the position reached by
    replaying `path` from `root_game`, and returns the root node; its `.N`
    is the visit-count distribution used both to pick a move and as the
    policy training target. Never mutates `root_game` itself.

    `root_game` must itself be a safely-clonable phase-action boundary
    (`pending_decision is None`) and not already over -- to search a
    sub-decision (mid-effect) position, pass the boundary it descends from
    as `root_game` and the sub-decision actions taken since then as `path`
    (default `[]`: the common case of searching a plain phase-action
    decision, identical to today's behavior).

    `add_noise` mixes Dirichlet noise into the root priors (standard
    AlphaZero self-play exploration). `action_bias` (see
    `_apply_action_continuation_bias`) nudges every node in the tree, not
    just the root, toward continuing to play Action cards. Leave both at
    their defaults (off) for evaluation/play."""
    if root_game.pending_decision is not None or root_game.is_game_over():
        raise ValueError("run_mcts requires root_game to be a non-terminal phase-action boundary; "
                          "pass sub-decision actions via `path`, not by handing in a mid-effect Game")
    if device is None:
        device = next(network.parameters()).device
    rng = rng or np.random.default_rng()

    root = _make_node(root_game, path or [])
    if root.is_terminal:
        raise ValueError("run_mcts requires a non-terminal decision point")
    evaluate_node(root, network, device, action_bias)
    if add_noise and root.legal_actions:
        noise = rng.dirichlet([dirichlet_alpha] * len(root.legal_actions))
        for a, n in zip(root.legal_actions, noise):
            root.P[a] = (1 - dirichlet_epsilon) * root.P[a] + dirichlet_epsilon * float(n)

    for _ in range(num_simulations):
        _simulate(root, network, device, c_puct, action_bias)
    return root


def merge_ensemble_roots(roots: list[MCTSNode]) -> MCTSNode:
    """Combines several ensemble members' root nodes (same `legal_actions`
    by construction -- see `redeal_hidden_info`) into one, by summing visit
    counts into `roots[0]`. `visit_distribution`/`select_action` need
    nothing else -- they only ever read `.N`/`.legal_actions`."""
    merged = roots[0]
    for other in roots[1:]:
        for a in merged.legal_actions:
            merged.N[a] += other.N[a]
    return merged


def run_mcts_ensemble(
    root_game: Game,
    network: torch.nn.Module,
    num_simulations: int,
    ensemble_size: int = 1,
    c_puct: float = C_PUCT,
    device: Optional[torch.device] = None,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    action_bias: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    py_rng: Optional[random.Random] = None,
) -> MCTSNode:
    """Like `run_mcts`, but searches `ensemble_size` independently redealt
    hidden-info samples (`redeal_hidden_info`) instead of the one true deal,
    then merges their visit counts -- a multi-determinization form of PIMC
    that averages the search over several plausible opponent hands instead
    of committing the whole tree to whichever one this game actually has
    (see module docstring). `ensemble_size <= 1` delegates straight to
    `run_mcts`, a true no-op so every existing caller stays exactly
    reproducible with the default.

    Only valid at a plain phase-action boundary (same requirement as
    `run_mcts` with `path=None`) -- `redeal_hidden_info`'s docstring covers
    why this doesn't extend to sub-decision searches yet. `num_simulations`
    is split evenly across the ensemble (at least 1 each), so a bigger
    `ensemble_size` trades search depth per world for world diversity at a
    fixed total budget."""
    if ensemble_size <= 1:
        return run_mcts(
            root_game, network, num_simulations, c_puct=c_puct, device=device, add_noise=add_noise,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, action_bias=action_bias, rng=rng,
        )
    py_rng = py_rng or random.Random()
    decider = root_game.current_decider()
    redealt = [redeal_hidden_info(root_game, decider, py_rng) for _ in range(ensemble_size)]
    sims_per_member = max(1, num_simulations // ensemble_size)
    rngs = [rng if rng is not None else np.random.default_rng() for _ in redealt]
    roots = run_mcts_batch(
        redealt, network, sims_per_member, c_puct=c_puct, device=device, add_noise=add_noise,
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, action_bias=action_bias, rngs=rngs,
    )
    return merge_ensemble_roots(roots)


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
