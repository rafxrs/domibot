"""Generate self-play games with MCTS + a network, producing training
examples for every decision -- phase actions *and* card-effect
sub-decisions (Chapel's trash choices, Militia's forced discard, ...) --
searched uniformly: the encoded state, the legal mask, the MCTS
visit-count distribution (the policy target), and — filled in once the
game ends — the actual outcome from that decision's perspective (the value
target).

`search_sub_decisions=False` (an ablation/fallback toggle, off by default)
resolves sub-decisions with the fixed, non-learned heuristic instead
(`training.heuristics.heuristic_reaction`) -- the behavior every version
before this one used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from domibot import Action, Game, KINGDOM_CARDS

from . import encoding
from .heuristics import advance_to_next_phase_action
from .mcts import (
    materialize,
    merge_ensemble_roots,
    redeal_hidden_info,
    run_mcts,
    run_mcts_batch,
    run_mcts_ensemble,
    select_action,
    terminal_value,
    visit_distribution,
)

# Cards with a genuine trash/discard/gain/topdeck/keep-or-not judgment call
# for whoever plays them (as opposed to a plain cantrip or a no-choice
# attack like Witch). Used by `_sample_kingdom`'s curriculum mode: plain
# random 10-of-26 sampling makes several of these landing in the *same*
# kingdom a much rarer joint event than any one of them individually (each
# shows up in ~38% of kingdoms on its own), and that co-occurrence is where
# a real weakness showed up worst -- over-trashing good cards via Chapel,
# most severe in kingdoms already juggling several other demanding
# decisions at once, since credit assignment for any one of them gets
# harder the more of them are simultaneously in play.
SUB_DECISION_CARDS = frozenset({
    "Cellar", "Chapel", "Harbinger", "Workshop", "Bureaucrat", "Militia",
    "Moneylender", "Poacher", "Remodel", "Throne Room", "Bandit", "Library",
    "Mine", "Sentry", "Artisan",
})

DEFAULT_C_PUCT = 1.5
DEFAULT_TEMPERATURE_MOVES = 15  # phase-action decisions before switching to near-greedy play
# Safety cap for an undertrained/near-random policy that can stall
# indefinitely (e.g. never buying anything worth ending the game over --
# see evaluate.py's MAX_STEPS docstring for the same pathology). Every
# decision counts against this now, sub-decisions included, so it needs to
# be noticeably higher than back when only phase-actions did: measured
# directly against an early (iteration-10) checkpoint, a 400 cap made
# self-play games with sub-decision search fail to finish naturally 8/8
# times (all value targets silently defaulting to 0.0, crippling value
# learning); 1000 got most of them to a real conclusion instead, matching
# or beating the old phase-actions-only completion rate at 400.
DEFAULT_MAX_MOVES = 1000
DEFAULT_ACTION_BIAS = 0.2  # see mcts._apply_action_continuation_bias


@dataclass
class Example:
    obs: np.ndarray
    mask: np.ndarray
    policy_target: np.ndarray  # length NUM_ACTIONS, mass only on that state's legal actions
    decider: int
    value_target: float = field(default=0.0)


def _sample_kingdom(rng: random.Random, min_sub_decision_cards: int = 0) -> list[str]:
    """A plain uniform 10-of-26 kingdom by default (`min_sub_decision_cards
    <= 0`) -- identical to `rng.sample(list(KINGDOM_CARDS), 10)`, so this is
    a no-op change for every existing caller. With `min_sub_decision_cards
    > 0`, forces that many of the 10 slots to come from `SUB_DECISION_CARDS`
    (the rest filled normally), directly boosting how often several
    judgment-heavy cards land in the same kingdom together -- see
    `SUB_DECISION_CARDS`'s comment for why that specific co-occurrence is
    the actual gap worth training more on."""
    if min_sub_decision_cards <= 0:
        return rng.sample(list(KINGDOM_CARDS), 10)
    pool = list(SUB_DECISION_CARDS)
    forced = rng.sample(pool, min(min_sub_decision_cards, len(pool)))
    remaining = [c for c in KINGDOM_CARDS if c not in forced]
    kingdom = forced + rng.sample(remaining, 10 - len(forced))
    rng.shuffle(kingdom)
    return kingdom


def play_self_play_game(
    network: torch.nn.Module,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    seed: int | None = None,
    search_sub_decisions: bool = True,
    sub_decision_simulations: int | None = None,
    min_sub_decision_cards: int = 0,
    determinization_ensemble_size: int = 1,
) -> list[Example]:
    if not search_sub_decisions:
        return _play_self_play_game_phase_actions_only(
            network, num_simulations, num_players, kingdom, c_puct,
            temperature_moves, max_moves, action_bias, seed,
        )
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if kingdom is None:
        kingdom = _sample_kingdom(py_rng, min_sub_decision_cards)
    game = Game(kingdom, num_players=num_players, seed=seed)

    boundary: Game = game
    path: list[Action] = []
    examples: list[Example] = []
    move_number = 0
    while not (not path and boundary.is_game_over()) and move_number < max_moves:
        sims = num_simulations if not path else (sub_decision_simulations or num_simulations)
        # Determinization ensembles (see mcts.redeal_hidden_info) only apply
        # to plain phase-action boundaries -- a non-empty path is an
        # in-progress sub-decision, out of scope for now.
        used_ensemble = not path and determinization_ensemble_size > 1
        if used_ensemble:
            root = run_mcts_ensemble(
                boundary, network, sims, determinization_ensemble_size, c_puct=c_puct, add_noise=True,
                action_bias=action_bias, rng=np_rng, py_rng=py_rng,
            )
        else:
            root = run_mcts(
                boundary, network, sims, c_puct=c_puct, add_noise=True,
                action_bias=action_bias, rng=np_rng, path=path,
            )
        decider = root.decider
        dist = visit_distribution(root)

        policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
        for action, prob in dist.items():
            policy_target[encoding.ACTION_INDEX[action]] = prob

        examples.append(Example(
            obs=encoding.encode_observation(root.game, decider),
            mask=encoding.legal_action_mask(root.game),
            policy_target=policy_target,
            decider=decider,
        ))

        temperature = 1.0 if move_number < temperature_moves else 0.0
        action = select_action(root, temperature, rng=np_rng)
        # Reuse the child search already materialized for `action` (present
        # whenever it was actually visited, i.e. whenever N[action] > 0 --
        # true here since select_action never picks an unvisited action).
        # Never step `root.game`/`root.boundary` directly: for a root that's
        # itself a fresh boundary, `_make_node` aliases boundary=game to
        # avoid a redundant clone, so mutating one would corrupt the other.
        if used_ensemble:
            # root.boundary/.children point at a *hypothetical* redealt
            # clone, not the real game -- step the true `boundary` directly.
            child_game = materialize(boundary, [action])
            if child_game.pending_decision is None:
                boundary, path = child_game, []
            else:
                path = [action]
        else:
            child = root.children.get(action)
            child_game = child.game if child is not None else materialize(root.boundary, root.path + [action])
            if child_game.pending_decision is None:
                boundary, path = child_game, []
            else:
                boundary, path = root.boundary, root.path + [action]
        move_number += 1

    final_game = boundary if not path else materialize(boundary, path)
    game_over = not path and final_game.is_game_over()
    for ex in examples:
        # a margin-based value from each example's own decider's perspective;
        # 0.0 for the rare case of hitting max_moves without a real result
        ex.value_target = terminal_value(final_game, ex.decider) if game_over else 0.0
    return examples


def _play_self_play_game_phase_actions_only(
    network: torch.nn.Module,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    seed: int | None = None,
) -> list[Example]:
    """The original phase-actions-only self-play loop, kept verbatim as the
    `search_sub_decisions=False` fallback: every sub-decision between one
    phase-action and the next is resolved instantly by the fixed heuristic
    (`advance_to_next_phase_action`) rather than searched, so it never
    produces a training example."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if kingdom is None:
        kingdom = py_rng.sample(list(KINGDOM_CARDS), 10)
    game = Game(kingdom, num_players=num_players, seed=seed)

    examples: list[Example] = []
    move_number = 0
    while not game.is_game_over() and move_number < max_moves:
        decider = game.current_decider()
        root = run_mcts(
            game, network, num_simulations, c_puct=c_puct, add_noise=True, action_bias=action_bias, rng=np_rng
        )
        dist = visit_distribution(root)

        policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
        for action, prob in dist.items():
            policy_target[encoding.ACTION_INDEX[action]] = prob

        examples.append(Example(
            obs=encoding.encode_observation(game, decider),
            mask=encoding.legal_action_mask(game),
            policy_target=policy_target,
            decider=decider,
        ))

        temperature = 1.0 if move_number < temperature_moves else 0.0
        action = select_action(root, temperature, rng=np_rng)
        advance_to_next_phase_action(game, action)
        move_number += 1

    game_over = game.is_game_over()
    for ex in examples:
        # a margin-based value from each example's own decider's perspective;
        # 0.0 for the rare case of hitting max_moves without a real result
        ex.value_target = terminal_value(game, ex.decider) if game_over else 0.0
    return examples


def play_self_play_games_batch(
    network: torch.nn.Module,
    num_games: int,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    device: torch.device | None = None,
    seed: int | None = None,
    search_sub_decisions: bool = True,
    sub_decision_simulations: int | None = None,
    min_sub_decision_cards: int = 0,
    determinization_ensemble_size: int = 1,
) -> list[list[Example]]:
    """Root-parallel version of `play_self_play_game`: plays `num_games`
    independent games side by side, one decision at a time, so every
    decision's MCTS search across all still-active games shares a single
    batched network call via `run_mcts_batch` instead of each game paying
    for its own batch-size-1 forward passes. Produces the exact same
    per-game example sequences `play_self_play_game` would (same move
    selection, same temperature schedule, same value-target backfill) --
    this only changes how the network is called, not what gets played."""
    if not search_sub_decisions:
        return _play_self_play_games_batch_phase_actions_only(
            network, num_games, num_simulations, num_players, kingdom, c_puct,
            temperature_moves, max_moves, action_bias, device, seed,
        )
    master_rng = random.Random(seed)
    boundaries: list[Game] = []
    paths: list[list[Action]] = []
    np_rngs: list[np.random.Generator] = []
    py_rngs: list[random.Random] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        game_kingdom = kingdom if kingdom is not None else _sample_kingdom(random.Random(g_seed), min_sub_decision_cards)
        boundaries.append(Game(game_kingdom, num_players=num_players, seed=g_seed))
        paths.append([])
        np_rngs.append(np.random.default_rng(g_seed))
        py_rngs.append(random.Random(g_seed))

    examples_per_game: list[list[Example]] = [[] for _ in range(num_games)]
    move_numbers = [0] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        sims = [num_simulations if not paths[i] else (sub_decision_simulations or num_simulations) for i in idxs]
        # run_mcts_batch takes one uniform num_simulations today (per-root
        # budgets are a Stage 2 extension); until then, run each distinct
        # sim count in its own batched call -- almost always one call
        # (uniform sims) since sub_decision_simulations defaults to None.
        roots: list = [None] * len(idxs)
        # Positions searched via a redealt ensemble this round -- their
        # root.boundary/.children point at a *hypothetical* redealt clone,
        # not the real game, so advancing the real game below must bypass
        # them and step the true `boundaries[i]` directly instead.
        ensemble_positions: set[int] = set()
        for sim_count in sorted(set(sims)):
            group = [pos for pos, s in enumerate(sims) if s == sim_count]
            # Determinization ensembles (see mcts.redeal_hidden_info) only
            # apply to plain phase-action boundaries -- a non-empty path is
            # an in-progress sub-decision, out of scope for now, so those
            # positions always fall back to the plain (unensembled) call.
            ensemble_group = [pos for pos in group
                               if determinization_ensemble_size > 1 and not paths[idxs[pos]]]
            plain_group = [pos for pos in group if pos not in ensemble_group]
            ensemble_positions.update(ensemble_group)

            if plain_group:
                plain_roots = run_mcts_batch(
                    [boundaries[idxs[pos]] for pos in plain_group], network, sim_count, c_puct=c_puct,
                    add_noise=True, action_bias=action_bias, device=device,
                    rngs=[np_rngs[idxs[pos]] for pos in plain_group],
                    paths=[paths[idxs[pos]] for pos in plain_group],
                )
                for pos, root in zip(plain_group, plain_roots):
                    roots[pos] = root

            if ensemble_group:
                sims_per_member = max(1, sim_count // determinization_ensemble_size)
                flat_games, flat_rngs = [], []
                for pos in ensemble_group:
                    i = idxs[pos]
                    for _ in range(determinization_ensemble_size):
                        flat_games.append(redeal_hidden_info(
                            boundaries[i], boundaries[i].current_decider(), py_rngs[i]))
                        flat_rngs.append(np_rngs[i])
                flat_roots = run_mcts_batch(
                    flat_games, network, sims_per_member, c_puct=c_puct, add_noise=True,
                    action_bias=action_bias, device=device, rngs=flat_rngs,
                )
                for j, pos in enumerate(ensemble_group):
                    chunk = flat_roots[j * determinization_ensemble_size:(j + 1) * determinization_ensemble_size]
                    roots[pos] = merge_ensemble_roots(chunk)

        for pos, i in enumerate(idxs):
            root = roots[pos]
            decider = root.decider
            dist = visit_distribution(root)

            policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
            for action, prob in dist.items():
                policy_target[encoding.ACTION_INDEX[action]] = prob

            examples_per_game[i].append(Example(
                obs=encoding.encode_observation(root.game, decider),
                mask=encoding.legal_action_mask(root.game),
                policy_target=policy_target,
                decider=decider,
            ))

            temperature = 1.0 if move_numbers[i] < temperature_moves else 0.0
            action = select_action(root, temperature, rng=np_rngs[i])
            if pos in ensemble_positions:
                child_game = materialize(boundaries[i], [action])
                if child_game.pending_decision is None:
                    boundaries[i], paths[i] = child_game, []
                else:
                    boundaries[i], paths[i] = boundaries[i], [action]
            else:
                child = root.children.get(action)
                child_game = child.game if child is not None else materialize(root.boundary, root.path + [action])
                if child_game.pending_decision is None:
                    boundaries[i], paths[i] = child_game, []
                else:
                    boundaries[i], paths[i] = root.boundary, root.path + [action]
            move_numbers[i] += 1
            terminal = not paths[i] and boundaries[i].is_game_over()
            if terminal or move_numbers[i] >= max_moves:
                active[i] = False

    for i in range(num_games):
        final_game = boundaries[i] if not paths[i] else materialize(boundaries[i], paths[i])
        game_over = not paths[i] and final_game.is_game_over()
        for ex in examples_per_game[i]:
            ex.value_target = terminal_value(final_game, ex.decider) if game_over else 0.0
    return examples_per_game


def _play_self_play_games_batch_phase_actions_only(
    network: torch.nn.Module,
    num_games: int,
    num_simulations: int,
    num_players: int = 2,
    kingdom: list[str] | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    action_bias: float = DEFAULT_ACTION_BIAS,
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[list[Example]]:
    """The original phase-actions-only batched self-play loop, kept
    verbatim as the `search_sub_decisions=False` fallback -- see
    `_play_self_play_game_phase_actions_only`."""
    master_rng = random.Random(seed)
    games: list[Game] = []
    np_rngs: list[np.random.Generator] = []
    for _ in range(num_games):
        g_seed = master_rng.randrange(2**31)
        if kingdom is None:
            game_kingdom = random.Random(g_seed).sample(list(KINGDOM_CARDS), 10)
        else:
            game_kingdom = kingdom
        games.append(Game(game_kingdom, num_players=num_players, seed=g_seed))
        np_rngs.append(np.random.default_rng(g_seed))

    examples_per_game: list[list[Example]] = [[] for _ in range(num_games)]
    move_numbers = [0] * num_games
    active = [True] * num_games

    while any(active):
        idxs = [i for i in range(num_games) if active[i]]
        roots = run_mcts_batch(
            [games[i] for i in idxs],
            network,
            num_simulations,
            c_puct=c_puct,
            add_noise=True,
            action_bias=action_bias,
            device=device,
            rngs=[np_rngs[i] for i in idxs],
        )

        for i, root in zip(idxs, roots):
            game = games[i]
            decider = game.current_decider()
            dist = visit_distribution(root)

            policy_target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
            for action, prob in dist.items():
                policy_target[encoding.ACTION_INDEX[action]] = prob

            examples_per_game[i].append(Example(
                obs=encoding.encode_observation(game, decider),
                mask=encoding.legal_action_mask(game),
                policy_target=policy_target,
                decider=decider,
            ))

            temperature = 1.0 if move_numbers[i] < temperature_moves else 0.0
            action = select_action(root, temperature, rng=np_rngs[i])
            advance_to_next_phase_action(game, action)
            move_numbers[i] += 1
            if game.is_game_over() or move_numbers[i] >= max_moves:
                active[i] = False

    for i, game in enumerate(games):
        game_over = game.is_game_over()
        for ex in examples_per_game[i]:
            ex.value_target = terminal_value(game, ex.decider) if game_over else 0.0
    return examples_per_game


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.examples: list[Example] = []

    def add_game(self, examples: list[Example]) -> None:
        self.examples.extend(examples)
        overflow = len(self.examples) - self.capacity
        if overflow > 0:
            del self.examples[:overflow]

    def sample(self, batch_size: int, rng: random.Random) -> list[Example]:
        return rng.sample(self.examples, min(batch_size, len(self.examples)))

    def __len__(self) -> int:
        return len(self.examples)
