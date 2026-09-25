"""Generalized Advantage Estimation over a self-play game's interleaved,
two-(or more-)player decision trajectory.

Every decision in a self-play game -- phase action or card-effect
sub-decision alike, the same "every decision is a training example" model
`training/self_play.py` already validated for the MCTS lineage -- produces
one `Transition`. Reward is 0 everywhere except the last transition each
decider gets, which carries that decider's own terminal outcome (a
margin-based value via `mcts.terminal_value` by default, so a blowout and
a nail-biter are still distinguishable, exactly like the MCTS lineage's
value targets). GAE is then just standard, undiscounted-by-default 1-D GAE
run independently along each decider's own ordered subsequence -- the same
"extract this decider's own subsequence from the interleaved trajectory"
pattern `training/self_play.py`'s `_backfill_value_targets` already used
for TD-bootstrapping, generalized here to a full advantage estimator
instead of a single-step blend.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from domibot import Game

from ..mcts import terminal_value


def win_weighted_value(game: Game, perspective: int, win_weight: float = 0.8) -> float:
    """`win_weight` * (+1 win / -1 loss / 0 tie, per `game.winners()`, so
    the fewest-turns tiebreak counts) plus the rest as `terminal_value`'s
    tanh margin. Margin alone makes a 1-point win and a 1-point loss worth
    +0.1 and -0.1, so the win/loss line barely registers; `win_weight=0`
    is exactly `terminal_value`."""
    winners = game.winners()
    result = 0.0 if len(winners) != 1 else (1.0 if winners[0] == perspective else -1.0)
    return win_weight * result + (1.0 - win_weight) * terminal_value(game, perspective)


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int  # index into encoding.ACTION_VOCAB
    log_prob: float  # log pi(action | obs) under the policy that acted, at collection time
    value: float  # V(obs), from the same network, at collection time
    decider: int
    turn_number: int  # game.players[decider].turns_taken at the time of the decision
    # Filled in by compute_gae -- 0.0 until then.
    reward: float = field(default=0.0)
    advantage: float = field(default=0.0)
    return_: float = field(default=0.0)  # the value-head training target: advantage + value


def compute_gae(
    transitions: list[Transition],
    final_game: Game,
    game_over: bool,
    gamma: float = 1.0,
    lam: float = 0.95,
    reward_fn: Callable[[Game, int], float] = terminal_value,
) -> None:
    """Mutates each Transition's `.reward`/`.advantage`/`.return_` in
    place, in order. `game_over=False` (the episode hit a step cap with no
    real outcome) leaves every reward at 0.0 -- no fabricated terminal
    value -- and still produces a usable, if less informative, advantage
    via value-to-value bootstrapping (see below); this recovers signal
    from a truncated episode the same way `td_lambda > 0` did for the MCTS
    lineage's value targets, rather than discarding it entirely.

    For each decider's own subsequence (preserving original order):
    `delta_t = reward_t + gamma * V_{t+1} - V_t`,
    `A_t = delta_t + gamma * lam * A_{t+1}`,
    with `V_{t+1}` for the last transition in the subsequence being 0.0 if
    the game genuinely ended (nothing follows), or that same transition's
    own `V_t` if it didn't (a truncated episode's tail bootstraps from its
    own value estimate -- the standard practical fix so a cut-off episode
    doesn't get a spuriously large "the world ended and that's bad"
    advantage at its last step)."""
    by_decider: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(transitions):
        by_decider[t.decider].append(i)

    for decider, idxs in by_decider.items():
        n = len(idxs)
        values = [transitions[i].value for i in idxs]
        rewards = [0.0] * n
        if game_over:
            rewards[-1] = float(reward_fn(final_game, decider))
        next_values = values[1:] + [0.0 if game_over else values[-1]]

        advantages = [0.0] * n
        running = 0.0
        for t in range(n - 1, -1, -1):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            running = delta + gamma * lam * running
            advantages[t] = running

        for pos, i in enumerate(idxs):
            transitions[i].reward = rewards[pos]
            transitions[i].advantage = advantages[pos]
            transitions[i].return_ = advantages[pos] + values[pos]
