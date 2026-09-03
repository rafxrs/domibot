"""A Gym-style wrapper around domibot.Game.

Not a subclass of gymnasium.Env (no dependency on the gymnasium package),
but it follows the same shape (reset -> obs/info, step -> obs/reward/
terminated/truncated/info) so wrapping it for a specific RL library later
is mechanical.

Dominion is inherently a variable-legal-actions game (what you can do
depends on the phase, your hand, and any card effect currently resolving),
so this uses the standard masked-discrete-action pattern: the action space
is `encoding.NUM_ACTIONS` fixed slots, and every observation carries an
`action_mask` alongside it. A policy should never sample a masked-out
action — `DominionEnv.step` raises if you do.

It's also inherently multi-agent / turn-based: `step` always acts on
behalf of whichever player currently must decide (`Game.current_decider()`,
which is the active player during their own turn, or an opponent reacting
to an attack). The observation returned is always from the perspective of
whoever must decide *next*, so a self-play loop just needs to route each
successive observation to whichever policy controls that seat.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from domibot import Game, KINGDOM_CARDS

from . import encoding

Observation = dict  # {"observation": np.ndarray[OBS_DIM], "action_mask": np.ndarray[NUM_ACTIONS] bool}


class DominionEnv:
    def __init__(self, num_players: int = 2, max_steps: int = 100_000):
        self.num_players = num_players
        self.max_steps = max_steps
        self.game: Optional[Game] = None
        self._steps = 0

    def reset(self, kingdom: Optional[list[str]] = None, seed: Optional[int] = None) -> tuple[Observation, dict]:
        rng = random.Random(seed)
        if kingdom is None:
            kingdom = rng.sample(list(KINGDOM_CARDS), 10)
        self.game = Game(kingdom, num_players=self.num_players, seed=seed)
        self._steps = 0
        return self._observe(), {"kingdom": kingdom}

    def step(self, action_index: int) -> tuple[Observation, float, bool, bool, dict]:
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        action = encoding.index_to_action(action_index)
        legal = self.game.legal_actions()
        if action not in legal:
            raise ValueError(f"action {action!r} is not legal right now (legal: {legal})")

        actor = self.game.current_decider()
        self.game.step(action)
        self._steps += 1

        terminated = self.game.is_game_over()
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = self._reward_for(actor) if terminated else 0.0

        info: dict = {}
        if terminated:
            info["scores"] = self.game.get_scores()
            info["winners"] = self.game.winners()

        return self._observe(), reward, terminated, truncated, info

    def _reward_for(self, player_idx: int) -> float:
        """Sparse terminal reward from `player_idx`'s perspective: +1 win,
        -1 loss, 0 tie. Reward is 0 on every non-terminal step; dense
        shaping (e.g. victory-point deltas) can be layered on later."""
        winners = self.game.winners()
        if len(winners) != 1:
            return 0.0
        return 1.0 if winners[0] == player_idx else -1.0

    def _observe(self) -> Observation:
        player = self.game.current_decider() if not self.game.is_game_over() else self.game.current_player
        return {
            "observation": encoding.encode_observation(self.game, player),
            "action_mask": encoding.legal_action_mask(self.game) if not self.game.is_game_over()
            else np.zeros(encoding.NUM_ACTIONS, dtype=bool),
        }
