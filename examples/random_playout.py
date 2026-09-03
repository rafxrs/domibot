"""Play random games to completion as a smoke test / usage example."""
from __future__ import annotations

import random
import sys

from domibot import Game, KINGDOM_CARDS


def random_playout(kingdom: list[str], num_players: int, seed: int) -> Game:
    game = Game(kingdom, num_players=num_players, seed=seed)
    rng = random.Random(seed)
    steps = 0
    max_steps = 200_000
    while not game.is_game_over():
        actions = game.legal_actions()
        assert actions, "no legal actions but game is not over"
        game.step(rng.choice(actions))
        steps += 1
        if steps > max_steps:
            raise RuntimeError("playout did not terminate")
    return game


def main() -> None:
    all_kingdom = list(KINGDOM_CARDS)
    rng = random.Random(0)
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    for i in range(n_games):
        kingdom = rng.sample(all_kingdom, 10)
        num_players = rng.choice([2, 3, 4])
        game = random_playout(kingdom, num_players, seed=i)
        scores = game.get_scores()
        print(
            f"game {i}: players={num_players} turns={game.turn_number} "
            f"winners={game.winners()} scores={scores} kingdom={sorted(kingdom)}"
        )


if __name__ == "__main__":
    main()
