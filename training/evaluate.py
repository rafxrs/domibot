"""Play two agents against each other over many games and report a win rate.
The sanity check for this whole foundation layer: if BigMoneyAgent can't
reliably beat RandomAgent, something upstream is broken.

    python -m training.evaluate [n_games] [seed]
"""
from __future__ import annotations

import random
import sys

from domibot import Game, KINGDOM_CARDS

from .agents import Agent, BigMoneyAgent, RandomAgent

# Comfortably above any real game (a few hundred to a couple thousand
# atomic steps) but bounded: an undertrained/greedy agent can otherwise
# stall forever always preferring END_BUY over any purchase, since nothing
# forces a buy. Rather than raise (which would crash train.py's periodic
# eval the first time it hits an undertrained network), a game that hits
# this cap is just returned as-is — winners()/get_scores() still work,
# deciding it by whatever the current standing is.
MAX_STEPS = 10_000


def play_game(agent_p0: Agent, agent_p1: Agent, kingdom: list[str], seed: int) -> Game:
    game = Game(kingdom, num_players=2, seed=seed)
    agents = [agent_p0, agent_p1]
    for _ in range(MAX_STEPS):
        if game.is_game_over():
            break
        game.step(agents[game.current_decider()].act(game))
    return game


def play_match(agent_a: Agent, agent_b: Agent, n_games: int, seed: int = 0) -> dict:
    """Alternates who goes first each game, to cancel out first-player
    advantage, then reports how often each agent wins."""
    rng = random.Random(seed)
    wins_a = wins_b = ties = 0
    for i in range(n_games):
        kingdom = rng.sample(list(KINGDOM_CARDS), 10)
        game_seed = rng.randrange(1_000_000)
        a_first = i % 2 == 0
        game = play_game(agent_a, agent_b, kingdom, game_seed) if a_first \
            else play_game(agent_b, agent_a, kingdom, game_seed)
        a_idx = 0 if a_first else 1

        winners = game.winners()
        if len(winners) != 1:
            ties += 1
        elif winners[0] == a_idx:
            wins_a += 1
        else:
            wins_b += 1
    return {"agent_a_wins": wins_a, "agent_b_wins": wins_b, "ties": ties, "games": n_games}


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    result = play_match(BigMoneyAgent(), RandomAgent(seed=seed), n_games, seed=seed)
    print(f"BigMoney vs Random over {result['games']} games:")
    print(f"  BigMoney wins: {result['agent_a_wins']}")
    print(f"  Random wins:   {result['agent_b_wins']}")
    print(f"  Ties:          {result['ties']}")


if __name__ == "__main__":
    main()
