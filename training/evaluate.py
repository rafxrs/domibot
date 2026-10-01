"""Head-to-head play between agents, and confidence intervals for the results.

    python -m training.evaluate [n_games] [seed]    # BigMoney vs Random, a sanity check
"""
from __future__ import annotations

import math
import random
import sys

from domibot import Game

from .agents import Agent, BigMoneyAgent, RandomAgent
from .env import random_kingdom

MAX_STEPS = 10_000  # a stalled game (an untrained agent never buying) ends here, scored as it stands


def play_game(agent_p0: Agent, agent_p1: Agent, kingdom: list[str], seed: int) -> Game:
    game = Game(kingdom, num_players=2, seed=seed)
    agents = [agent_p0, agent_p1]
    for _ in range(MAX_STEPS):
        if game.is_game_over():
            break
        game.step(agents[game.current_decider()].act(game))
    return game


def play_match(agent_a: Agent, agent_b: Agent, n_games: int, seed: int = 0, paired: bool = True) -> dict:
    """Win counts over `n_games` random kingdoms. `paired` replays each kingdom and
    seed with the seats swapped, which cancels most of the kingdom luck."""
    rng = random.Random(seed)
    wins_a = wins_b = ties = 0
    for i in range(n_games):
        if not paired or i % 2 == 0:
            kingdom, game_seed = random_kingdom(rng), rng.randrange(1_000_000)
        a_first = i % 2 == 0
        game = play_game(agent_a, agent_b, kingdom, game_seed) if a_first \
            else play_game(agent_b, agent_a, kingdom, game_seed)
        winners = game.winners()
        if len(winners) != 1:
            ties += 1
        elif winners[0] == (0 if a_first else 1):
            wins_a += 1
        else:
            wins_b += 1
    return {"agent_a_wins": wins_a, "agent_b_wins": wins_b, "ties": ties, "games": n_games}


def play_both_seats(agent_a: Agent, agent_b: Agent, games: list[tuple[list[str], int]]) -> tuple[int, int, int]:
    """`agent_a`'s (wins, losses, ties) over (kingdom, seed) pairs, each played from both seats."""
    w = l = t = 0
    for kingdom, seed in games:
        for seat in (0, 1):
            winners = play_game(*((agent_a, agent_b) if seat == 0 else (agent_b, agent_a)), kingdom, seed).winners()
            if len(winners) != 1:
                t += 1
            elif winners[0] == seat:
                w += 1
            else:
                l += 1
    return w, l, t


def wilson(wins: int, ties: int, games: int) -> tuple[float, float, float]:
    """(score, low, high): wins + half the ties as a fraction of games, with its 95% Wilson interval."""
    p = (wins + 0.5 * ties) / games
    z = 1.96
    d = 1 + z * z / games
    c = (p + z * z / (2 * games)) / d
    h = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return p, c - h, c + h


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = play_match(BigMoneyAgent(), RandomAgent(seed=seed), n_games, seed=seed)
    print(f"BigMoney vs Random over {r['games']} games: {r['agent_a_wins']}-{r['agent_b_wins']}-{r['ties']}")
