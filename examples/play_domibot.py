"""Watch checkpoints play each other (2-4 players), optionally saving every game's log.

    python examples/play_domibot.py --games 20 --players 2
    python examples/play_domibot.py --games 10 --players 4 --save-logs
    python examples/play_domibot.py --games 10 --checkpoints checkpoints/domibot2/domibot2.3.pt checkpoints/domibot2/domibot2.4.pt

`--checkpoints` gives each seat its own checkpoint.
"""
from __future__ import annotations

import argparse
import datetime
import random
from collections import Counter
from pathlib import Path

from common import DEFAULT_CHECKPOINT, GAME_LOGS_DIR, load_network
from domibot import Game, KINGDOM_CARDS
from training.agents import DomibotAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--players", type=int, default=2, help="2-4")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="for every seat")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="one per seat")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per decision")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 adds variety")
    parser.add_argument("--save-logs", action="store_true", help="save every game's log to game_logs/")
    parser.add_argument("--kingdom", type=str, nargs=10, default=None, help="fix the kingdom")
    parser.add_argument("--seed", type=int, default=0, help="game i uses seed + i")
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available")
    args = parser.parse_args()
    if not 2 <= args.players <= 4:
        raise SystemExit("--players must be between 2 and 4")
    paths = args.checkpoints or [args.checkpoint] * args.players
    if len(paths) != args.players:
        raise SystemExit(f"--checkpoints has {len(paths)} entries but --players is {args.players}")

    agents = []
    for path in paths:
        network, device = load_network(path, args.gpu)
        agents.append(DomibotAgent(network, num_simulations=args.simulations, temperature=args.temperature,
                                   device=device))
    print(f"Players: {[Path(p).name for p in paths]} on {device}, {args.simulations} sims/decision")

    rng = random.Random(args.seed)
    wins: Counter[int] = Counter()
    ties = 0
    totals = [0.0] * args.players
    for i in range(args.games):
        kingdom = list(args.kingdom) if args.kingdom else rng.sample(list(KINGDOM_CARDS), 10)
        game = Game(kingdom, num_players=args.players, seed=args.seed + i)
        while not game.is_game_over():
            game.step(agents[game.current_decider()].act(game))
        scores, winners = game.get_scores(), game.winners()
        for p in range(args.players):
            totals[p] += scores[p]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
        print(f"game {i}: seed={args.seed + i} scores=[{', '.join(f'P{p}={scores[p]}' for p in range(args.players))}] "
              f"winners={winners}")
        if args.save_logs:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            game.save_log(GAME_LOGS_DIR / "_vs_".join(Path(p).stem for p in paths) /
                          f"{stamp}_game{i}_seed{args.seed + i}.log")
    print(f"\n=== {args.games} games complete ===")
    for p in range(args.players):
        print(f"  P{p} ({Path(paths[p]).name}): {wins[p]} wins, avg score {totals[p] / args.games:.1f}")
    print(f"  ties: {ties}")


if __name__ == "__main__":
    main()
