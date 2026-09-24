"""Play N games between M Domibot agents (2-4 players) and, optionally,
save every game's log.

    python examples/play_domibot.py --games 20 --players 2
    python examples/play_domibot.py --games 10 --players 4 --checkpoint checkpoints/domibot_v4.4.pt --save-logs
    python examples/play_domibot.py --games 10 --checkpoints checkpoints/domibot_v4.3.pt checkpoints/domibot_v4.4.pt

The last example loads a *different* checkpoint per seat (length must match
--players) -- handy for checking whether a later checkpoint actually beats
an earlier one, not just BigMoney/Random.

Defaults to checkpoints/domibot2/domibot2.1.pt, the strongest checkpoint
in the project (see training/README.md's checkpoint lineage table).

Runs on CPU by default so it doesn't compete with a training run that may
still be using the GPU; pass --gpu once nothing else needs it.
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
from collections import Counter
from pathlib import Path

import torch

from domibot import Game, KINGDOM_CARDS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # training/ is a sibling of examples/, not on sys.path by default
from training.agents import DomibotAgent  # noqa: E402
from training.network import DomibotNet, get_device  # noqa: E402

GAME_LOGS_DIR = ROOT / "game_logs"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "domibot2" / "domibot2.1.pt"


def load_agents(checkpoint_paths: list[str], num_simulations: int, temperature: float, device: torch.device) -> list[DomibotAgent]:
    agents = []
    for path in checkpoint_paths:
        net = DomibotNet.load(path, map_location=device).to(device)
        net.eval()
        agents.append(DomibotAgent(net, num_simulations=num_simulations, temperature=temperature, device=device))
    return agents


def play_one_game(agents: list[DomibotAgent], kingdom: list[str], num_players: int, seed: int) -> Game:
    game = Game(kingdom, num_players=num_players, seed=seed)
    while not game.is_game_over():
        game.step(agents[game.current_decider()].act(game))
    return game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--players", type=int, default=2, help="number of Domibot agents / players (2-4)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                         help="checkpoint used for every seat, unless --checkpoints overrides it")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                         help="one checkpoint per seat (length must equal --players), for mixed matchups")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per decision")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy/strongest; >0 adds variety")
    parser.add_argument("--save-logs", action="store_true", help="save every game's log to game_logs/")
    parser.add_argument("--kingdom", type=str, nargs=10, default=None, help="fix the kingdom instead of randomizing per game")
    parser.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed seed+i")
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available (avoid while also training)")
    args = parser.parse_args()

    if not (2 <= args.players <= 4):
        raise SystemExit("--players must be between 2 and 4")

    checkpoint_paths = args.checkpoints if args.checkpoints else [args.checkpoint] * args.players
    if len(checkpoint_paths) != args.players:
        raise SystemExit(f"--checkpoints has {len(checkpoint_paths)} entries but --players is {args.players}")
    for path in checkpoint_paths:
        if not Path(path).exists():
            raise SystemExit(f"no checkpoint at {path} -- download domibot2.1.pt from https://github.com/rafxrs/domibot/releases into checkpoints/domibot2/ (see README Setup), or train your own")

    device = get_device() if args.gpu else torch.device("cpu")
    agents = load_agents(checkpoint_paths, args.simulations, args.temperature, device)
    print(f"Players: {[Path(p).name for p in checkpoint_paths]} on {device}, {args.simulations} sims/decision")

    matchup_dir = GAME_LOGS_DIR / "_vs_".join(Path(p).stem for p in checkpoint_paths)

    rng = random.Random(args.seed)
    win_counts: Counter[int] = Counter()
    tie_games = 0
    score_totals = [0.0] * args.players

    for i in range(args.games):
        kingdom = list(args.kingdom) if args.kingdom else rng.sample(list(KINGDOM_CARDS), 10)
        game_seed = args.seed + i
        game = play_one_game(agents, kingdom, args.players, game_seed)

        scores = game.get_scores()
        winners = game.winners()
        for p in range(args.players):
            score_totals[p] += scores[p]
        if len(winners) == 1:
            win_counts[winners[0]] += 1
        else:
            tie_games += 1

        score_str = ", ".join(f"P{p}={scores[p]}" for p in range(args.players))
        print(f"game {i}: seed={game_seed} scores=[{score_str}] winners={winners}")

        if args.save_logs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = matchup_dir / f"{timestamp}_game{i}_seed{game_seed}.log"
            game.save_log(path)

    print(f"\n=== {args.games} games complete ===")
    for p in range(args.players):
        avg_score = score_totals[p] / args.games
        print(f"  P{p} ({Path(checkpoint_paths[p]).name}): {win_counts[p]} wins, avg score {avg_score:.1f}")
    print(f"  ties: {tie_games}")


if __name__ == "__main__":
    main()
