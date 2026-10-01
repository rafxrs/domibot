"""Play one game against a trained checkpoint, in the terminal or a pygame window.

    python examples/play_vs_domibot.py
    python examples/play_vs_domibot.py --checkpoint checkpoints/domibot1/domibot_v4.4.pt --simulations 400
    python examples/play_vs_domibot.py --gui

Domibot runs MCTS before each decision: more --simulations is stronger but slower.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import DEFAULT_CHECKPOINT, GAME_LOGS_DIR, load_network
from domibot import Game, KINGDOM_CARDS
from play_vs_random import HUMAN, finish, play_in_terminal
from training.agents import DomibotAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per Domibot decision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available")
    parser.add_argument("--gui", action="store_true", help="play in a pygame window instead of the text prompt")
    args = parser.parse_args()

    network, device = load_network(args.checkpoint, args.gpu)
    domibot = DomibotAgent(network, num_simulations=args.simulations, device=device)
    print(f"Loaded {args.checkpoint} onto {device}, {args.simulations} sims/decision.")
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    kingdom = random.Random(seed).sample(list(KINGDOM_CARDS), 10)
    print(f"Kingdom (seed {seed}): {sorted(kingdom)}")
    game = Game(kingdom, num_players=2, seed=seed)
    if args.gui:
        from gui.app import DominionGUI  # needs pygame only when used

        DominionGUI(game, domibot, human_seat=HUMAN).run()
    else:
        play_in_terminal(game, domibot, "Domibot")
    finish(game, "Domibot", GAME_LOGS_DIR / f"human_vs_{Path(args.checkpoint).stem}", seed)


if __name__ == "__main__":
    main()
