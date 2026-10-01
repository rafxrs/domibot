"""How much does search add on top of a PPO-trained network? Plays the
network with MCTS against the same network's raw policy (or a scripted
baseline), on paired random kingdoms.

    python -m training.ppo.search_eval --checkpoint checkpoints/domibot2/domibot2.4.pt \\
        --simulations 400 --games 100 --seed 1

By default the search is fair (`agents.DeterminizedSearchAgent`): it only
searches what a player could know, as the relay tool does. `--perfect-info`
switches to `agents.DomibotAgent`, which searches the true game -- the
opponent's hand and every deck's order included -- for comparison. Games
are slow (one search per phase decision), so run several copies with
different --seed values in parallel and add up their W-L-T lines.
"""
from __future__ import annotations

import argparse
import time

import torch

from ..agents import BigMoneyAgent, BigMoneyTerminalAgent, DeterminizedSearchAgent, DomibotAgent
from ..evaluate import play_match, wilson
from ..network import DomibotNet
from .train import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=("raw", "bigmoney", "bigmoney_terminal"), default="raw",
                        help="raw: the same checkpoint's raw policy (no search)")
    parser.add_argument("--determinizations", type=int, default=1,
                        help="fair search only: independently resampled copies of the hidden information and "
                             "future draws to search, --simulations split between them")
    parser.add_argument("--perfect-info", action="store_true",
                        help="search the true game (agents.DomibotAgent) instead of a fair determinization")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)  # several copies run side by side; the network is tiny
    device = torch.device(args.device)
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    network.eval()
    if args.perfect_info:
        searcher = DomibotAgent(network, num_simulations=args.simulations, device=device,
                                search_sub_decisions=False)
    else:
        searcher = DeterminizedSearchAgent(network, num_simulations=args.simulations, device=device,
                                           seed=args.seed, determinizations=args.determinizations)
    opponent = {"raw": lambda: PPOAgent(network, device=device), "bigmoney": BigMoneyAgent,
                "bigmoney_terminal": BigMoneyTerminalAgent}[args.opponent]()

    t0 = time.time()
    r = play_match(searcher, opponent, n_games=args.games, seed=args.seed)
    w, t = r["agent_a_wins"], r["ties"]
    n = r["games"]
    p, lo, hi = wilson(w, t, n)
    kind = "perfect-info" if args.perfect_info else f"fair x{args.determinizations}"
    label = f"{kind} search({args.simulations}) vs {args.opponent}"
    print(f"{label}: {w}-{n - w - t}-{t} (W-L-T) over {n} = {p:.1%} [95% CI {lo:.1%}-{hi:.1%}]  "
          f"({time.time() - t0:.0f}s, seed {args.seed})", flush=True)


if __name__ == "__main__":
    main()
