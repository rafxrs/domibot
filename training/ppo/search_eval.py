"""How much does search add to a network? MCTS with it vs its raw policy (or a baseline).

    python -m training.ppo.search_eval --checkpoint checkpoints/domibot2/domibot2.4.pt \\
        --simulations 400 --games 100 --seed 1

The search is fair by default (`DeterminizedSearchAgent`, hidden information
resampled); `--perfect-info` searches the true game. Games are slow, so run
several seeds in parallel and add up the W-L-T lines.
"""
from __future__ import annotations

import argparse
import time

import torch

from ..agents import BigMoneyAgent, BigMoneyTerminalAgent, DeterminizedSearchAgent, DomibotAgent, PPOAgent
from ..evaluate import play_match, wilson
from ..network import DomibotNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=("raw", "bigmoney", "bigmoney_terminal"), default="raw",
                        help="raw: the same checkpoint's raw policy")
    parser.add_argument("--determinizations", type=int, default=1,
                        help="fair search: resampled worlds to search, splitting --simulations")
    parser.add_argument("--perfect-info", action="store_true", help="search the true game (agents.DomibotAgent)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    network.eval()
    if args.perfect_info:
        searcher = DomibotAgent(network, num_simulations=args.simulations, device=device, search_sub_decisions=False)
    else:
        searcher = DeterminizedSearchAgent(network, num_simulations=args.simulations, device=device, seed=args.seed,
                                           determinizations=args.determinizations)
    opponent = {"raw": lambda: PPOAgent(network, device=device), "bigmoney": BigMoneyAgent,
                "bigmoney_terminal": BigMoneyTerminalAgent}[args.opponent]()

    t0 = time.time()
    r = play_match(searcher, opponent, n_games=args.games, seed=args.seed)
    w, t, n = r["agent_a_wins"], r["ties"], r["games"]
    p, lo, hi = wilson(w, t, n)
    kind = "perfect-info" if args.perfect_info else f"fair x{args.determinizations}"
    print(f"{kind} search({args.simulations}) vs {args.opponent}: {w}-{n - w - t}-{t} (W-L-T) over {n} = {p:.1%} "
          f"[95% CI {lo:.1%}-{hi:.1%}]  ({time.time() - t0:.0f}s, seed {args.seed})", flush=True)


if __name__ == "__main__":
    main()
