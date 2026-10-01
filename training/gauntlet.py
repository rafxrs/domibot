"""The strategy gauntlet: a checkpoint's raw policy against each scripted bot, on
kingdoms holding the bot's cards, with BigMoney + terminal on the same kingdoms
for comparison.

    python -m training.gauntlet checkpoints/domibot2/domibot2.4.pt --kingdoms 300 --workers 6

The subject can also be a scripted agent's name (`strategy_bots.SCRIPTED`).
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from .env import random_kingdom
from .evaluate import play_both_seats, wilson
from .strategy_bots import ALL_BOTS, SCRIPTED

_AGENT_CACHE: dict[str, object] = {}


def load_agent(spec: str):
    """A scripted agent by name, or a checkpoint's greedy policy on CPU (cached per process)."""
    if spec in SCRIPTED:
        return SCRIPTED[spec]()
    if spec not in _AGENT_CACHE:
        from .agents import PPOAgent
        from .network import DomibotNet

        torch.set_num_threads(1)  # several worker processes run side by side
        network = DomibotNet.load(spec, map_location="cpu")
        network.eval()
        _AGENT_CACHE[spec] = PPOAgent(network, device=torch.device("cpu"))
    return _AGENT_CACHE[spec]


def home_kingdoms(home: tuple[str, ...], n: int, seed: int) -> list[tuple[list[str], int]]:
    """`n` (kingdom, game seed) pairs, each kingdom `home` plus random cards."""
    rng = random.Random(seed)
    return [(random_kingdom(rng, home), rng.randrange(1_000_000)) for _ in range(n)]


def _play_chunk(subject: str, opponent: str, games: list[tuple[list[str], int]]) -> tuple[int, int, int]:
    return play_both_seats(load_agent(subject), load_agent(opponent), games)


def run_matchup(pool: ProcessPoolExecutor, subject: str, opponent: str, games: list[tuple[list[str], int]],
                workers: int) -> tuple[int, int, int]:
    chunks = [games[i::workers] for i in range(workers) if games[i::workers]]
    results = list(pool.map(_play_chunk, [subject] * len(chunks), [opponent] * len(chunks), chunks))
    return tuple(sum(r[x] for r in results) for x in range(3))


def _fmt(wlt: tuple[int, int, int]) -> str:
    w, l, t = wlt
    p, lo, hi = wilson(w, t, w + l + t)
    return f"{p:6.1%} ({w}-{l}-{t}) [{lo:.0%}-{hi:.0%}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", help="checkpoint path, or a scripted agent's name")
    parser.add_argument("--kingdoms", type=int, default=200, help="kingdoms per bot, each played from both seats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bots", nargs="+", default=[b.name for b in ALL_BOTS], choices=[b.name for b in ALL_BOTS])
    parser.add_argument("--no-reference", action="store_true", help="skip BigMoney + terminal on the same kingdoms")
    args = parser.parse_args()

    print(f"subject: {args.subject}  |  {args.kingdoms} kingdoms per bot, each from both seats, seed {args.seed}")
    print(f"{'opponent':20s} {'home cards':40s} {'subject vs opponent':30s} subject vs BigMoney+terminal, same kingdoms")
    worst = None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in args.bots:
            t0 = time.time()
            games = home_kingdoms(SCRIPTED[name].HOME, args.kingdoms, args.seed)
            result = run_matchup(pool, args.subject, name, games, args.workers)
            reference = "" if args.no_reference else \
                _fmt(run_matchup(pool, args.subject, "bigmoney_terminal", games, args.workers))
            score = wilson(result[0], result[2], sum(result))[0]
            if worst is None or score < worst[1]:
                worst = (name, score)
            print(f"{name:20s} {', '.join(SCRIPTED[name].HOME):40s} {_fmt(result):30s} {reference}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"worst: {worst[0]} {worst[1]:.1%}")


if __name__ == "__main__":
    main()
