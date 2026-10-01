"""The strategy gauntlet: a checkpoint's raw policy against scripted bots
for specific base-set strategies (`strategy_bots.py`), each on kingdoms
containing the cards its strategy needs. Self-play and the BigMoney evals
only show average strength against opponents that play like the network
or like Big Money; this shows the worst case against strategies it doesn't
play itself, which is what a strong human would pick when the kingdom
suits them.

    python -m training.gauntlet checkpoints/domibot2/domibot2.4.pt --kingdoms 300 --workers 6

Each bot plays `--kingdoms` kingdoms (its home cards plus random others),
each from both seats. For comparison, BigMoney+terminal plays the same
kingdoms and seeds. The subject can also be a scripted agent's name, e.g.
`bigmoney_terminal`, to check how strong a gauntlet bot is by itself.
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from domibot import KINGDOM_CARDS

from .agents import BigMoneyAgent, BigMoneyTerminalAgent
from .evaluate import play_game, wilson
from .strategy_bots import ALL_BOTS

SCRIPTED = {"bigmoney": BigMoneyAgent, "bigmoney_terminal": BigMoneyTerminalAgent,
            **{bot.name: bot for bot in ALL_BOTS}}

_AGENT_CACHE: dict[str, object] = {}


def load_agent(spec: str):
    """A scripted agent by name (`SCRIPTED`), or a checkpoint path's raw
    greedy policy (`PPOAgent`, no search) on CPU. Checkpoints are cached per
    process."""
    if spec in SCRIPTED:
        return SCRIPTED[spec]()
    if spec not in _AGENT_CACHE:
        from .network import DomibotNet
        from .ppo.train import PPOAgent

        torch.set_num_threads(1)  # several worker processes run side by side
        network = DomibotNet.load(spec, map_location="cpu")
        network.eval()
        _AGENT_CACHE[spec] = PPOAgent(network, device=torch.device("cpu"))
    return _AGENT_CACHE[spec]


def home_kingdoms(home: tuple[str, ...], n: int, seed: int) -> list[tuple[list[str], int]]:
    """`n` (kingdom, game seed) pairs, each kingdom `home` plus random others."""
    rng = random.Random(seed)
    rest = [c for c in KINGDOM_CARDS if c not in home]
    return [(list(home) + rng.sample(rest, 10 - len(home)), rng.randrange(1_000_000)) for _ in range(n)]


def _play_chunk(subject: str, opponent: str, games: list[tuple[list[str], int]]) -> tuple[int, int, int]:
    """(subject wins, opponent wins, ties) over `games`, each from both seats."""
    a, b = load_agent(subject), load_agent(opponent)
    wins = losses = ties = 0
    for kingdom, seed in games:
        for subject_seat in (0, 1):
            players = (a, b) if subject_seat == 0 else (b, a)
            winners = play_game(players[0], players[1], kingdom, seed).winners()
            if len(winners) != 1:
                ties += 1
            elif winners[0] == subject_seat:
                wins += 1
            else:
                losses += 1
    return wins, losses, ties


def run_matchup(pool: ProcessPoolExecutor, subject: str, opponent: str,
                games: list[tuple[list[str], int]], workers: int) -> tuple[int, int, int]:
    chunks = [games[i::workers] for i in range(workers) if games[i::workers]]
    results = pool.map(_play_chunk, [subject] * len(chunks), [opponent] * len(chunks), chunks)
    totals = [0, 0, 0]
    for r in results:
        totals = [x + y for x, y in zip(totals, r)]
    return tuple(totals)


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
    parser.add_argument("--bots", nargs="+", default=[bot.name for bot in ALL_BOTS], choices=[b.name for b in ALL_BOTS])
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the BigMoney+terminal games on the same kingdoms")
    args = parser.parse_args()

    bots = {bot.name: bot for bot in ALL_BOTS}
    print(f"subject: {args.subject}  |  {args.kingdoms} kingdoms per bot, each from both seats, seed {args.seed}",
          flush=True)
    print(f"{'opponent':20s} {'home cards':40s} {'subject vs opponent':30s} subject vs BigMoney+terminal, "
          f"same kingdoms", flush=True)
    worst = None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in args.bots:
            t0 = time.time()
            games = home_kingdoms(bots[name].HOME, args.kingdoms, args.seed)
            result = run_matchup(pool, args.subject, name, games, args.workers)
            reference = "" if args.no_reference else \
                _fmt(run_matchup(pool, args.subject, "bigmoney_terminal", games, args.workers))
            score = wilson(result[0], result[2], sum(result))[0]
            if worst is None or score < worst[1]:
                worst = (name, score)
            print(f"{name:20s} {', '.join(bots[name].HOME):40s} {_fmt(result):30s} {reference}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"worst: {worst[0]} {worst[1]:.1%}")


if __name__ == "__main__":
    main()
