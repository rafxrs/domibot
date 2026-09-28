"""What a checkpoint actually plays: its raw policy against itself on
random kingdoms, reporting for each kingdom card how often it ends up in a
deck when it's available, how often it's bought and how often it's played,
plus what Throne Room is used on and how many Actions a turn plays.

    python -m training.strategy_profile checkpoints/domibot2/domibot2.2.pt --games 600 --workers 6
    python -m training.strategy_profile checkpoints/domibot2/domibot2.2.pt --force "Throne Room" Village

`--force` puts those cards in every kingdom. A card a policy never buys is
one it never learns to play, so the list of cards it has stopped using is
the first thing to look at when it plateaus (see training/README.md, "Card
use").
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from domibot import KINGDOM_CARDS, Game

from .evaluate import MAX_STEPS
from .gauntlet import load_agent

PLAYS_CAP = 8  # the per-turn histogram's last bucket is "this many or more"


def _profile_chunk(subject: str, games: list[tuple[list[str], int]]) -> dict[str, Counter]:
    agent = load_agent(subject)
    stats = {k: Counter() for k in ("available", "in_deck", "bought", "plays", "throne_targets", "turn_plays", "misc")}
    for kingdom, seed in games:
        game = Game(kingdom, num_players=2, seed=seed)
        turn_key, turn_plays = None, 0
        for _ in range(MAX_STEPS):
            if game.is_game_over():
                break
            key = (game.current_player, game.turn_number)
            if key != turn_key:
                if turn_key is not None:
                    stats["turn_plays"][min(turn_plays, PLAYS_CAP)] += 1
                turn_key, turn_plays = key, 0
            action = agent.act(game)
            decision = game.pending_decision
            played = []
            if decision is None:
                if action.verb == "PLAY":
                    played = [action.card]
                elif action.verb == "BUY":
                    stats["bought"][action.card] += 1
            elif decision.source_card == "Throne Room" and action.verb == "PLAY":
                stats["throne_targets"][action.card] += 1
                played = [action.card, action.card]
            elif decision.source_card == "Vassal" and action.verb == "YES":
                played = [game.players[game.current_decider()].set_aside[-1]]
            for card in played:
                stats["plays"][card] += 1
            if game.current_decider() == game.current_player:
                turn_plays += len(played)
            game.step(action)
        if turn_key is not None:
            stats["turn_plays"][min(turn_plays, PLAYS_CAP)] += 1
        stats["misc"]["games"] += 1
        stats["misc"]["turns"] += sum(p.turns_taken for p in game.players)
        for player in game.players:
            owned = set(player.all_cards())
            for card in kingdom:
                stats["available"][card] += 1
                stats["in_deck"][card] += card in owned
    return stats


def profile(subject: str, games: list[tuple[list[str], int]], workers: int) -> dict[str, Counter]:
    chunks = [games[i::workers] for i in range(workers) if games[i::workers]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_profile_chunk, [subject] * len(chunks), chunks))
    total = {k: Counter() for k in results[0]}
    for r in results:
        for k, c in r.items():
            total[k].update(c)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", help="checkpoint path, or a scripted agent's name")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", nargs="*", default=[], choices=sorted(KINGDOM_CARDS),
                        help="cards put in every kingdom")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rest = [c for c in KINGDOM_CARDS if c not in args.force]
    games = [(list(args.force) + rng.sample(rest, 10 - len(args.force)), rng.randrange(1_000_000))
             for _ in range(args.games)]
    s = profile(args.subject, games, args.workers)

    n = s["misc"]["games"]
    print(f"{args.subject}: {n} games against itself, forced cards: {args.force or 'none'}, "
          f"{s['misc']['turns'] / (2 * n):.1f} turns per player")
    print(f"{'card':14s} {'in deck at end':>15s} {'buys':>6s} {'plays':>6s}   (per player-game where available)")
    rate = {c: s["in_deck"][c] / s["available"][c] for c in s["available"]}
    for card in sorted(rate, key=lambda c: -rate[c]):
        avail = s["available"][card]
        print(f"{card:14s} {rate[card]:15.0%} {s['bought'][card] / avail:6.2f} {s['plays'][card] / avail:6.2f}")
    unused = sorted(c for c in rate if rate[c] < 0.05)
    print(f"in fewer than 5% of decks: {', '.join(unused) if unused else 'none'}")
    targets = s["throne_targets"]
    print(f"Throne Room played on: {dict(targets.most_common()) or 'never played'}"
          + (f" (on another Throne Room {targets['Throne Room']} times)" if targets["Throne Room"] else ""))
    turns = sum(s["turn_plays"].values())
    hist = "  ".join(f"{k}{'+' if k == PLAYS_CAP else ''}: {s['turn_plays'][k] / turns:.1%}"
                     for k in range(PLAYS_CAP + 1))
    print(f"Actions played per turn ({turns} turns): {hist}")


if __name__ == "__main__":
    main()
