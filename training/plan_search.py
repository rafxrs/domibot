"""Search for the buy plan that beats a checkpoint on a board, as Provincial (an
evolutionary Dominion AI) did.

A `Plan` is a buy menu, (card, copies wanted) in priority order, plus when
Provinces, Duchies and Estates take over. `PlanAgent` plays its cards with
generic rules, or with `--network-plays` the checkpoint's own policy does, so
the search finds where the network's buying is wrong given its own play.

`evolve` scores a population on the same fresh games each generation (both
seats), keeps the best as parents and fills the rest with mutations. Scores
add up across generations and are ranked shrunk toward the population mean,
so a newcomer's lucky generation doesn't outrank a proven plan. The finalists
are re-scored on fresh games, since the search overrates its own winners.

    python -m training.plan_search checkpoints/domibot2/domibot2.4.pt --boards 12 --workers 7
    python -m training.plan_search checkpoints/domibot2/domibot2.4.pt --home "Throne Room" Village Smithy
    python -m training.plan_search checkpoints/domibot2/domibot2.4.pt --kingdom Cellar Chapel ...
    python -m training.plan_search checkpoints/domibot2/domibot2.4.pt --boards 12 --network-plays

Each board's result is printed and appended to `--out` as a JSON line.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from domibot import Action, CardType, Game, Phase
from domibot.models import DONE, END_ACTIONS, END_BUY, NO, YES

from .env import random_kingdom
from .evaluate import play_both_seats, wilson
from .gauntlet import load_agent
from .heuristics import heuristic_reaction
from .strategy_bots import TREASURE, chapel_choice, is_junk

UNLIMITED = 99
MONEY = (("Gold", UNLIMITED), ("Silver", UNLIMITED))
MENU_BASICS = ("Copper", "Silver", "Gold", "Estate", "Duchy")
PLAN_MONEY = {**TREASURE, "Market": 1, "Festival": 2}  # what Chapel keeps $6 of

# Play order: +Actions first, then Throne Room, then terminals.
NON_TERMINALS = ("Festival", "Village", "Laboratory", "Market", "Sentry", "Poacher", "Merchant", "Harbinger", "Cellar")
TERMINALS = ("Witch", "Council Room", "Library", "Smithy", "Moat", "Militia", "Bandit", "Artisan", "Mine",
             "Moneylender", "Bureaucrat", "Remodel", "Workshop", "Vassal", "Chapel")
VILLAGES = ("Village", "Festival")
THRONE_TARGETS = ("Smithy", "Council Room", "Witch", "Library", "Laboratory", "Market", "Festival", "Militia",
                  "Bandit", "Moneylender", "Mine", "Artisan", "Workshop", "Remodel", "Bureaucrat", "Sentry",
                  "Poacher", "Merchant", "Village", "Harbinger", "Cellar", "Moat", "Vassal", "Chapel")
JUNK = ("Curse", "Estate", "Duchy", "Province", "Gardens")


@dataclass(frozen=True)
class Plan:
    menu: tuple[tuple[str, int], ...]  # (card, copies wanted), highest priority first
    province_turn: int = 0  # Provinces at $8 only from this turn on
    duchy_at: int = 4  # a Duchy at $5+ once this few Provinces are left
    estate_at: int = 2  # an Estate at $2+ once this few Provinces are left

    def describe(self) -> str:
        menu = ", ".join(f"{c}{'' if n >= UNLIMITED else ' x' + str(n)}" for c, n in self.menu)
        return (f"[{menu}]  Provinces from turn {self.province_turn}, Duchy at <={self.duchy_at} "
                f"Provinces left, Estate at <={self.estate_at}")


def _first_play(actions: list[Action], cards) -> Action | None:
    return next((Action("PLAY", c) for c in cards if Action("PLAY", c) in actions), None)


class PlanAgent:
    """Buys by `plan`; plays everything else with `player` if given, else generic rules.
    Stateless, so one instance can play many games at once."""

    def __init__(self, plan: Plan, player=None):
        self.plan = plan
        self.player = player

    def act(self, game: Game) -> Action:
        actions, me, decision = game.legal_actions(), game.players[game.current_decider()], game.pending_decision
        if decision is None and game.phase == Phase.BUY:
            return self.buy(game, actions, me)
        if self.player is not None:
            return self.player.act(game)
        if decision is None:
            return self._play(game, actions, me)
        source = decision.source_card if game.current_decider() == game.current_player else None
        verb = actions[0].verb
        if source == "Throne Room" and verb == "PLAY":
            return self._throne_target(actions, me)
        if source == "Chapel":
            return chapel_choice(game, PLAN_MONEY)
        if source in ("Workshop", "Artisan", "Remodel") and verb == "GAIN":
            return self._gain(game, actions, me)
        if source == "Remodel" and verb == "TRASH":
            order = (("Gold",) if game.supply.get("Province", 0) <= 4 else ()) + ("Curse", "Estate", "Copper")
            pick = next((Action("TRASH", c) for c in order if Action("TRASH", c) in actions), None)
            return pick or min(actions, key=lambda a: game.cards[a.card].cost if a.card else 99)
        if source == "Artisan" and verb == "TOPDECK" and me.actions == 0:
            playable = [a for a in actions if a.card and CardType.ACTION in game.cards[a.card].types]
            if playable:  # an Action it can't play now is best drawn next turn
                return max(playable, key=lambda a: game.cards[a.card].cost)
        if source == "Mine" and verb == "TRASH":
            pick = next((Action("TRASH", c) for c in ("Silver", "Copper") if Action("TRASH", c) in actions), None)
            if pick:
                return pick
        if source == "Harbinger":
            options = [a for a in actions if a.card and a.card not in JUNK and a.card != "Copper"]
            return max(options, key=lambda a: game.cards[a.card].cost) if options else Action("NONE")
        if source == "Cellar":
            return next((a for a in actions if a.card in JUNK), DONE)
        if source == "Vassal":
            return YES
        if source == "Library":
            return YES if me.actions > 0 else NO
        return heuristic_reaction(game)

    def _play(self, game: Game, actions: list[Action], me) -> Action:
        cellar_ok = any(c in JUNK for c in me.hand)
        pick = _first_play(actions, [c for c in NON_TERMINALS if c != "Cellar" or cellar_ok])
        if pick:
            return pick
        if Action("PLAY", "Throne Room") in actions and any(
                c != "Throne Room" and CardType.ACTION in game.cards[c].types for c in me.hand):
            return Action("PLAY", "Throne Room")
        return _first_play(actions, [c for c in TERMINALS if self._worth_playing(game, c, me)]) or END_ACTIONS

    @staticmethod
    def _worth_playing(game: Game, card: str, me) -> bool:
        hand = me.hand
        if card == "Moneylender":
            return "Copper" in hand
        if card == "Mine":
            return "Copper" in hand or "Silver" in hand
        if card == "Remodel":
            return ("Gold" in hand and game.supply.get("Province", 0) <= 4) or "Curse" in hand or "Estate" in hand
        if card == "Chapel":
            return any(is_junk(game, me, c, PLAN_MONEY) for c in hand)
        if card == "Library":
            return len(hand) <= 5
        return True

    @staticmethod
    def _throne_target(actions: list[Action], me) -> Action:
        hand = Counter(me.hand)
        if me.actions == 0 and any(c in TERMINALS for c in hand):  # out of actions: a village first
            pick = _first_play(actions, VILLAGES)
            if pick:
                return pick
        others = sum(n for c, n in hand.items() if c != "Throne Room" and (c in TERMINALS or c in NON_TERMINALS))
        return _first_play(actions, (("Throne Room",) if others >= 2 else ()) + THRONE_TARGETS) or actions[0]

    def _gain(self, game: Game, actions: list[Action], me) -> Action:
        """The plan's first card still wanted, else the most expensive non-Curse."""
        owned = Counter(me.all_cards())
        for card, wanted in self.plan.menu:
            if owned[card] < wanted and Action("GAIN", card) in actions:
                return Action("GAIN", card)
        options = [a for a in actions if a.card and a.card != "Curse"] or actions
        return max(options, key=lambda a: game.cards[a.card].cost if a.card else -1)

    def buy(self, game: Game, actions: list[Action], me) -> Action:
        plan, coins, provinces = self.plan, me.coins, game.supply.get("Province", 0)
        if coins >= 8 and me.turns_taken >= plan.province_turn and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if provinces <= plan.duchy_at and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        if provinces <= plan.estate_at and coins >= 2 and Action("BUY", "Estate") in actions:
            return Action("BUY", "Estate")
        owned = Counter(me.all_cards())
        return next((Action("BUY", c) for c, n in plan.menu if owned[c] < n and Action("BUY", c) in actions), END_BUY)


def menu_options(kingdom: list[str]) -> list[str]:
    return list(kingdom) + list(MENU_BASICS)


def _random_count(rng: random.Random) -> int:
    return rng.choice((1, 1, 2, 2, 3, 4, 6, UNLIMITED))


def engine_plans(kingdom: list[str]) -> list[Plan]:
    """Village/draw engines (none without both), with Chapel, Throne Room and +Buy
    where available. Gold first, then a draw card and a Silver, then a village,
    Throne Room and +Buy card per further draw card, so the parts arrive together."""
    villages = [c for c in ("Village", "Festival") if c in kingdom]
    draws = [c for c in ("Witch", "Smithy", "Council Room", "Library", "Laboratory") if c in kingdom]
    if not (villages and draws):
        return []
    village, draw = villages[0], draws[0]
    extras = [c for c in ("Laboratory", "Market", "Festival") if c in kingdom and c not in (village, draw)]
    plans = []
    for levels, thrones, turn in ((3, 0, 5), (3, 1, 6), (4, 2, 8), (2, 0, 4)):
        menu = [("Gold", UNLIMITED)] + ([("Chapel", 1)] if "Chapel" in kingdom else []) + [(draw, 1), ("Silver", 1)]
        for i in range(1, levels + 1):
            menu += [(draw, i)] if i > 1 else []
            menu.append((village, i))
            menu += [("Throne Room", i)] if "Throne Room" in kingdom and i <= thrones else []
            menu += [(e, i - 1) for e in extras] if i > 1 else []
        plans.append(Plan(tuple(menu) + MONEY, province_turn=turn))
    return plans


def seed_plans(kingdom: list[str], rng: random.Random, n_random: int) -> list[Plan]:
    """Big Money, Big Money with one or two of each Action, engines, a Gardens rush, and random menus."""
    plans = [Plan(MONEY)]
    for card in kingdom:
        if card != "Gardens":
            plans += [Plan(((card, 1),) + MONEY), Plan(((card, 2),) + MONEY)]
    plans += engine_plans(kingdom)
    if "Gardens" in kingdom:
        gainers = [c for c in ("Workshop", "Artisan") if c in kingdom]
        plans.append(Plan(tuple((g, 8) for g in gainers) + (("Gardens", 8), ("Estate", 8), ("Copper", UNLIMITED)),
                          duchy_at=8, estate_at=8))
    options = menu_options(kingdom)
    for _ in range(n_random):
        menu = tuple((rng.choice(options), _random_count(rng)) for _ in range(rng.randint(2, 6)))
        plans.append(Plan(menu, province_turn=rng.randint(0, 12), duchy_at=rng.randint(0, 6),
                          estate_at=rng.randint(0, 3)))
    return list(dict.fromkeys(plans))


def mutate(plan: Plan, options: list[str], rng: random.Random) -> Plan:
    menu = list(plan.menu)
    for _ in range(rng.randint(1, 3)):
        op = rng.randrange(8)
        i = rng.randrange(len(menu))
        if op == 0:
            menu[i] = (rng.choice(options), menu[i][1])
        elif op == 1:
            menu[i] = (menu[i][0], _random_count(rng))
        elif op == 2 and len(menu) > 1:
            j = rng.randrange(len(menu))
            menu[i], menu[j] = menu[j], menu[i]
        elif op == 3 and len(menu) < 10:
            menu.insert(rng.randrange(len(menu) + 1), (rng.choice(options), _random_count(rng)))
        elif op == 4 and len(menu) > 1:
            del menu[i]
        elif op == 5:
            plan = replace(plan, province_turn=max(0, min(20, plan.province_turn + rng.choice((-3, -1, 1, 3)))))
        elif op == 6:
            plan = replace(plan, duchy_at=max(0, min(8, plan.duchy_at + rng.choice((-1, 1)))))
        else:
            plan = replace(plan, estate_at=max(0, min(8, plan.estate_at + rng.choice((-1, 1)))))
    return replace(plan, menu=tuple(menu))


def load_plans(path: str, min_score: float = 0.5) -> list[tuple[list[str], Plan]]:
    """(board, plan) for each board in an `--out` file whose best plan scored at least `min_score`."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        best = result["finalists"][0]
        if best["score"] >= min_score:
            out.append((result["kingdom"], Plan(**{**best["plan"], "menu": tuple(map(tuple, best["plan"]["menu"]))})))
    return out


def _score_chunk(opponent: str, jobs: list[tuple[Plan, list[str], list[int]]],
                 network_plays: bool = False) -> list[tuple[int, int, int]]:
    opp = load_agent(opponent)
    return [play_both_seats(PlanAgent(plan, opp if network_plays else None), opp, [(kingdom, s) for s in seeds])
            for plan, kingdom, seeds in jobs]


def score_plans(pool: ProcessPoolExecutor, workers: int, opponent: str, plans: list[Plan], kingdom: list[str],
                seeds: list[int], network_plays: bool = False) -> list[tuple[int, int, int]]:
    """Each plan's (wins, losses, ties) on the same `seeds`, spread over the workers."""
    jobs = [(p, kingdom, [s]) for p in plans for s in seeds]
    chunks = [jobs[i::workers] for i in range(workers) if jobs[i::workers]]
    results = pool.map(_score_chunk, [opponent] * len(chunks), chunks, [network_plays] * len(chunks))
    totals = [[0, 0, 0] for _ in plans]
    for c, chunk_results in enumerate(results):
        for k, r in enumerate(chunk_results):
            acc = totals[(c + k * workers) // len(seeds)]  # job c + k*workers went to chunk c
            for x in range(3):
                acc[x] += r[x]
    return [tuple(t) for t in totals]


def evolve(pool: ProcessPoolExecutor, workers: int, opponent: str, kingdom: list[str], rng: random.Random,
           generations: int = 20, population: int = 32, parents: int = 8, games: int = 12,
           final_top: int = 3, final_games: int = 200, network_plays: bool = False, log=print) -> dict:
    options = menu_options(kingdom)
    record: dict[Plan, list[int]] = {}
    pop = seed_plans(kingdom, rng, n_random=max(0, population - 2 * len(kingdom) - 3))[:population * 2]

    def mean(p: Plan) -> float:
        w, l, t = record[p]
        return (w + 0.5 * t) / (w + l + t)

    def shrunk(p: Plan, prior: float) -> float:  # one generation's games of the prior mixed in
        w, l, t = record[p]
        return (w + 0.5 * t + prior * 2 * games) / (w + l + t + 2 * games)

    for gen in range(generations):
        seeds = [rng.randrange(1_000_000) for _ in range(games)]
        for plan, r in zip(pop, score_plans(pool, workers, opponent, pop, kingdom, seeds, network_plays)):
            acc = record.setdefault(plan, [0, 0, 0])
            for x in range(3):
                acc[x] += r[x]
        prior = sum(mean(p) for p in pop) / len(pop)
        ranked = sorted(pop, key=lambda p: shrunk(p, prior), reverse=True)
        log(f"  gen {gen + 1:2d}: best {mean(ranked[0]):.1%} over {sum(record[ranked[0]])} games  "
            f"{ranked[0].describe()}")
        kept, children = ranked[:parents], []
        while len(children) < population - len(kept):
            child = mutate(rng.choice(kept), options, rng)
            if child not in kept and child not in children:
                children.append(child)
        pop = kept + children

    prior = sum(mean(p) for p in record) / len(record)
    finalists = sorted(record, key=lambda p: shrunk(p, prior), reverse=True)[:final_top]
    fresh_seeds = [rng.randrange(1_000_000) for _ in range(final_games)]
    results = []
    for plan, (w, l, t) in zip(finalists, score_plans(pool, workers, opponent, finalists, kingdom, fresh_seeds,
                                                      network_plays)):
        p, lo, hi = wilson(w, t, w + l + t)
        results.append({"plan": asdict(plan), "describe": plan.describe(), "wins": w, "losses": l, "ties": t,
                        "score": p, "ci": [lo, hi], "search_score": mean(plan)})
    results.sort(key=lambda r: -r["score"])
    return {"kingdom": kingdom, "opponent": opponent, "finalists": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("opponent", help="checkpoint path, or a scripted agent's name")
    parser.add_argument("--boards", type=int, default=1, help="random boards to search")
    parser.add_argument("--home", nargs="*", default=[], help="cards every random board must hold")
    parser.add_argument("--kingdom", nargs=10, default=None, help="search this one board instead")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--games", type=int, default=12, help="seeds per plan per generation, each from both seats")
    parser.add_argument("--final-games", type=int, default=200, help="fresh seeds to re-score the finalists on")
    parser.add_argument("--network-plays", action="store_true",
                        help="the checkpoint's policy makes every decision but the plan's buys")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="logs/domibot2/plan_search/results.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    boards = [list(args.kingdom)] if args.kingdom else [random_kingdom(rng, args.home) for _ in range(args.boards)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for b, kingdom in enumerate(boards):
            t0 = time.time()
            print(f"board {b + 1}/{len(boards)}: {', '.join(sorted(kingdom))}", flush=True)
            result = evolve(pool, args.workers, args.opponent, kingdom, rng, generations=args.generations,
                            population=args.population, parents=args.parents, games=args.games,
                            final_games=args.final_games, network_plays=args.network_plays,
                            log=lambda s: print(s, flush=True))
            for f in result["finalists"]:
                print(f"  fresh: {f['score']:.1%} [{f['ci'][0]:.0%}-{f['ci'][1]:.0%}] "
                      f"({f['wins']}-{f['losses']}-{f['ties']}; search said {f['search_score']:.0%})  {f['describe']}",
                      flush=True)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
