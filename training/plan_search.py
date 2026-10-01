"""Search for the best buy plan against a checkpoint on a given board, to
find out whether (and where) a better strategy than the network's exists.

A `Plan` is what the classic Dominion simulators and Provincial (an
evolutionary Dominion AI) call a buy menu: an ordered list of (card,
copies wanted), bought in order of priority whenever affordable, plus three
greening rules (when Provinces, Duchies and Estates take over). `PlanAgent`
follows a plan with fixed, generic rules for playing the cards: Actions
that give +Actions first, then Throne Room, then draw, then payload, with
card-specific rules for their choices (what Chapel trashes, what Throne
Room plays, what Workshop gains, ...).

With `--network-plays`, the checkpoint's own policy makes every decision
but the plan's buys, so the search asks where the network's buying is
wrong given its own play.

`evolve` searches plans for one board: a population of plans, each scored
against the opponent on the same fresh games every generation (both seats),
the best kept as parents, mutations of them filling the rest. Scores add
up across generations, so a plan kept for longer is judged on more games,
and plans are ranked with their score shrunk toward the population's by
one generation's worth of games, so a newcomer's lucky first generation
doesn't outrank a plan proven over many. The best few are then re-scored
on fresh games, since the search's own score for its winner is still
biased upward.

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

from domibot import KINGDOM_CARDS, Action, CardType, Game, Phase
from domibot.models import DONE, END_ACTIONS, END_BUY, NO, YES

from .evaluate import play_game, wilson
from .gauntlet import load_agent
from .heuristics import heuristic_reaction

UNLIMITED = 99
MONEY = (("Gold", UNLIMITED), ("Silver", UNLIMITED))
MENU_BASICS = ("Copper", "Silver", "Gold", "Estate", "Duchy")

# Play order: Actions that give +Actions first, then Throne Room, then
# terminal draw, then terminal payload.
NON_TERMINALS = ("Festival", "Village", "Laboratory", "Market", "Sentry", "Poacher", "Merchant", "Harbinger", "Cellar")
TERMINALS = ("Witch", "Council Room", "Library", "Smithy", "Moat", "Militia", "Bandit", "Artisan", "Mine",
             "Moneylender", "Bureaucrat", "Remodel", "Workshop", "Vassal", "Chapel")
VILLAGES = ("Village", "Festival")
# What Throne Room plays, best first.
THRONE_TARGETS = ("Smithy", "Council Room", "Witch", "Library", "Laboratory", "Market", "Festival", "Militia",
                  "Bandit", "Moneylender", "Mine", "Artisan", "Workshop", "Remodel", "Bureaucrat", "Sentry",
                  "Poacher", "Merchant", "Village", "Harbinger", "Cellar", "Moat", "Vassal", "Chapel")
JUNK = ("Curse", "Estate", "Duchy", "Province", "Gardens")


@dataclass(frozen=True)
class Plan:
    menu: tuple[tuple[str, int], ...]  # (card, copies wanted), highest priority first
    province_turn: int = 0  # buy Provinces with $8 only from this turn on (before it, $8 goes to the menu)
    duchy_at: int = 4  # buy a Duchy with $5+ once this few Provinces are left
    estate_at: int = 2  # buy an Estate with $2+ once this few Provinces are left

    def describe(self) -> str:
        menu = ", ".join(f"{c}{'' if n >= UNLIMITED else ' x' + str(n)}" for c, n in self.menu)
        return (f"[{menu}]  Provinces from turn {self.province_turn}, Duchy at <={self.duchy_at} "
                f"Provinces left, Estate at <={self.estate_at}")


class PlanAgent:
    """Plays `plan`'s buys, and everything else with `player` (any agent,
    e.g. a network's policy) if given, else with fixed, generic rules.
    Stateless, so one instance can play many games at once."""

    def __init__(self, plan: Plan, player=None):
        self.plan = plan
        self.player = player

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        me = game.players[game.current_decider()]
        decision = game.pending_decision
        if decision is None and game.phase == Phase.BUY:
            return self.buy(game, actions, me)
        if self.player is not None:
            return self.player.act(game)
        if decision is None:
            return self._play(game, actions, me)
        mine = game.current_decider() == game.current_player
        source = decision.source_card
        if mine and source == "Throne Room" and actions[0].verb == "PLAY":
            return self._throne_target(actions, me)
        if mine and source == "Chapel":
            return self._chapel(game, actions, me)
        if mine and source in ("Workshop", "Artisan", "Remodel") and actions[0].verb == "GAIN":
            return self._gain(game, actions, me)
        if mine and source == "Remodel" and actions[0].verb == "TRASH":
            return self._remodel_trash(game, actions)
        if mine and source == "Artisan" and actions[0].verb == "TOPDECK":
            return self._artisan_topdeck(game, actions, me)
        if mine and source == "Mine" and actions[0].verb == "TRASH":
            for card in ("Silver", "Copper"):
                if Action("TRASH", card) in actions:
                    return Action("TRASH", card)
        if mine and source == "Harbinger":
            options = [a for a in actions if a.card and a.card not in JUNK and a.card != "Copper"]
            return max(options, key=lambda a: game.cards[a.card].cost) if options else Action("NONE")
        if mine and source == "Cellar":
            junk = [a for a in actions if a.card in JUNK]
            return junk[0] if junk else DONE
        if mine and source == "Vassal":
            return YES
        if mine and source == "Library":
            return YES if me.actions > 0 else NO
        return heuristic_reaction(game)

    # ---- action phase ---------------------------------------------------
    def _play(self, game: Game, actions: list[Action], me) -> Action:
        for card in NON_TERMINALS:
            if Action("PLAY", card) in actions and (card != "Cellar" or any(c in JUNK for c in me.hand)):
                return Action("PLAY", card)
        others = [c for c in me.hand if c != "Throne Room" and CardType.ACTION in game.cards[c].types]
        if Action("PLAY", "Throne Room") in actions and others:
            return Action("PLAY", "Throne Room")
        for card in TERMINALS:
            if Action("PLAY", card) in actions and self._worth_playing(game, card, me):
                return Action("PLAY", card)
        return END_ACTIONS

    def _worth_playing(self, game: Game, card: str, me) -> bool:
        hand = me.hand
        if card == "Moneylender":
            return "Copper" in hand
        if card == "Mine":
            return "Copper" in hand or "Silver" in hand
        if card == "Remodel":
            return ("Gold" in hand and game.supply.get("Province", 0) <= 4) or "Curse" in hand or "Estate" in hand
        if card == "Chapel":
            return any(self._chapel_junk(game, me, c, 0) for c in hand)
        if card == "Library":
            return len(hand) <= 5
        return True

    def _throne_target(self, actions: list[Action], me) -> Action:
        hand = Counter(me.hand)
        terminals = sum(n for c, n in hand.items() if c in TERMINALS)
        if me.actions == 0 and terminals:
            for village in VILLAGES:
                if Action("PLAY", village) in actions:
                    return Action("PLAY", village)
        others = sum(n for c, n in hand.items() if c != "Throne Room" and (c in TERMINALS or c in NON_TERMINALS))
        order = (("Throne Room",) if others >= 2 else ()) + THRONE_TARGETS
        for card in order:
            if Action("PLAY", card) in actions:
                return Action("PLAY", card)
        return actions[0]

    # ---- card choices ---------------------------------------------------
    @staticmethod
    def _money(me) -> int:
        values = {"Copper": 1, "Silver": 2, "Gold": 3, "Market": 1, "Festival": 2}
        return sum(values.get(c, 0) for c in me.all_cards())

    def _chapel_junk(self, game: Game, me, card: str, coppers_chosen: int) -> bool:
        if card == "Curse":
            return True
        if card == "Estate":
            return game.supply.get("Province", 0) > 4
        if card == "Copper":
            return self._money(me) - coppers_chosen - 1 >= 6
        return False

    def _chapel(self, game: Game, actions: list[Action], me) -> Action:
        chosen = 0  # Coppers picked so far in this Chapel: still in hand until it resolves
        for entry in reversed(game.action_log):
            if entry.action == Action("PLAY", "Chapel"):
                break
            chosen += entry.action == Action("TRASH", "Copper")
        for card in ("Curse", "Estate", "Copper"):
            if Action("TRASH", card) in actions and self._chapel_junk(game, me, card, chosen):
                return Action("TRASH", card)
        return DONE if DONE in actions else actions[-1]

    def _gain(self, game: Game, actions: list[Action], me) -> Action:
        """The plan's first card still wanted, if on offer; else the most
        expensive card that isn't a Curse."""
        owned = Counter(me.all_cards())
        for card, wanted in self.plan.menu:
            if owned[card] < wanted and Action("GAIN", card) in actions:
                return Action("GAIN", card)
        options = [a for a in actions if a.card and a.card != "Curse"] or actions
        return max(options, key=lambda a: game.cards[a.card].cost if a.card else -1)

    def _remodel_trash(self, game: Game, actions: list[Action]) -> Action:
        order = (("Gold",) if game.supply.get("Province", 0) <= 4 else ()) + ("Curse", "Estate", "Copper")
        for card in order:
            if Action("TRASH", card) in actions:
                return Action("TRASH", card)
        return min(actions, key=lambda a: game.cards[a.card].cost if a.card else 99)

    def _artisan_topdeck(self, game: Game, actions: list[Action], me) -> Action:
        if me.actions == 0:  # an Action it can't play this turn is best drawn next turn
            playable = [a for a in actions if a.card and CardType.ACTION in game.cards[a.card].types]
            if playable:
                return max(playable, key=lambda a: game.cards[a.card].cost)
        return heuristic_reaction(game)

    # ---- buy phase ------------------------------------------------------
    def buy(self, game: Game, actions: list[Action], me) -> Action:
        plan, coins = self.plan, me.coins
        provinces = game.supply.get("Province", 0)
        if coins >= 8 and me.turns_taken >= plan.province_turn and Action("BUY", "Province") in actions:
            return Action("BUY", "Province")
        if provinces <= plan.duchy_at and coins >= 5 and Action("BUY", "Duchy") in actions:
            return Action("BUY", "Duchy")
        if provinces <= plan.estate_at and coins >= 2 and Action("BUY", "Estate") in actions:
            return Action("BUY", "Estate")
        owned = Counter(me.all_cards())
        for card, wanted in plan.menu:
            if owned[card] < wanted and Action("BUY", card) in actions:
                return Action("BUY", card)
        return END_BUY


# ---- search -------------------------------------------------------------
def menu_options(kingdom: list[str]) -> list[str]:
    return list(kingdom) + list(MENU_BASICS)


def _random_count(rng: random.Random) -> int:
    return rng.choice((1, 1, 2, 2, 3, 4, 6, UNLIMITED))


def engine_plans(kingdom: list[str]) -> list[Plan]:
    """Village/draw engines, with Chapel, Throne Room and +Buy cards where
    the board has them; none without a village and a draw card. Gold comes
    first, then the draw card and a Silver, then a village, a Throne Room
    and a +Buy card for each further draw card, so the parts arrive
    together. (Villages bought first, and no money, won 1% against
    `domibot2.3`; these win about 11%.)"""
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
    """Big Money, Big Money with one or two of each Action, engines if the
    board has the parts, a Gardens rush, and random menus."""
    money = MONEY
    plans = [Plan(money)]
    actions = [c for c in kingdom if c != "Gardens"]
    for card in actions:
        plans.append(Plan(((card, 1),) + money))
        plans.append(Plan(((card, 2),) + money))
    plans += engine_plans(kingdom)
    if "Gardens" in kingdom:
        gainer = [c for c in ("Workshop", "Artisan") if c in kingdom]
        plans.append(Plan(tuple((g, 8) for g in gainer) + (("Gardens", 8), ("Estate", 8), ("Copper", UNLIMITED)),
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
    """(board, plan) for each board in a search's `--out` file whose best
    plan scored at least `min_score` on fresh games."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        best = result["finalists"][0]
        if best["score"] >= min_score:
            plan = Plan(**{**best["plan"], "menu": tuple(tuple(m) for m in best["plan"]["menu"])})
            out.append((result["kingdom"], plan))
    return out


def _score_chunk(opponent: str, jobs: list[tuple[Plan, list[str], list[int]]],
                 network_plays: bool = False) -> list[tuple[int, int, int]]:
    """(wins, losses, ties) of each plan against `opponent`, every seed
    played from both seats; with `network_plays`, the opponent's policy
    plays the plan's cards."""
    opp = load_agent(opponent)
    out = []
    for plan, kingdom, seeds in jobs:
        agent = PlanAgent(plan, opp if network_plays else None)
        w = l = t = 0
        for seed in seeds:
            for seat in (0, 1):
                players = (agent, opp) if seat == 0 else (opp, agent)
                winners = play_game(players[0], players[1], kingdom, seed).winners()
                if len(winners) != 1:
                    t += 1
                elif winners[0] == seat:
                    w += 1
                else:
                    l += 1
        out.append((w, l, t))
    return out


def score_plans(pool: ProcessPoolExecutor, workers: int, opponent: str, plans: list[Plan], kingdom: list[str],
                seeds: list[int], network_plays: bool = False) -> list[tuple[int, int, int]]:
    """Every plan on the same `seeds`, spread over the workers."""
    jobs = [(p, kingdom, [s]) for p in plans for s in seeds]
    chunks = [jobs[i::workers] for i in range(workers) if jobs[i::workers]]
    results = list(pool.map(_score_chunk, [opponent] * len(chunks), chunks, [network_plays] * len(chunks)))
    totals: dict[int, list[int]] = {}
    for c, chunk_results in enumerate(results):
        for k, r in enumerate(chunk_results):
            plan_idx = (c + k * workers) // len(seeds)
            acc = totals.setdefault(plan_idx, [0, 0, 0])
            for x in range(3):
                acc[x] += r[x]
    return [tuple(totals[i]) for i in range(len(plans))]


def evolve(pool: ProcessPoolExecutor, workers: int, opponent: str, kingdom: list[str], rng: random.Random,
           generations: int = 20, population: int = 32, parents: int = 8, games: int = 12,
           final_top: int = 3, final_games: int = 200, network_plays: bool = False, log=print) -> dict:
    options = menu_options(kingdom)
    record: dict[Plan, list[int]] = {}
    pop = seed_plans(kingdom, rng, n_random=max(0, population - 2 * len(kingdom) - 3))[:population * 2]

    def mean(p: Plan) -> float:
        w, l, t = record[p]
        return (w + 0.5 * t) / (w + l + t)

    def shrunk(p: Plan, prior: float) -> float:
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
        best = ranked[0]
        log(f"  gen {gen + 1:2d}: best {mean(best):.1%} over {sum(record[best])} games  {best.describe()}")
        kept = ranked[:parents]
        children: list[Plan] = []
        while len(children) < population - len(kept):
            child = mutate(rng.choice(kept), options, rng)
            if child not in kept and child not in children:
                children.append(child)
        pop = kept + children

    prior = sum(mean(p) for p in record) / len(record)
    finalists = sorted(record, key=lambda p: shrunk(p, prior), reverse=True)[:final_top]
    fresh_seeds = [rng.randrange(1_000_000) for _ in range(final_games)]
    results = []
    scores = score_plans(pool, workers, opponent, finalists, kingdom, fresh_seeds, network_plays)
    for plan, (w, l, t) in zip(finalists, scores):
        p, lo, hi = wilson(w, t, w + l + t)
        results.append({"plan": asdict(plan), "describe": plan.describe(), "wins": w, "losses": l, "ties": t,
                        "score": p, "ci": [lo, hi], "search_score": mean(plan)})
    results.sort(key=lambda r: -r["score"])
    return {"kingdom": kingdom, "opponent": opponent, "finalists": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("opponent", help="checkpoint path, or a scripted agent's name (see gauntlet.SCRIPTED)")
    parser.add_argument("--boards", type=int, default=1, help="random boards to search")
    parser.add_argument("--home", nargs="*", default=[], help="cards every random board must hold")
    parser.add_argument("--kingdom", nargs=10, default=None, help="search this one board instead")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--games", type=int, default=12, help="seeds per plan per generation, each from both seats")
    parser.add_argument("--final-games", type=int, default=200,
                        help="fresh seeds (each from both seats) to re-score the best plans on")
    parser.add_argument("--network-plays", action="store_true",
                        help="the opponent checkpoint's policy makes every decision but the plan's buys")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="logs/domibot2/plan_search/results.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.kingdom:
        boards = [list(args.kingdom)]
    else:
        rest = [c for c in KINGDOM_CARDS if c not in args.home]
        boards = [list(args.home) + rng.sample(rest, 10 - len(args.home)) for _ in range(args.boards)]
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
