import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from domibot import KINGDOM_CARDS, Phase

from training.agents import BigMoneyTerminalAgent
from training.evaluate import play_game
from training.plan_search import (UNLIMITED, Plan, PlanAgent, engine_plans, evolve, load_plans, menu_options, mutate,
                                  score_plans, seed_plans)

MONEY = Plan((("Gold", UNLIMITED), ("Silver", UNLIMITED)))
NOTHING = Plan((("Copper", 0),), province_turn=99, duchy_at=-1, estate_at=-1)  # never buys anything


def test_random_plans_play_legal_games_on_random_kingdoms():
    rng = random.Random(0)
    for _ in range(6):
        kingdom = rng.sample(list(KINGDOM_CARDS), 10)
        for plan in rng.sample(seed_plans(kingdom, rng, n_random=10), 3):
            for players in ((PlanAgent(plan), BigMoneyTerminalAgent()), (BigMoneyTerminalAgent(), PlanAgent(plan))):
                assert play_game(*players, kingdom, rng.randrange(1000)).is_game_over()


def test_mutations_stay_within_the_board():
    rng = random.Random(1)
    kingdom = list(KINGDOM_CARDS)[:10]
    options = set(menu_options(kingdom))
    plan = MONEY
    for _ in range(300):
        plan = mutate(plan, sorted(options), rng)
        assert 1 <= len(plan.menu) <= 10
        assert all(card in options and count >= 1 for card, count in plan.menu)
        assert 0 <= plan.province_turn <= 20 and 0 <= plan.duchy_at <= 8 and 0 <= plan.estate_at <= 8


def test_scores_go_to_the_right_plans_and_the_search_runs():
    kingdom = list(KINGDOM_CARDS)[:10]
    with ProcessPoolExecutor(max_workers=2) as pool:
        (w1, l1, t1), (w2, l2, t2) = score_plans(pool, 2, "bigmoney", [MONEY, NOTHING], kingdom, [1, 2, 3])
        assert w1 + l1 + t1 == w2 + l2 + t2 == 6
        assert w1 > w2 == 0
        result = evolve(pool, 2, "bigmoney", kingdom, random.Random(2), generations=2, population=6, parents=2,
                        games=1, final_top=2, final_games=2, log=lambda s: None)
    assert result["kingdom"] == kingdom and len(result["finalists"]) == 2
    for f in result["finalists"]:
        assert f["wins"] + f["losses"] + f["ties"] == 4 and 0 <= f["score"] <= 1


class _Recorder:
    """An agent that records the decisions it's asked to make."""

    def __init__(self):
        self.agent, self.asked = BigMoneyTerminalAgent(), []

    def act(self, game):
        self.asked.append((game.phase, game.pending_decision))
        return self.agent.act(game)


def test_a_player_makes_every_decision_but_the_plans_buys():
    player = _Recorder()
    kingdom = list(KINGDOM_CARDS)[:10]
    game = play_game(PlanAgent(MONEY, player), BigMoneyTerminalAgent(), kingdom, 3)
    assert game.is_game_over() and game.players[0].all_cards().count("Gold") > 0
    assert player.asked and all(not (phase == Phase.BUY and pending is None) for phase, pending in player.asked)


def test_engine_plans_need_a_village_and_a_draw_card():
    assert engine_plans(["Cellar", "Chapel", "Moat", "Workshop", "Bureaucrat", "Gardens", "Militia", "Moneylender",
                         "Remodel", "Mine"]) == []
    plans = engine_plans(["Village", "Smithy", "Throne Room", "Chapel", "Market", "Cellar", "Moat", "Workshop",
                          "Gardens", "Mine"])
    assert plans and all(dict(p.menu)["Village"] >= 2 and "Smithy" in dict(p.menu) for p in plans)


def test_load_plans_keeps_each_boards_best_plan_if_it_scored_enough(tmp_path):
    kingdom = list(KINGDOM_CARDS)[:10]
    path = tmp_path / "results.jsonl"
    rows = [{"kingdom": kingdom, "finalists": [{"plan": asdict(plan), "score": score}]}
            for plan, score in ((Plan(((kingdom[0], 2),) + MONEY.menu, province_turn=9), 0.6), (MONEY, 0.4))]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert load_plans(str(path), 0.5) == [(kingdom, Plan(((kingdom[0], 2),) + MONEY.menu, province_turn=9))]
