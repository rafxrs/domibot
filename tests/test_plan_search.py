import random
from concurrent.futures import ProcessPoolExecutor

from domibot import KINGDOM_CARDS

from training.agents import BigMoneyTerminalAgent
from training.evaluate import play_game
from training.plan_search import UNLIMITED, Plan, PlanAgent, evolve, menu_options, mutate, score_plans, seed_plans

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
