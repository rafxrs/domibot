import random

import torch

from training import encoding
from training.agents import BigMoneyAgent
from training.network import DomibotNet
from training.ppo.league import League, Opponent, collect_league_rollouts, load_league


def _net(extra: bool = True) -> DomibotNet:
    net = DomibotNet(hidden_dim=32, num_blocks=1, extra_dim=encoding.EXTRA_DIM if extra else 0)
    net.eval()
    return net


def test_league_rollouts_train_only_the_learners_seat():
    torch.manual_seed(0)
    learner = _net()
    scripted = Opponent("bigmoney", agent=BigMoneyAgent())
    older = Opponent("old", network=_net(extra=False))  # a base-encoding network in a full-encoding game
    opponents = [scripted, older, scripted, older]
    games, results = collect_league_rollouts(learner, opponents, max_moves=400, seed=3)
    assert len(games) == len(results) == 4
    assert all(r in (0.0, 0.5, 1.0) for r in results)
    for transitions in games:
        assert transitions, "the learner made no decisions"
        assert len({t.decider for t in transitions}) == 1  # one seat only
        assert all(t.obs.shape == (encoding.FULL_OBS_DIM,) for t in transitions)


def test_pfsp_weights_favor_the_opponents_the_learner_beats_least():
    easy, hard = Opponent("easy"), Opponent("hard")
    league = League([easy, hard], hard_power=2.0, uniform_mix=0.2)
    assert league.weights() == [0.5, 0.5]  # no games yet: both at 50%
    league.record(easy, [1.0] * 20)
    league.record(hard, [0.0] * 20)
    w_easy, w_hard = league.weights()
    assert w_hard > 0.8 and w_easy >= 0.1  # the uniform floor keeps the easy one in play
    assert abs(w_easy + w_hard - 1) < 1e-9
    league.end_iteration()
    assert easy.games == 20 * league.decay


def test_snapshots_are_frozen_copies_and_capped():
    learner = _net()
    league = League([Opponent("bigmoney", agent=BigMoneyAgent())], max_snapshots=2)
    for i in range(3):
        league.add_snapshot(learner, f"self@{i}")
    assert [o.name for o in league.opponents] == ["bigmoney", "self@1", "self@2"]
    snapshot = league.opponents[-1].network
    assert snapshot is not learner
    with torch.no_grad():
        for p in learner.parameters():
            p.add_(1.0)
    assert not torch.equal(next(snapshot.parameters()), next(learner.parameters()))
    assert len(league.sample(5, random.Random(0))) == 5


def test_load_league_expands_globs_and_adds_scripted(tmp_path):
    for i in (1, 2):
        _net().save(tmp_path / f"run_iter_{i}.pt")
    league = load_league([str(tmp_path / "run_iter_*.pt")], ["bigmoney_terminal"], torch.device("cpu"))
    assert [o.name for o in league.opponents] == ["run_iter_1", "run_iter_2", "bigmoney_terminal"]
    assert league.opponents[0].network is not None and league.opponents[2].agent is not None


def test_strategy_bots_play_only_on_kingdoms_holding_their_cards():
    torch.manual_seed(0)
    league = load_league([], ["workshop_gardens", "bigmoney"], torch.device("cpu"))
    bot, bigmoney = league.opponents
    assert bot.home == ("Workshop", "Gardens") and bigmoney.home == ()
    games, _results = collect_league_rollouts(_net(), [bot, bot, bot], max_moves=400, seed=5)
    for transitions in games:
        in_kingdom = transitions[0].obs[encoding.OBS_DIM:encoding.OBS_DIM + encoding.NUM_CARDS]
        assert in_kingdom[encoding.CARD_INDEX["Workshop"]] == 1 and in_kingdom[encoding.CARD_INDEX["Gardens"]] == 1
