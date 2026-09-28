from domibot import Game

from training.agents import BigMoneyTerminalAgent
from training.evaluate import play_game
from training.gauntlet import _play_chunk, home_kingdoms
from training.strategy_bots import ALL_BOTS, ThroneRoomEngineAgent
from training.strategy_profile import _profile_chunk


def test_every_bot_finishes_games_on_its_home_kingdoms():
    for bot_cls in ALL_BOTS:
        for kingdom, seed in home_kingdoms(bot_cls.HOME, 2, seed=1):
            assert set(bot_cls.HOME) <= set(kingdom) and len(set(kingdom)) == 10
            for players in ((bot_cls(), BigMoneyTerminalAgent()), (BigMoneyTerminalAgent(), bot_cls())):
                assert play_game(*players, kingdom, seed).is_game_over()


def test_throne_room_engine_chains_throne_rooms():
    # The bot the gauntlet uses to test play against Throne Room engines
    # must actually build one and play it.
    bot = ThroneRoomEngineAgent()
    targets = 0
    for kingdom, seed in home_kingdoms(bot.HOME, 4, seed=2):
        game = Game(kingdom, num_players=2, seed=seed)
        agents = (bot, BigMoneyTerminalAgent())
        while not game.is_game_over():
            action = agents[game.current_decider()].act(game)
            decision = game.pending_decision
            targets += (decision is not None and decision.source_card == "Throne Room"
                        and game.current_decider() == 0 and action.verb == "PLAY")
            game.step(action)
    assert targets >= 4


def test_gauntlet_and_profile_chunks_on_scripted_agents():
    games = home_kingdoms(("Workshop", "Gardens"), 2, seed=3)
    wins, losses, ties = _play_chunk("bigmoney_terminal", "workshop_gardens", games)
    assert wins + losses + ties == 4
    stats = _profile_chunk("throne_room_engine", home_kingdoms(ThroneRoomEngineAgent.HOME, 2, seed=4))
    assert stats["misc"]["games"] == 2
    assert stats["available"]["Throne Room"] == 4  # 2 games x 2 players
    assert stats["throne_targets"] and stats["plays"]["Smithy"] > 0
    assert sum(stats["turn_plays"].values()) > 0
