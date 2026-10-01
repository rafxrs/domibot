"""Scripted bots for well-known strategies built on cards `domibot2.2` never bought.

The gauntlet and the league play each bot only on kingdoms holding its
`HOME` cards. Choices a strategy doesn't care about use `heuristic_reaction`.
"""
from __future__ import annotations

from collections import Counter

from domibot import Action, Game, Phase
from domibot.models import DONE, END_ACTIONS, END_BUY

from .agents import BigMoneyAgent, BigMoneyTerminalAgent
from .heuristics import heuristic_reaction

TREASURE = {"Copper": 1, "Silver": 2, "Gold": 3}


def _first_legal(actions: list[Action], candidates) -> Action | None:
    return next((a for a in candidates if a in actions), None)


def _buy_first(actions: list[Action], cards) -> Action:
    return _first_legal(actions, (Action("BUY", c) for c in cards)) or END_BUY


def is_junk(game: Game, me, card: str, money: dict[str, int], coppers_chosen: int = 0) -> bool:
    """Trash-worthy: a Curse; an Estate while more than 4 Provinces are left; a
    Copper while the deck keeps $6 of `money` without it."""
    if card == "Curse":
        return True
    if card == "Estate":
        return game.supply.get("Province", 0) > 4
    if card == "Copper":
        return sum(money.get(c, 0) for c in me.all_cards()) - coppers_chosen - 1 >= 6
    return False


def chapel_choice(game: Game, money: dict[str, int]) -> Action:
    """Chapel's next trash. It asks one card at a time and trashes at the end, so
    Coppers already picked this play (read from the log) still count as money."""
    actions, me = game.legal_actions(), game.players[game.current_decider()]
    chosen = 0
    for entry in reversed(game.action_log):
        if entry.action == Action("PLAY", "Chapel"):
            break
        chosen += entry.action == Action("TRASH", "Copper")
    for card in ("Curse", "Estate", "Copper"):
        if Action("TRASH", card) in actions and is_junk(game, me, card, money, chosen):
            return Action("TRASH", card)
    return DONE if DONE in actions else actions[-1]


def _greening(game: Game, coins: int) -> Action | None:
    """Province at $8; Duchy at $5+ once <=4 Provinces are left; Estate at $2+ at <=2."""
    provinces = game.supply.get("Province", 0)
    if coins >= 8:
        return Action("BUY", "Province")
    if provinces <= 4 and coins >= 5:
        return Action("BUY", "Duchy")
    if provinces <= 2 and coins >= 2:
        return Action("BUY", "Estate")
    return None


def _own_choice(game: Game, source: str) -> bool:
    """A choice raised by the current player's own `source` card."""
    d = game.pending_decision
    return d is not None and d.source_card == source and game.current_decider() == game.current_player


class WorkshopGardensAgent:
    """Workshop/Gardens rush: gain Workshops, then Gardens and Estates, and end the game on piles."""

    HOME = ("Workshop", "Gardens")
    name = "workshop_gardens"

    def act(self, game: Game) -> Action:
        actions, me = game.legal_actions(), game.players[game.current_decider()]
        if game.pending_decision is not None:
            if _own_choice(game, "Workshop"):
                early = me.all_cards().count("Workshop") < 8 and game.supply.get("Gardens", 0) > 4
                order = ["Workshop", "Gardens"] if early else ["Gardens", "Workshop"]
                pick = _first_legal(actions, (Action("GAIN", c) for c in order + ["Estate", "Silver"]))
                if pick is not None:
                    return pick
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            return Action("PLAY", "Workshop") if Action("PLAY", "Workshop") in actions else END_ACTIONS
        order = (["Province"] if me.coins >= 8 else []) + (["Gardens"] if me.coins >= 4 else [])
        return _buy_first(actions, order + ["Workshop", "Estate", "Copper"])


class ThroneRoomEngineAgent:
    """A Chapel/Village/Smithy/Market engine with Throne Rooms and one Militia.

    Opens Chapel/Silver (Market/Chapel on 5/2), trashes down, builds Smithies
    with Villages and Throne Rooms for actions, and greens once built. The
    weakest gauntlet bot: ~29% against BigMoney + terminal.
    """

    HOME = ("Chapel", "Throne Room", "Village", "Smithy", "Market", "Militia")
    name = "throne_room_engine"
    MONEY = {**TREASURE, "Market": 1}

    def act(self, game: Game) -> Action:
        actions, me = game.legal_actions(), game.players[game.current_decider()]
        if _own_choice(game, "Throne Room"):
            return self._throne_target(actions, me)
        if _own_choice(game, "Chapel"):
            return chapel_choice(game, self.MONEY)
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        return self._play(game, actions, me) if game.phase == Phase.ACTION else self._buy(game, actions, me)

    def _play(self, game: Game, actions: list[Action], me) -> Action:
        pick = _first_legal(actions, [Action("PLAY", "Market"), Action("PLAY", "Village")])
        if pick is not None:
            return pick
        junk = sum(is_junk(game, me, c, self.MONEY) for c in me.hand)
        if Action("PLAY", "Chapel") in actions and junk >= 3:
            return Action("PLAY", "Chapel")
        if Action("PLAY", "Throne Room") in actions and any(c in ("Smithy", "Village", "Market", "Militia")
                                                             for c in me.hand):
            return Action("PLAY", "Throne Room")
        pick = _first_legal(actions, [Action("PLAY", "Smithy"), Action("PLAY", "Militia")])
        if pick is not None:
            return pick
        return Action("PLAY", "Chapel") if Action("PLAY", "Chapel") in actions and junk else END_ACTIONS

    def _throne_target(self, actions: list[Action], me) -> Action:
        hand = Counter(me.hand)
        if me.actions == 0 and hand["Village"] and hand["Smithy"]:  # Village for actions, then the Smithy
            return Action("PLAY", "Village")
        others = hand["Smithy"] + hand["Village"] + hand["Market"]
        order = ["Smithy"] + (["Throne Room"] if others >= 2 else []) + ["Militia", "Market", "Village"]
        pick = _first_legal(actions, (Action("PLAY", c) for c in order))
        if pick is not None:
            return pick
        playable = [a for a in actions if a.verb == "PLAY" and a.card != "Chapel"]
        return playable[0] if playable else actions[-1]

    def _buy(self, game: Game, actions: list[Action], me) -> Action:
        coins, owned = me.coins, Counter(me.all_cards())
        smithies, villages, thrones = owned["Smithy"], owned["Village"], owned["Throne Room"]
        built = smithies + thrones >= 3 and villages + thrones >= 2
        if coins >= 8 and (built or me.turns_taken >= 12 or game.supply.get("Province", 0) <= 6):
            return _buy_first(actions, ["Province"])
        green = _greening(game, coins)
        if green is not None and green in actions and green.card != "Province":
            return green
        short_of_actions = smithies > villages + thrones // 2  # a Throne Room counts as half a Village
        order: list[str] = []
        if me.turns_taken < 2:
            order += (["Market"] if coins >= 5 else []) + (["Chapel"] if owned["Chapel"] == 0 else ["Silver"])
        if coins >= 6 and owned["Gold"] < 2:
            order.append("Gold")
        if short_of_actions:
            order.append("Village")
        if smithies < 2:
            order.append("Smithy")
        if owned["Militia"] == 0 and smithies >= 1:
            order.append("Militia")
        if thrones < 2 and smithies >= 1:
            order.append("Throne Room")
        if built:
            order += ["Gold", "Market"]
        if owned["Market"] < 2:
            order.append("Market")
        if smithies < 3 and not short_of_actions:
            order.append("Smithy")
        if villages < 2:
            order.append("Village")
        order += ["Gold", "Market"] + (["Silver"] if owned["Silver"] < 2 else [])
        return _buy_first(actions, [c for c in order if game.cards[c].cost <= coins])


class ChapelWitchAgent:
    """Chapel/Witch: open Chapel with Silver (or Witch), trash to a thin deck, add two Witches and Gold."""

    HOME = ("Chapel", "Witch")
    name = "chapel_witch"

    def act(self, game: Game) -> Action:
        actions, me = game.legal_actions(), game.players[game.current_decider()]
        if _own_choice(game, "Chapel"):
            return chapel_choice(game, TREASURE)
        if game.pending_decision is not None:
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            if Action("PLAY", "Witch") in actions:
                return Action("PLAY", "Witch")
            if Action("PLAY", "Chapel") in actions and any(is_junk(game, me, c, TREASURE) for c in me.hand):
                return Action("PLAY", "Chapel")
            return END_ACTIONS
        coins, owned = me.coins, Counter(me.all_cards())
        green = _greening(game, coins)
        if green is not None and green in actions and (coins < 8 or owned["Gold"] + owned["Witch"] >= 2):
            return green
        order = (["Witch"] if coins >= 5 and owned["Witch"] < 2 else []) + (["Gold"] if coins >= 6 else [])
        order += ["Province"] if coins >= 8 else []
        order += ["Chapel"] if 2 <= coins <= 4 and owned["Chapel"] == 0 and me.turns_taken < 2 else []
        return _buy_first(actions, order + (["Silver"] if coins >= 3 else []))


ALL_BOTS = (WorkshopGardensAgent, ThroneRoomEngineAgent, ChapelWitchAgent)
SCRIPTED = {"bigmoney": BigMoneyAgent, "bigmoney_terminal": BigMoneyTerminalAgent,
            **{bot.name: bot for bot in ALL_BOTS}}
