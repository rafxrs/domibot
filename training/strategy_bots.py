"""Scripted bots for specific, well-known base-set strategies, each built
around cards `domibot2.2` never buys (see training/README.md, "Card use").
They're the gauntlet's opponents (`training/gauntlet.py`): self-play evals
and the BigMoney baselines can't show whether a network handles a strategy
it never plays itself, and elite players pick the best strategy for each
kingdom.

Each bot declares `HOME`, the cards its strategy needs; the gauntlet plays
it only on kingdoms that contain them. The rules are simple priority lists
in the style of the classic Dominion simulators, not tuned: a bot scoring
well against a network shows a gap, one scoring badly doesn't prove there
isn't one. Sub-decisions the strategy doesn't care about use
`heuristic_reaction`, as the BigMoney agents do.
"""
from __future__ import annotations

from collections import Counter

from domibot import Action, CardType, Game, Phase
from domibot.models import DONE, END_ACTIONS, END_BUY

from .heuristics import heuristic_reaction


def _me(game: Game):
    return game.players[game.current_decider()]


def _first_legal(actions: list[Action], candidates) -> Action | None:
    for a in candidates:
        if a in actions:
            return a
    return None


def _coppers_chosen_for_chapel(game: Game) -> int:
    """Coppers already picked during the Chapel resolving now. Chapel asks
    for one card at a time and trashes them all at the end, so they still
    count toward the deck's money until then. Read from the game's log
    rather than kept on the bot, so one bot can play many games at once."""
    chosen = 0
    for entry in reversed(game.action_log):
        if entry.action == Action("PLAY", "Chapel"):
            break
        chosen += entry.action == Action("TRASH", "Copper")
    return chosen


def _greening(game: Game, coins: int) -> Action | None:
    """Province at 8; Duchy at 5+ once 4 or fewer Provinces are left; Estate
    at 2+ once 2 or fewer are."""
    provinces = game.supply.get("Province", 0)
    if coins >= 8:
        return Action("BUY", "Province")
    if provinces <= 4 and coins >= 5:
        return Action("BUY", "Duchy")
    if provinces <= 2 and coins >= 2:
        return Action("BUY", "Estate")
    return None


class WorkshopGardensAgent:
    """The Workshop/Gardens rush: fill the deck with Workshops, then use them
    and every buy to gain Gardens (1 VP per 10 cards owned) and Estates,
    aiming to end the game on piles before an opponent's economy pays off."""

    HOME = ("Workshop", "Gardens")
    name = "workshop_gardens"

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        me = _me(game)
        decision = game.pending_decision
        if decision is not None:
            if decision.source_card == "Workshop" and game.current_decider() == game.current_player:
                pick = _first_legal(actions, (Action("GAIN", c) for c in self._gain_order(game, me)))
                if pick is not None:
                    return pick
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            return Action("PLAY", "Workshop") if Action("PLAY", "Workshop") in actions else END_ACTIONS
        order = []
        if me.coins >= 8:
            order.append("Province")
        if me.coins >= 4:
            order.append("Gardens")
        order += ["Workshop", "Estate", "Copper"]
        return _first_legal(actions, (Action("BUY", c) for c in order)) or END_BUY

    def _gain_order(self, game: Game, me) -> list[str]:
        if me.all_cards().count("Workshop") < 8 and game.supply.get("Gardens", 0) > 4:
            return ["Workshop", "Gardens", "Estate", "Silver"]
        return ["Gardens", "Workshop", "Estate", "Silver"]


class ThroneRoomEngineAgent:
    """A Chapel/Village/Smithy/Market engine with Throne Rooms and a Militia,
    the kind of deck strong players build on a board like this: Chapel away
    the Estates and Coppers, add Smithies (Throne-Roomed for +6 cards),
    Villages and Throne Rooms for actions, Markets for +Buy, one Militia to
    slow the opponent, and Gold; start greening once the deck draws itself
    (or the opponent has started on the Provinces). Opens Chapel/Silver
    (Market/Chapel on a 5/2 split).

    Play order: Market, Village, Chapel while the hand holds 3+ junk cards,
    Throne Room (on Smithy; on another Throne Room with two more Actions in
    hand; on Village when out of actions with a Smithy still to play), then
    Smithy, Militia, and Chapel on any junk left.

    The weakest of the gauntlet's bots: it scores ~29% against
    BigMoney+terminal, which plays Big Money + Smithy on this board."""

    HOME = ("Chapel", "Throne Room", "Village", "Smithy", "Market", "Militia")
    name = "throne_room_engine"

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        me = _me(game)
        decision = game.pending_decision
        if decision is not None and game.current_decider() == game.current_player:
            if decision.source_card == "Throne Room":
                return self._throne_target(actions, me)
            if decision.source_card == "Chapel":
                return self._chapel_choice(game, actions, me)
        if decision is not None:
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            return self._play(game, actions, me)
        return self._buy(game, actions, me)

    def _play(self, game: Game, actions: list[Action], me) -> Action:
        for card in ("Market", "Village"):
            if Action("PLAY", card) in actions:
                return Action("PLAY", card)
        junk = sum(self._is_junk(game, me, c, 0) for c in me.hand)
        if Action("PLAY", "Chapel") in actions and junk >= 3:
            return Action("PLAY", "Chapel")
        others = [c for c in me.hand if c in ("Smithy", "Village", "Market", "Militia")]
        if Action("PLAY", "Throne Room") in actions and others:
            return Action("PLAY", "Throne Room")
        for card in ("Smithy", "Militia"):
            if Action("PLAY", card) in actions:
                return Action("PLAY", card)
        if Action("PLAY", "Chapel") in actions and junk:
            return Action("PLAY", "Chapel")
        return END_ACTIONS

    def _throne_target(self, actions: list[Action], me) -> Action:
        hand = Counter(me.hand)
        other_actions = sum(n for c, n in hand.items() if c in ("Smithy", "Village", "Market"))
        if me.actions == 0 and hand["Village"] and hand["Smithy"]:
            return Action("PLAY", "Village")
        order = ["Smithy"]
        if other_actions >= 2:
            order.append("Throne Room")
        order += ["Militia", "Market", "Village"]
        pick = _first_legal(actions, (Action("PLAY", c) for c in order))
        if pick is not None:
            return pick
        playable = [a for a in actions if a.verb == "PLAY" and a.card != "Chapel"]
        return playable[0] if playable else actions[-1]

    @staticmethod
    def _money(me) -> int:
        values = {"Copper": 1, "Silver": 2, "Gold": 3, "Market": 1}
        return sum(values.get(c, 0) for c in me.all_cards())

    def _is_junk(self, game: Game, me, card: str, coppers_chosen: int) -> bool:
        if card == "Curse":
            return True
        if card == "Estate":
            return game.supply.get("Province", 0) > 4
        if card == "Copper":
            return self._money(me) - coppers_chosen - 1 >= 6
        return False

    def _chapel_choice(self, game: Game, actions: list[Action], me) -> Action:
        chosen = _coppers_chosen_for_chapel(game)
        for card in ("Curse", "Estate", "Copper"):
            a = Action("TRASH", card)
            if a in actions and self._is_junk(game, me, card, chosen):
                return a
        return DONE if DONE in actions else actions[-1]

    def _buy(self, game: Game, actions: list[Action], me) -> Action:
        coins = me.coins
        owned = Counter(me.all_cards())
        smithies, villages, thrones = owned["Smithy"], owned["Village"], owned["Throne Room"]
        built = smithies + thrones >= 3 and villages + thrones >= 2
        provinces = game.supply.get("Province", 0)
        if coins >= 8 and (built or me.turns_taken >= 12 or provinces <= 6):
            return _first_legal(actions, [Action("BUY", "Province")]) or END_BUY
        green = _greening(game, coins)
        if green is not None and green in actions and green.card != "Province":
            return green
        # Terminal draw wants one action each; Villages supply them, and a
        # Throne Room either doubles a Smithy or acts as a Village.
        short_of_actions = smithies > villages + thrones // 2
        order: list[str] = []
        if me.turns_taken < 2:
            order += ["Market"] if coins >= 5 else []
            order += ["Chapel"] if owned["Chapel"] == 0 else ["Silver"]
        if coins >= 6 and owned["Gold"] < 2:
            order.append("Gold")  # early money
        if short_of_actions:
            order.append("Village")
        if smithies < 2:
            order.append("Smithy")
        if owned["Militia"] == 0 and smithies >= 1:
            order.append("Militia")
        if thrones < 2 and smithies >= 1:
            order.append("Throne Room")
        if built:
            order += ["Gold", "Market"]  # the deck draws itself: add payload
        if owned["Market"] < 2:
            order.append("Market")
        if smithies < 3 and not short_of_actions:
            order.append("Smithy")
        if villages < 2:
            order.append("Village")
        order += ["Gold", "Market"]
        if owned["Silver"] < 2:
            order.append("Silver")
        # only what this hand can afford, in that order
        order = [c for c in order if game.cards[c].cost <= coins]
        return _first_legal(actions, (Action("BUY", c) for c in order)) or END_BUY


class ChapelWitchAgent:
    """Chapel/Witch: open Chapel with Silver (or Witch), trash Estates,
    Curses and Coppers down to a thin deck while keeping enough money to
    hit $5-6, and add two Witches and Gold."""

    HOME = ("Chapel", "Witch")
    name = "chapel_witch"

    def act(self, game: Game) -> Action:
        actions = game.legal_actions()
        me = _me(game)
        decision = game.pending_decision
        if decision is not None:
            if decision.source_card == "Chapel" and game.current_decider() == game.current_player:
                return self._chapel_choice(game, actions, me)
            return heuristic_reaction(game)
        if game.phase == Phase.ACTION:
            if Action("PLAY", "Witch") in actions:
                return Action("PLAY", "Witch")
            if Action("PLAY", "Chapel") in actions and self._junk_in_hand(game, me):
                return Action("PLAY", "Chapel")
            return END_ACTIONS
        return self._buy(game, actions, me)

    @staticmethod
    def _money(me, cards=None) -> int:
        values = {"Copper": 1, "Silver": 2, "Gold": 3}
        return sum(values.get(c, 0) for c in (cards if cards is not None else me.all_cards()))

    def _junk_in_hand(self, game: Game, me) -> bool:
        return any(self._is_junk(game, me, c, 0) for c in me.hand)

    def _is_junk(self, game: Game, me, card: str, coppers_trashing: int) -> bool:
        if card == "Curse":
            return True
        if card == "Estate":
            return game.supply.get("Province", 0) > 4
        if card == "Copper":
            return self._money(me) - coppers_trashing - 1 >= 6
        return False

    def _chapel_choice(self, game: Game, actions: list[Action], me) -> Action:
        chosen = _coppers_chosen_for_chapel(game)
        for card in ("Curse", "Estate", "Copper"):
            a = Action("TRASH", card)
            if a in actions and self._is_junk(game, me, card, chosen):
                return a
        return DONE if DONE in actions else actions[-1]

    def _buy(self, game: Game, actions: list[Action], me) -> Action:
        coins = me.coins
        owned = Counter(me.all_cards())
        green = _greening(game, coins)
        if green is not None and green in actions and (coins < 8 or owned["Gold"] + owned["Witch"] >= 2):
            return green
        order: list[str] = []
        if coins >= 5 and owned["Witch"] < 2:
            order.append("Witch")
        if coins >= 6:
            order.append("Gold")
        if coins >= 8:
            order.append("Province")
        if 2 <= coins <= 4 and owned["Chapel"] == 0 and me.turns_taken < 2:
            order.append("Chapel")
        if coins >= 3:
            order.append("Silver")
        return _first_legal(actions, (Action("BUY", c) for c in order)) or END_BUY


ALL_BOTS = (WorkshopGardensAgent, ThroneRoomEngineAgent, ChapelWitchAgent)
