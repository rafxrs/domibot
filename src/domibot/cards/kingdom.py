"""The 26 base-set kingdom cards.

An effect is a plain function `(game, player_idx)` when it never asks for a
decision, or a generator yielding `Decision`s when it does.
"""
from __future__ import annotations

from ..card import Card
from ..effects import attack_each_opponent, choose_cards, choose_from_supply, choose_one, gain, move, peek_top, \
    trash_from, yes_no
from ..enums import CardType

ACTION = (CardType.ACTION,)
ACTION_ATTACK = (CardType.ACTION, CardType.ATTACK)
ACTION_REACTION = (CardType.ACTION, CardType.REACTION)


def _reveal(game, player, n: int) -> list[str]:
    """Take up to `n` cards off the deck into set-aside (a real zone while a choice is pending)."""
    revealed = []
    for _ in range(n):
        top = peek_top(game, player)
        if top is None:
            break
        player.deck.pop()
        revealed.append(top)
        player.set_aside.append(top)
    return revealed


def cellar_effect(game, p):
    player = game.players[p]
    discarded = yield from choose_cards(game, p, player.hand, "DISCARD",
                                        "Cellar: discard any number of cards, then draw that many.", min_count=0)
    for c in discarded:
        move(c, player.hand, player.discard)
    player.draw(len(discarded), game.rng)


def chapel_effect(game, p):
    player = game.players[p]
    to_trash = yield from choose_cards(game, p, player.hand, "TRASH", "Chapel: trash up to 4 cards from your hand.",
                                       min_count=0, max_count=4)
    for c in to_trash:
        trash_from(game, c, player.hand)


def harbinger_effect(game, p):
    player = game.players[p]
    if not player.discard:
        return
    choice = yield from choose_one(game, p, player.discard, "TOPDECK",
                                   "Harbinger: put a card from your discard pile onto your deck?", allow_none=True)
    if choice:
        move(choice, player.discard, player.deck)


def merchant_effect(game, p):
    game.turn_merchant_bonus += 1


def vassal_effect(game, p):
    player = game.players[p]
    top = peek_top(game, player)
    if top is None:
        return
    player.deck.pop()
    player.set_aside.append(top)  # a real zone while the choice is pending
    if CardType.ACTION in game.cards[top].types:
        play_it = yield from yes_no(p, f"Vassal discarded {top}. Play it?")
        if play_it:
            player.set_aside.remove(top)
            player.play_area.append(top)
            yield from game.resolve_action(p, top)
            return
    player.set_aside.remove(top)
    player.discard.append(top)


def workshop_effect(game, p):
    choice = yield from choose_from_supply(game, p, "Workshop: gain a card costing up to 4.", "GAIN",
                                           lambda c: c.cost <= 4, allow_none=False)
    if choice:
        gain(game, p, choice, to="discard")


def _bureaucrat_hit(game, opp):
    player = game.players[opp]
    victories = [c for c in player.hand if CardType.VICTORY in game.cards[c].types]
    if not victories:
        return
    choice = yield from choose_one(game, opp, victories, "TOPDECK",
                                   "Bureaucrat: put a Victory card from your hand onto your deck.", allow_none=False)
    if choice:
        move(choice, player.hand, player.deck)


def bureaucrat_effect(game, p):
    gain(game, p, "Silver", to="deck_top")
    yield from attack_each_opponent(game, p, _bureaucrat_hit)


def _gardens_vp(player) -> int:
    return len(player.all_cards()) // 10


def _militia_hit(game, opp):
    player = game.players[opp]
    n = len(player.hand) - 3
    if n <= 0:
        return
    chosen = yield from choose_cards(game, opp, player.hand, "DISCARD",
                                     f"Militia: discard down to 3 cards in hand ({n} to discard).",
                                     min_count=n, max_count=n)
    for c in chosen:
        move(c, player.hand, player.discard)


def militia_effect(game, p):
    yield from attack_each_opponent(game, p, _militia_hit)


def moneylender_effect(game, p):
    # Trashing a Copper isn't optional ("trash a Copper... if you do"), so there's no choice to ask.
    player = game.players[p]
    if "Copper" in player.hand:
        trash_from(game, "Copper", player.hand)
        player.coins += 3


def poacher_effect(game, p):
    player = game.players[p]
    n = min(sum(1 for count in game.supply.values() if count == 0), len(player.hand))
    if n <= 0:
        return
    chosen = yield from choose_cards(game, p, player.hand, "DISCARD",
                                     f"Poacher: discard {n} card(s), one per empty supply pile.",
                                     min_count=n, max_count=n)
    for c in chosen:
        move(c, player.hand, player.discard)


def remodel_effect(game, p):
    player = game.players[p]
    if not player.hand:
        return
    to_trash = yield from choose_one(game, p, player.hand, "TRASH", "Remodel: trash a card from your hand.",
                                     allow_none=False)
    if not to_trash:
        return
    max_cost = game.cards[to_trash].cost + 2
    trash_from(game, to_trash, player.hand)
    choice = yield from choose_from_supply(game, p, f"Remodel: gain a card costing up to {max_cost}.", "GAIN",
                                           lambda c: c.cost <= max_cost, allow_none=False)
    if choice:
        gain(game, p, choice, to="discard")


def throne_room_effect(game, p):
    player = game.players[p]
    action_cards = [c for c in player.hand if CardType.ACTION in game.cards[c].types]
    if not action_cards:
        return
    choice = yield from choose_one(game, p, action_cards, "PLAY",
                                   "Throne Room: play an Action card from your hand twice.", allow_none=True)
    if not choice:
        return
    move(choice, player.hand, player.play_area)
    for _ in range(2):
        yield from game.resolve_action(p, choice)


def _bandit_hit(game, opp):
    player = game.players[opp]
    revealed = _reveal(game, player, 2)
    targets = [c for c in revealed if CardType.TREASURE in game.cards[c].types and c != "Copper"]
    if targets:  # asked even with one target, so the trash shows up as a logged action
        choice = yield from choose_one(game, opp, targets, "TRASH", "Bandit: choose a Treasure to trash.",
                                       allow_none=False)
        revealed.remove(choice)
        player.set_aside.remove(choice)
        game.trash.append(choice)
    for c in revealed:
        player.set_aside.remove(c)
        player.discard.append(c)


def bandit_effect(game, p):
    gain(game, p, "Gold", to="discard")
    yield from attack_each_opponent(game, p, _bandit_hit)


def council_room_effect(game, p):
    for opp in game.other_players_in_order(p):
        game.players[opp].draw(1, game.rng)


def library_effect(game, p):
    player = game.players[p]
    skipped: list[str] = []
    while len(player.hand) < 7:
        top = peek_top(game, player)
        if top is None:
            break
        player.deck.pop()
        player.set_aside.append(top)
        if CardType.ACTION in game.cards[top].types:
            keep = yield from yes_no(p, f"Library: draw {top} into your hand? (No sets it aside instead)")
            if not keep:
                skipped.append(top)
                continue
        player.set_aside.remove(top)
        player.hand.append(top)
    for c in skipped:
        player.set_aside.remove(c)
        player.discard.append(c)


def mine_effect(game, p):
    player = game.players[p]
    treasures = [c for c in player.hand if CardType.TREASURE in game.cards[c].types]
    if not treasures:
        return
    to_trash = yield from choose_one(game, p, treasures, "TRASH", "Mine: trash a Treasure from your hand?",
                                     allow_none=True)
    if not to_trash:
        return
    max_cost = game.cards[to_trash].cost + 3
    trash_from(game, to_trash, player.hand)
    choice = yield from choose_from_supply(game, p, f"Mine: gain a Treasure costing up to {max_cost} to your hand.",
                                           "GAIN", lambda c: CardType.TREASURE in c.types and c.cost <= max_cost,
                                           allow_none=False)
    if choice:
        gain(game, p, choice, to="hand")


def sentry_effect(game, p):
    player = game.players[p]
    remaining = _reveal(game, player, 2)
    if not remaining:
        return
    to_trash = yield from choose_cards(game, p, remaining, "TRASH", "Sentry: trash any of the two revealed cards.",
                                       min_count=0)
    for c in to_trash:
        remaining.remove(c)
        player.set_aside.remove(c)
        game.trash.append(c)
    to_discard = yield from choose_cards(game, p, remaining, "DISCARD",
                                         "Sentry: discard any of the remaining revealed cards.", min_count=0)
    for c in to_discard:
        remaining.remove(c)
        player.set_aside.remove(c)
        player.discard.append(c)
    order: list[str] = []
    while remaining:  # the cards stay set aside until the whole order is chosen
        choice = yield from choose_one(
            game, p, remaining, "TOPDECK",
            "Sentry: pick the next card to return to your deck (the last one you pick ends up on top).",
            allow_none=False)
        remaining.remove(choice)
        order.append(choice)
    for c in order:
        player.set_aside.remove(c)
        player.deck.append(c)


def _witch_hit(game, opp):
    gain(game, opp, "Curse", to="discard")


def witch_effect(game, p):
    yield from attack_each_opponent(game, p, _witch_hit)


def artisan_effect(game, p):
    player = game.players[p]
    choice = yield from choose_from_supply(game, p, "Artisan: gain a card to your hand costing up to 5.", "GAIN",
                                           lambda c: c.cost <= 5, allow_none=False)
    if choice:
        gain(game, p, choice, to="hand")
    if player.hand:
        topdeck = yield from choose_one(game, p, player.hand, "TOPDECK",
                                        "Artisan: put a card from your hand onto your deck.", allow_none=False)
        if topdeck:
            move(topdeck, player.hand, player.deck)


KINGDOM_CARDS: dict[str, Card] = {c.name: c for c in [
    Card("Cellar", cost=2, types=ACTION, plus_actions=1, effect=cellar_effect),
    Card("Chapel", cost=2, types=ACTION, effect=chapel_effect),
    Card("Moat", cost=2, types=ACTION_REACTION, plus_cards=2),  # its reaction is in effects.attack_each_opponent
    Card("Harbinger", cost=3, types=ACTION, plus_cards=1, plus_actions=1, effect=harbinger_effect),
    Card("Merchant", cost=3, types=ACTION, plus_cards=1, plus_actions=1, effect=merchant_effect),
    Card("Vassal", cost=3, types=ACTION, plus_coins=2, effect=vassal_effect),
    Card("Village", cost=3, types=ACTION, plus_cards=1, plus_actions=2),
    Card("Workshop", cost=3, types=ACTION, effect=workshop_effect),
    Card("Bureaucrat", cost=4, types=ACTION_ATTACK, effect=bureaucrat_effect),
    Card("Gardens", cost=4, types=(CardType.VICTORY,), vp_value=_gardens_vp),
    Card("Militia", cost=4, types=ACTION_ATTACK, plus_coins=2, effect=militia_effect),
    Card("Moneylender", cost=4, types=ACTION, effect=moneylender_effect),
    Card("Poacher", cost=4, types=ACTION, plus_cards=1, plus_actions=1, plus_coins=1, effect=poacher_effect),
    Card("Remodel", cost=4, types=ACTION, effect=remodel_effect),
    Card("Smithy", cost=4, types=ACTION, plus_cards=3),
    Card("Throne Room", cost=4, types=ACTION, effect=throne_room_effect),
    Card("Bandit", cost=5, types=ACTION_ATTACK, effect=bandit_effect),
    Card("Council Room", cost=5, types=ACTION, plus_cards=4, plus_buys=1, effect=council_room_effect),
    Card("Festival", cost=5, types=ACTION, plus_actions=2, plus_buys=1, plus_coins=2),
    Card("Laboratory", cost=5, types=ACTION, plus_cards=2, plus_actions=1),
    Card("Library", cost=5, types=ACTION, effect=library_effect),
    Card("Market", cost=5, types=ACTION, plus_cards=1, plus_actions=1, plus_buys=1, plus_coins=1),
    Card("Mine", cost=5, types=ACTION, effect=mine_effect),
    Card("Sentry", cost=5, types=ACTION, plus_cards=1, plus_actions=1, effect=sentry_effect),
    Card("Witch", cost=5, types=ACTION_ATTACK, plus_cards=2, effect=witch_effect),
    Card("Artisan", cost=6, types=ACTION, effect=artisan_effect),
]}
