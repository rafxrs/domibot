"""All 26 Base-set kingdom cards and their effects.

Each effect is either a plain function `(game, player_idx) -> None` when it
never needs a decision, or a generator `(game, player_idx) -> Generator[...]`
when it does. `Game.resolve_action` drives either form transparently.
"""
from __future__ import annotations

from ..card import Card
from ..enums import CardType
from ..effects import (
    choose_cards,
    choose_from_supply,
    choose_one,
    gain,
    move,
    peek_top,
    trash_from,
    yes_no,
    attack_each_opponent,
)

ACTION = (CardType.ACTION,)
ACTION_ATTACK = (CardType.ACTION, CardType.ATTACK)
ACTION_REACTION = (CardType.ACTION, CardType.REACTION)


# ---------------------------------------------------------------- Cellar ---
def cellar_effect(game, p):
    player = game.players[p]
    discarded = yield from choose_cards(
        game, p, player.hand, "DISCARD",
        "Cellar: discard any number of cards, then draw that many.",
        min_count=0,
    )
    for c in discarded:
        move(c, player.hand, player.discard)
    player.draw(len(discarded), game.rng)


CELLAR = Card("Cellar", cost=2, types=ACTION, plus_actions=1, effect=cellar_effect)


# ---------------------------------------------------------------- Chapel ---
def chapel_effect(game, p):
    player = game.players[p]
    to_trash = yield from choose_cards(
        game, p, player.hand, "TRASH", "Chapel: trash up to 4 cards from your hand.",
        min_count=0, max_count=4,
    )
    for c in to_trash:
        trash_from(game, c, player.hand)


CHAPEL = Card("Chapel", cost=2, types=ACTION, effect=chapel_effect)


# ------------------------------------------------------------------ Moat ---
# The Reaction half (block an Attack) is handled by effects.attack_each_opponent,
# which checks each opponent's hand for "Moat" directly.
MOAT = Card("Moat", cost=2, types=ACTION_REACTION, plus_cards=2)


# ------------------------------------------------------------- Harbinger ---
def harbinger_effect(game, p):
    player = game.players[p]
    if not player.discard:
        return
    choice = yield from choose_one(
        game, p, player.discard, "TOPDECK",
        "Harbinger: put a card from your discard pile onto your deck?",
        allow_none=True,
    )
    if choice:
        move(choice, player.discard, player.deck)


HARBINGER = Card("Harbinger", cost=3, types=ACTION, plus_cards=1, plus_actions=1, effect=harbinger_effect)


# --------------------------------------------------------------- Merchant ---
def merchant_effect(game, p):
    game.turn_merchant_bonus += 1


MERCHANT = Card("Merchant", cost=3, types=ACTION, plus_cards=1, plus_actions=1, effect=merchant_effect)


# ----------------------------------------------------------------- Vassal ---
def vassal_effect(game, p):
    player = game.players[p]
    top = peek_top(game, player)
    if top is None:
        return
    player.deck.pop()
    # Staged in set_aside rather than held only in this generator's locals:
    # while the Decision below is pending the card must still live in a real
    # zone, or every state query (all_cards, total_cards, and therefore the
    # RL observation) silently under-counts the player's deck.
    player.set_aside.append(top)
    card = game.cards[top]
    if CardType.ACTION in card.types:
        play_it = yield from yes_no(p, f"Vassal discarded {top}. Play it?")
        if play_it:
            player.set_aside.remove(top)
            player.play_area.append(top)
            yield from game.resolve_action(p, top)
            return
    player.set_aside.remove(top)
    player.discard.append(top)


VASSAL = Card("Vassal", cost=3, types=ACTION, plus_coins=2, effect=vassal_effect)


# ---------------------------------------------------------------- Village ---
VILLAGE = Card("Village", cost=3, types=ACTION, plus_cards=1, plus_actions=2)


# -------------------------------------------------------------- Workshop ---
def workshop_effect(game, p):
    choice = yield from choose_from_supply(
        game, p, "Workshop: gain a card costing up to 4.", "GAIN",
        lambda c: c.cost <= 4, allow_none=False,
    )
    if choice:
        gain(game, p, choice, to="discard")


WORKSHOP = Card("Workshop", cost=3, types=ACTION, effect=workshop_effect)


# ------------------------------------------------------------- Bureaucrat ---
def _bureaucrat_hit(game, opp):
    player = game.players[opp]
    victories = [c for c in player.hand if CardType.VICTORY in game.cards[c].types]
    if not victories:
        return
    choice = yield from choose_one(
        game, opp, victories, "TOPDECK",
        "Bureaucrat: put a Victory card from your hand onto your deck.",
        allow_none=False,
    )
    if choice:
        move(choice, player.hand, player.deck)


def bureaucrat_effect(game, p):
    gain(game, p, "Silver", to="deck_top")
    yield from attack_each_opponent(game, p, _bureaucrat_hit)


BUREAUCRAT = Card("Bureaucrat", cost=4, types=ACTION_ATTACK, effect=bureaucrat_effect)


# --------------------------------------------------------------- Gardens ---
def _gardens_vp(player) -> int:
    return len(player.all_cards()) // 10


GARDENS = Card("Gardens", cost=4, types=(CardType.VICTORY,), vp_value=_gardens_vp)


# --------------------------------------------------------------- Militia ---
def _militia_hit(game, opp):
    player = game.players[opp]
    n = len(player.hand) - 3
    if n <= 0:
        return
    chosen = yield from choose_cards(
        game, opp, player.hand, "DISCARD",
        f"Militia: discard down to 3 cards in hand ({n} to discard).",
        min_count=n, max_count=n,
    )
    for c in chosen:
        move(c, player.hand, player.discard)


def militia_effect(game, p):
    yield from attack_each_opponent(game, p, _militia_hit)


MILITIA = Card("Militia", cost=4, types=ACTION_ATTACK, plus_coins=2, effect=militia_effect)


# ----------------------------------------------------------- Moneylender ---
def moneylender_effect(game, p):
    player = game.players[p]
    if "Copper" not in player.hand:
        return
    do_it = yield from yes_no(p, "Moneylender: trash a Copper from your hand for +3 Coins?")
    if do_it:
        trash_from(game, "Copper", player.hand)
        player.coins += 3


MONEYLENDER = Card("Moneylender", cost=4, types=ACTION, effect=moneylender_effect)


# --------------------------------------------------------------- Poacher ---
def poacher_effect(game, p):
    player = game.players[p]
    empty_piles = sum(1 for count in game.supply.values() if count == 0)
    n = min(empty_piles, len(player.hand))
    if n <= 0:
        return
    chosen = yield from choose_cards(
        game, p, player.hand, "DISCARD",
        f"Poacher: discard {n} card(s), one per empty supply pile.",
        min_count=n, max_count=n,
    )
    for c in chosen:
        move(c, player.hand, player.discard)


POACHER = Card("Poacher", cost=4, types=ACTION, plus_cards=1, plus_actions=1, plus_coins=1, effect=poacher_effect)


# --------------------------------------------------------------- Remodel ---
def remodel_effect(game, p):
    player = game.players[p]
    if not player.hand:
        return
    to_trash = yield from choose_one(
        game, p, player.hand, "TRASH", "Remodel: trash a card from your hand.", allow_none=False
    )
    if not to_trash:
        return
    max_cost = game.cards[to_trash].cost + 2
    trash_from(game, to_trash, player.hand)
    choice = yield from choose_from_supply(
        game, p, f"Remodel: gain a card costing up to {max_cost}.", "GAIN",
        lambda c: c.cost <= max_cost, allow_none=False,
    )
    if choice:
        gain(game, p, choice, to="discard")


REMODEL = Card("Remodel", cost=4, types=ACTION, effect=remodel_effect)


# ---------------------------------------------------------------- Smithy ---
SMITHY = Card("Smithy", cost=4, types=ACTION, plus_cards=3)


# ----------------------------------------------------------- Throne Room ---
def throne_room_effect(game, p):
    player = game.players[p]
    action_cards = [c for c in player.hand if CardType.ACTION in game.cards[c].types]
    if not action_cards:
        return
    choice = yield from choose_one(
        game, p, action_cards, "PLAY",
        "Throne Room: play an Action card from your hand twice.", allow_none=True,
    )
    if not choice:
        return
    move(choice, player.hand, player.play_area)
    for _ in range(2):
        yield from game.resolve_action(p, choice)


THRONE_ROOM = Card("Throne Room", cost=4, types=ACTION, effect=throne_room_effect)


# ---------------------------------------------------------------- Bandit ---
def _bandit_hit(game, opp):
    player = game.players[opp]
    revealed = []
    for _ in range(2):
        top = peek_top(game, player)
        if top is None:
            break
        player.deck.pop()
        revealed.append(top)
        player.set_aside.append(top)  # a real zone while the Decision is pending
    targets = [c for c in revealed if CardType.TREASURE in game.cards[c].types and c != "Copper"]
    if targets:
        # Always routed through choose_one, even with a single distinct
        # target (choose_one yields regardless of candidate count) -- this
        # is what makes the trash show up as a real Action/LogEntry (see
        # Game.step), rather than a silent state mutation nothing can see
        # in a saved log or the GUI's activity feed.
        choice = yield from choose_one(
            game, opp, targets, "TRASH", "Bandit: choose a Treasure to trash.", allow_none=False
        )
        revealed.remove(choice)
        player.set_aside.remove(choice)
        game.trash.append(choice)
    for c in revealed:
        player.set_aside.remove(c)
        player.discard.append(c)


def bandit_effect(game, p):
    gain(game, p, "Gold", to="discard")
    yield from attack_each_opponent(game, p, _bandit_hit)


BANDIT = Card("Bandit", cost=5, types=ACTION_ATTACK, effect=bandit_effect)


# ---------------------------------------------------------- Council Room ---
def council_room_effect(game, p):
    for opp in game.other_players_in_order(p):
        game.players[opp].draw(1, game.rng)


COUNCIL_ROOM = Card("Council Room", cost=5, types=ACTION, plus_cards=4, plus_buys=1, effect=council_room_effect)


# --------------------------------------------------------------- Festival ---
FESTIVAL = Card("Festival", cost=5, types=ACTION, plus_actions=2, plus_buys=1, plus_coins=2)


# ------------------------------------------------------------- Laboratory ---
LABORATORY = Card("Laboratory", cost=5, types=ACTION, plus_cards=2, plus_actions=1)


# ---------------------------------------------------------------- Library ---
def library_effect(game, p):
    player = game.players[p]
    set_aside: list[str] = []
    while len(player.hand) < 7:
        top = peek_top(game, player)
        if top is None:
            break
        player.deck.pop()
        player.set_aside.append(top)  # a real zone while the Decision is pending
        card = game.cards[top]
        if CardType.ACTION in card.types:
            keep = yield from yes_no(p, f"Library: draw {top} into your hand? (No sets it aside instead)")
            if not keep:
                set_aside.append(top)  # stays in player.set_aside until cleanup below
                continue
        player.set_aside.remove(top)
        player.hand.append(top)
    for c in set_aside:
        player.set_aside.remove(c)
        player.discard.append(c)


LIBRARY = Card("Library", cost=5, types=ACTION, effect=library_effect)


# ---------------------------------------------------------------- Market ---
MARKET = Card("Market", cost=5, types=ACTION, plus_cards=1, plus_actions=1, plus_buys=1, plus_coins=1)


# ------------------------------------------------------------------ Mine ---
def mine_effect(game, p):
    player = game.players[p]
    treasures = [c for c in player.hand if CardType.TREASURE in game.cards[c].types]
    if not treasures:
        return
    to_trash = yield from choose_one(
        game, p, treasures, "TRASH", "Mine: trash a Treasure from your hand?", allow_none=True
    )
    if not to_trash:
        return
    max_cost = game.cards[to_trash].cost + 3
    trash_from(game, to_trash, player.hand)
    choice = yield from choose_from_supply(
        game, p, f"Mine: gain a Treasure costing up to {max_cost} to your hand.", "GAIN",
        lambda c: CardType.TREASURE in c.types and c.cost <= max_cost, allow_none=False,
    )
    if choice:
        gain(game, p, choice, to="hand")


MINE = Card("Mine", cost=5, types=ACTION, effect=mine_effect)


# ---------------------------------------------------------------- Sentry ---
def sentry_effect(game, p):
    player = game.players[p]
    revealed = []
    for _ in range(2):
        top = peek_top(game, player)
        if top is None:
            break
        player.deck.pop()
        revealed.append(top)
        player.set_aside.append(top)  # a real zone while the Decisions are pending
    if not revealed:
        return

    remaining = list(revealed)
    to_trash = yield from choose_cards(
        game, p, remaining, "TRASH", "Sentry: trash any of the two revealed cards.", min_count=0
    )
    for c in to_trash:
        remaining.remove(c)
        player.set_aside.remove(c)
        game.trash.append(c)

    to_discard = yield from choose_cards(
        game, p, remaining, "DISCARD", "Sentry: discard any of the remaining revealed cards.", min_count=0
    )
    for c in to_discard:
        remaining.remove(c)
        player.set_aside.remove(c)
        player.discard.append(c)

    order: list[str] = []
    pool = list(remaining)
    while pool:
        choice = yield from choose_one(
            game, p, pool, "TOPDECK",
            "Sentry: pick the next card to return to your deck (the last one you pick ends up on top).",
            allow_none=False,
        )
        pool.remove(choice)
        order.append(choice)
    for c in order:
        player.set_aside.remove(c)
        player.deck.append(c)


SENTRY = Card("Sentry", cost=5, types=ACTION, plus_cards=1, plus_actions=1, effect=sentry_effect)


# ---------------------------------------------------------------- Witch ---
def _witch_hit(game, opp):
    gain(game, opp, "Curse", to="discard")


def witch_effect(game, p):
    yield from attack_each_opponent(game, p, _witch_hit)


WITCH = Card("Witch", cost=5, types=ACTION_ATTACK, plus_cards=2, effect=witch_effect)


# -------------------------------------------------------------- Artisan ---
def artisan_effect(game, p):
    player = game.players[p]
    choice = yield from choose_from_supply(
        game, p, "Artisan: gain a card to your hand costing up to 5.", "GAIN",
        lambda c: c.cost <= 5, allow_none=False,
    )
    if choice:
        gain(game, p, choice, to="hand")
    if player.hand:
        topdeck_choice = yield from choose_one(
            game, p, player.hand, "TOPDECK", "Artisan: put a card from your hand onto your deck.", allow_none=False
        )
        if topdeck_choice:
            move(topdeck_choice, player.hand, player.deck)


ARTISAN = Card("Artisan", cost=6, types=ACTION, effect=artisan_effect)


KINGDOM_CARDS: dict[str, Card] = {
    c.name: c
    for c in [
        CELLAR, CHAPEL, MOAT, HARBINGER, MERCHANT, VASSAL, VILLAGE, WORKSHOP,
        BUREAUCRAT, GARDENS, MILITIA, MONEYLENDER, POACHER, REMODEL, SMITHY,
        THRONE_ROOM, BANDIT, COUNCIL_ROOM, FESTIVAL, LABORATORY, LIBRARY,
        MARKET, MINE, SENTRY, WITCH, ARTISAN,
    ]
}
