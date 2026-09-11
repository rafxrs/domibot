"""Reconstructs a `domibot.Game` from what's actually visible to a player
at the table during a real game (e.g. on dominion.games), so `DomibotAgent`
can be asked "what would you do here?" as a move advisor -- see
`examples/domibot_relay.py`. You play every move yourself; this only tells
you what Domibot recommends, using exactly the information available to a
player: your own hand and total card ownership exactly, everything else
only through what Dominion actually makes public (the supply, the trash,
discard piles, and your opponent's hand/draw-pile *sizes*, never their
contents).

Hidden information (the opponent's hand and draw-pile contents, and the
literal draw order of your own deck) is filled in by determinization:
- The opponent's *exact* total card ownership needs no guessing at all --
  every card is somewhere (the supply, the trash, or one of the two
  players), so `opponent_total = game_total - supply - trash - your_total`
  is exact arithmetic, not a guess.
- Whatever of that total isn't in their visible discard/play area is
  randomly split into a fake hand and draw pile matching the sizes you
  report, and your own draw pile (a known *set* of cards, since you know
  everything you've gained, but not the order they're stacked in) is
  randomly shuffled.

This is "perfect information Monte Carlo", a standard approximation for
search under hidden information -- not exact, but exactly as much as a
human opponent has to guess with.

Scope: mainly phase-action decisions (what to play, what to buy) --
`reconstruct_game` only ever produces boundary states (`pending_decision is
None`). The one exception is `reconstruct_opponent_turn_boundary`, which
positions the Game at the *opponent's* turn start instead of yours, so a
`log_parser`-derived path of their plays-so-far can be replayed (via
`training.mcts.materialize`) to reach a forced reaction pending on you --
currently just Militia's discard (see `log_parser._SUPPORTED_TERMINAL_ATTACKS`).
Every other sub-decision (your own mid-turn choices, or any other card's
forced reaction) still can't be represented here; follow the same fixed
"keep the good stuff, give up junk" rule `heuristics.heuristic_reaction`
uses for those instead.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from domibot import ALL_CARDS, Action, Game
from domibot.enums import Phase

# Short codes for the 26 base-set kingdom cards, so you don't have to type
# the full name at every prompt (dominion.games' log never lists the
# kingdom, so this is entered by hand every game). 2 letters where that's
# unambiguous; 3 where several cards share a prefix (the Market/Merchant/
# Militia/Mine/Moat/Moneylender cluster all start with "M").
CARD_ABBREVIATIONS: dict[str, str] = {
    "ART": "Artisan",
    "BAN": "Bandit",
    "BUR": "Bureaucrat",
    "CEL": "Cellar",
    "CHA": "Chapel",
    "CR": "Council Room",
    "FES": "Festival",
    "GAR": "Gardens",
    "HAR": "Harbinger",
    "LAB": "Laboratory",
    "LIB": "Library",
    "MAR": "Market",
    "MER": "Merchant",
    "MIL": "Militia",
    "MIN": "Mine",
    "MOA": "Moat",
    "MLR": "Moneylender",
    "POA": "Poacher",
    "REM": "Remodel",
    "SEN": "Sentry",
    "SMI": "Smithy",
    "TR": "Throne Room",
    "VAS": "Vassal",
    "VIL": "Village",
    "WIT": "Witch",
    "WOR": "Workshop",
}


def resolve_card_name(token: str) -> str:
    """A full card name, unchanged; or an abbreviation from
    CARD_ABBREVIATIONS (case-insensitive) expanded to one. Raises
    ValueError naming the token otherwise."""
    if token in ALL_CARDS:
        return token
    expanded = CARD_ABBREVIATIONS.get(token.upper())
    if expanded is not None:
        return expanded
    # case-insensitive full-name match too (e.g. "copper", "COPPER")
    for name in ALL_CARDS:
        if name.lower() == token.lower():
            return name
    raise ValueError(f"not a recognized card name or abbreviation: {token!r}")


@dataclass
class TableState:
    """Everything needed for one query, straight off what's on screen."""

    kingdom: list[str]
    supply: dict[str, int]
    trash: list[str]

    my_hand: list[str]
    my_discard: list[str]
    my_play_area: list[str] = field(default_factory=list)
    my_total: list[str] = field(default_factory=list)  # every card you own, any zone
    my_actions: int = 1
    my_buys: int = 1
    my_coins: int = 0
    my_phase: str = "ACTION"  # "ACTION" or "BUY"
    my_turns_taken: int = 0  # completed turns *before* this one (0 on your first turn)

    opp_discard: list[str] = field(default_factory=list)
    opp_play_area: list[str] = field(default_factory=list)
    opp_hand_size: int = 5
    opp_draw_pile_size: int = 5


def _over_accounted(known: Counter, total: Counter) -> Counter:
    """Cards where `known` claims more copies than `total` allows for --
    using Counter.subtract (not the `-` operator, which silently clips
    negative results at every step and would hide exactly the mismatches
    this is meant to catch)."""
    diff = known.copy()
    diff.subtract(total)
    return +diff  # unary + keeps only strictly-positive entries


def reconstruct_game(state: TableState, seed: int | None = None) -> Game:
    """Builds a `Game` positioned at your current phase-action decision,
    consistent with everything you reported. Raises `ValueError` (naming
    the mismatch) if the counts don't add up -- almost always a sign one
    of the inputs was mistyped, not a bug in this reconstruction, since
    the arithmetic it checks is exact."""
    game = Game(state.kingdom, num_players=2, seed=seed)
    rng = random.Random(seed)

    # Fixed for the whole game: every copy of every card is currently in
    # exactly one of {supply, trash, player 0, player 1} -- read off this
    # fresh game's own starting setup, before anything below overwrites it.
    total: Counter = Counter()
    for name, count in game.supply.items():
        total[name] += count
    for p in game.players:
        total.update(p.all_cards())

    game.supply = dict(state.supply)
    game.trash = list(state.trash)

    me = game.players[0]
    me.hand = list(state.my_hand)
    me.discard = list(state.my_discard)
    me.play_area = list(state.my_play_area)
    me.set_aside = []
    my_total_counter = Counter(state.my_total)
    my_known = Counter(me.hand) + Counter(me.discard) + Counter(me.play_area)
    bad = _over_accounted(my_known, my_total_counter)
    if bad:
        raise ValueError(
            f"my_total doesn't include enough copies of {dict(bad)} to cover my_hand/my_discard/"
            f"my_play_area -- my_total should list *everything* you currently own, any zone"
        )
    my_deck_counter = my_total_counter.copy()
    my_deck_counter.subtract(my_known)
    me.deck = list(my_deck_counter.elements())  # elements() already skips non-positive counts
    rng.shuffle(me.deck)  # the order is genuinely unknown, even to you
    me.actions = state.my_actions
    me.buys = state.my_buys
    me.coins = state.my_coins
    me.turns_taken = state.my_turns_taken

    opp = game.players[1]
    opp.discard = list(state.opp_discard)
    opp.play_area = list(state.opp_play_area)
    opp.set_aside = []
    opp_total_counter = total.copy()
    opp_total_counter.subtract(Counter(game.supply))
    opp_total_counter.subtract(Counter(game.trash))
    opp_total_counter.subtract(my_total_counter)
    bad = -opp_total_counter  # unary - keeps only strictly-negative entries (as positive magnitudes)
    if bad:
        raise ValueError(
            f"supply + trash + my_total account for {dict(bad)} more copies than exist in the game -- "
            f"double check them against what's on screen"
        )
    opp_known = Counter(opp.discard) + Counter(opp.play_area)
    bad = _over_accounted(opp_known, opp_total_counter)
    if bad:
        raise ValueError(
            f"opp_discard/opp_play_area claim {dict(bad)} more copies than the opponent could possibly "
            f"own by elimination -- double check my_total and the supply/trash counts"
        )
    opp_hidden_counter = opp_total_counter.copy()
    opp_hidden_counter.subtract(opp_known)
    opp_hidden = list(opp_hidden_counter.elements())
    reported_hidden = state.opp_hand_size + state.opp_draw_pile_size
    if len(opp_hidden) != reported_hidden:
        raise ValueError(
            f"the opponent must be holding {len(opp_hidden)} unseen cards by elimination "
            f"(their total minus their visible discard/play area), but you reported "
            f"opp_hand_size={state.opp_hand_size} + opp_draw_pile_size={state.opp_draw_pile_size} "
            f"= {reported_hidden} -- check those two, and also my_total (an error there shows up here, "
            f"not on your own side, since it throws off the opponent's total by elimination)"
        )
    rng.shuffle(opp_hidden)
    opp.hand = opp_hidden[: state.opp_hand_size]
    opp.deck = opp_hidden[state.opp_hand_size:]
    opp.actions = opp.buys = opp.coins = 0
    opp.turns_taken = max(state.my_turns_taken - 1, 0)  # never read for a non-perspective player; kept plausible

    game.current_player = 0
    game.turn_number = state.my_turns_taken + 1
    game.phase = Phase[state.my_phase]
    game.turn_merchant_bonus = 0
    game.turn_silver_played = False
    game.pending_decision = None
    game.pending_gen = None
    game.action_log = []
    return game


def reconstruct_opponent_turn_boundary(
    state: TableState, opp_turn_start_discard: list[str], path: list[Action] | None = None,
    seed: int | None = None,
) -> Game:
    """Like reconstruct_game, but positions the Game at the START of the
    opponent's still-open turn (current_player=1, Phase.ACTION) instead of
    your own phase-action decision -- for replaying
    log_parser.ParsedLog.pending_reaction_path via training.mcts.materialize
    to reach whatever Decision (if any) is now pending on you (Militia's
    forced discard, currently the only supported case -- see
    log_parser._SUPPORTED_TERMINAL_ATTACKS).

    Only valid together with a path built from log_parser's
    _SAFE_MIDTURN_ACTION_CARDS whitelist: every such card is known to never
    touch gain/buy/trash/topdeck, so every other field on `state` is safe
    to reuse as-is -- your own hand hasn't changed since it's not your turn
    yet, and the opponent's play_area/hand_size at *any* turn start are
    always [] and 5 (Game._cleanup_and_advance always empties play_area and
    draws back to a full hand). Only their *discard* can differ between
    "now" and "turn start" (whatever they've bought/gained this turn
    already sits there), hence the separate `opp_turn_start_discard` -- a
    snapshot taken the moment their turn began, before any of that.

    `path` is that same list of plays: pass it here too (not just to
    materialize/run_mcts afterward) so the cards it names are forced into
    the opponent's *reconstructed* turn-start hand rather than left to
    chance. Their hidden hand/deck split is otherwise random (perfect
    information Monte Carlo, see the module docstring) -- but we don't
    actually need to guess whether they held e.g. Militia, we *know* they
    did, they just played it. Without this, `materialize` would raise
    "illegal action" any time the random split happened to put a known
    play in their deck instead of their hand."""
    opp_total_now = (state.opp_hand_size + state.opp_draw_pile_size
                      + len(state.opp_discard) + len(state.opp_play_area))
    turn_start_state = TableState(
        kingdom=state.kingdom, supply=state.supply, trash=state.trash,
        my_hand=state.my_hand, my_discard=state.my_discard,
        my_play_area=state.my_play_area, my_total=state.my_total,
        my_actions=state.my_actions, my_buys=state.my_buys, my_coins=state.my_coins,
        my_phase=state.my_phase, my_turns_taken=state.my_turns_taken,
        opp_discard=opp_turn_start_discard, opp_play_area=[],
        opp_hand_size=5,
        opp_draw_pile_size=opp_total_now - len(opp_turn_start_discard) - 5,
    )
    game = reconstruct_game(turn_start_state, seed=seed)  # reuses its own arithmetic/validation
    game.current_player = 1
    game.phase = Phase.ACTION
    game.players[1].actions = 1
    game.players[1].buys = 1

    opp = game.players[1]
    rng = random.Random(seed)
    for action in path or []:
        card = action.card
        if card in opp.hand:
            continue
        if card not in opp.deck:
            raise ValueError(
                f"path plays {card!r}, but the opponent's reconstructed turn-start hand+deck "
                f"don't contain it -- their total ownership (by elimination) must be wrong"
            )
        opp.deck.remove(card)
        opp.deck.append(opp.hand.pop())  # displace an arbitrary hand card to make room
        opp.hand.append(card)
        rng.shuffle(opp.deck)
    return game
