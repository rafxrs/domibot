"""Fixed-size numeric encodings of Game state and Action, for feeding the
policy/value network (see network.py). Nothing in this module trains
anything — it just defines the (observation, action) interface every
learning agent shares, kept separate from domibot itself since it's
specific to the RL side, not the rules engine.

Two vocabularies, fixed once at import time regardless of which 10 kingdom
cards a given game uses:

- `CARD_NAMES` — all 33 base-set cards (7 basic + 26 kingdom), sorted.
- `ACTION_VOCAB` — every (verb, card) pair that can ever be produced by any
  card effect (see effects.py/kingdom.py), plus the verb-only actions
  (DONE, YES, NO, END_ACTIONS, END_BUY, REVEAL_MOAT, NO_REVEAL, NONE).
  `Game.legal_actions()` at any point is always a subset of this list.
"""
from __future__ import annotations

import numpy as np

from domibot import ALL_CARDS, Action, DecisionKind, Game, Phase

CARD_NAMES: list[str] = sorted(ALL_CARDS)
NUM_CARDS = len(CARD_NAMES)
CARD_INDEX = {name: i for i, name in enumerate(CARD_NAMES)}

_CARD_VERBS = ("PLAY", "BUY", "GAIN", "TRASH", "DISCARD", "TOPDECK")
_VERB_ONLY_ACTIONS = (
    Action("DONE"), Action("YES"), Action("NO"), Action("END_ACTIONS"),
    Action("END_BUY"), Action("REVEAL_MOAT"), Action("NO_REVEAL"), Action("NONE"),
)

ACTION_VOCAB: list[Action] = [
    Action(verb, name) for verb in _CARD_VERBS for name in CARD_NAMES
] + list(_VERB_ONLY_ACTIONS)
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
NUM_ACTIONS = len(ACTION_VOCAB)

MAX_OPPONENTS = 3  # the base set supports up to 4 players, i.e. up to 3 opponents

_DECISION_KINDS = (DecisionKind.PHASE_ACTION, DecisionKind.SELECT_CARD, DecisionKind.YES_NO, DecisionKind.REACT)

# supply + trash + my hand + my total + my set_aside + source card
# + MAX_OPPONENTS * (discard + play_area + [hand_size, draw_pile_size, set_aside_size, active])
# + scalars (phase(2) + my counters(3) + my_turn_number(1) + is_my_turn(1) + decision_kind(4))
OBS_DIM = NUM_CARDS * 6 + MAX_OPPONENTS * (NUM_CARDS * 2 + 4) + 11


def action_to_index(action: Action) -> int:
    return ACTION_INDEX[action]


def index_to_action(index: int) -> Action:
    return ACTION_VOCAB[index]


def legal_action_mask(game: Game) -> np.ndarray:
    """Boolean mask over ACTION_VOCAB, True at indices that are legal right now."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in game.legal_actions():
        mask[ACTION_INDEX[a]] = True
    return mask


def _card_counts(names) -> np.ndarray:
    """One entry per card *instance* in `names` (e.g. a hand or discard pile)."""
    counts = np.zeros(NUM_CARDS, dtype=np.float32)
    for name in names:
        counts[CARD_INDEX[name]] += 1
    return counts


def _source_card_onehot(game: Game) -> np.ndarray:
    """Which card's effect raised the pending decision (all zeros at a
    plain phase-action boundary, where nothing is being resolved)."""
    counts = np.zeros(NUM_CARDS, dtype=np.float32)
    decision = game.pending_decision
    if decision is not None and decision.source_card is not None:
        counts[CARD_INDEX[decision.source_card]] = 1.0
    return counts


def _supply_counts(supply: dict[str, int]) -> np.ndarray:
    counts = np.zeros(NUM_CARDS, dtype=np.float32)
    for name, count in supply.items():
        counts[CARD_INDEX[name]] = count
    return counts


def encode_observation(game: Game, player_idx: int) -> np.ndarray:
    """Encode `game` from `player_idx`'s point of view. Respects hidden
    information: only `player_idx`'s own hand/deck composition is fully
    known; opponents expose only what's actually public in Dominion (their
    discard pile and play area, plus hand/draw-pile/set-aside *sizes* —
    never hand or deck *contents*).

    Includes the card whose effect raised the pending decision. Without it
    a "trash a card" prompt from Chapel (dump junk) and from Remodel (give
    up your best card to upgrade it) are byte-identical inputs, so no
    policy can answer both correctly — the options differ but the action
    mask is applied to the *output*, never seen as input."""
    me = game.players[player_idx]
    parts = [
        _supply_counts(game.supply),
        _card_counts(game.trash),
        _card_counts(me.hand),
        _card_counts(me.all_cards()),
        # Cards staged mid-effect (Sentry's two revealed cards, Library's
        # skipped Actions) -- for those decisions these *are* the cards
        # being decided about.
        _card_counts(me.set_aside),
        _source_card_onehot(game),
    ]

    opponents = game.other_players_in_order(player_idx)
    for slot in range(MAX_OPPONENTS):
        if slot < len(opponents):
            opp = game.players[opponents[slot]]
            parts.append(_card_counts(opp.discard))
            parts.append(_card_counts(opp.play_area))
            parts.append(np.array(
                [len(opp.hand), len(opp.deck), len(opp.set_aside), 1.0], dtype=np.float32))
        else:
            parts.append(np.zeros(NUM_CARDS, dtype=np.float32))
            parts.append(np.zeros(NUM_CARDS, dtype=np.float32))
            parts.append(np.zeros(4, dtype=np.float32))

    phase_onehot = np.array([1.0 if game.phase == Phase.ACTION else 0.0,
                              1.0 if game.phase == Phase.BUY else 0.0], dtype=np.float32)
    my_counters = np.array([me.actions, me.buys, me.coins], dtype=np.float32)
    my_turn_number = np.array([me.turns_taken + 1], dtype=np.float32)
    is_my_turn = np.array([1.0 if game.current_player == player_idx else 0.0], dtype=np.float32)
    kind = game.pending_decision.kind if game.pending_decision is not None else DecisionKind.PHASE_ACTION
    decision_onehot = np.array([1.0 if kind == k else 0.0 for k in _DECISION_KINDS], dtype=np.float32)

    parts += [phase_onehot, my_counters, my_turn_number, is_my_turn, decision_onehot]
    obs = np.concatenate(parts)
    assert obs.shape == (OBS_DIM,)
    return obs


# Public information the base encoding above leaves out, appended after it
# so the base vector is an exact prefix: a network trained on the base
# encoding just ignores the tail (see DomibotNet.extra_dim). Every gain and
# trash in Dominion is announced, so each opponent's total card ownership --
# and with it their score -- is public even though its split across
# hand/deck/discard isn't. Supply counts alone can't tell an emptied kingdom
# pile from a card that isn't in this game (both read 0), so the kingdom
# membership and empty-pile count are explicit too.
# kingdom membership + empty piles + my VP + MAX_OPPONENTS * (total cards + VP) + VP lead
EXTRA_DIM = NUM_CARDS + 1 + 1 + MAX_OPPONENTS * (NUM_CARDS + 1) + 1
FULL_OBS_DIM = OBS_DIM + EXTRA_DIM


def encode_public_extras(game: Game, player_idx: int) -> np.ndarray:
    scores = game.get_scores()
    in_game = np.zeros(NUM_CARDS, dtype=np.float32)
    for name in game.supply:
        in_game[CARD_INDEX[name]] = 1.0
    empty_piles = sum(1 for count in game.supply.values() if count == 0)
    parts = [in_game, np.array([empty_piles, scores[player_idx]], dtype=np.float32)]

    opponents = game.other_players_in_order(player_idx)
    for slot in range(MAX_OPPONENTS):
        if slot < len(opponents):
            parts.append(_card_counts(game.players[opponents[slot]].all_cards()))
            parts.append(np.array([scores[opponents[slot]]], dtype=np.float32))
        else:
            parts.append(np.zeros(NUM_CARDS + 1, dtype=np.float32))
    best_opp = max(scores[o] for o in opponents)
    parts.append(np.array([scores[player_idx] - best_opp], dtype=np.float32))
    extras = np.concatenate(parts)
    assert extras.shape == (EXTRA_DIM,)
    return extras


def encode_full_observation(game: Game, player_idx: int) -> np.ndarray:
    """`encode_observation` followed by `encode_public_extras`."""
    return np.concatenate([encode_observation(game, player_idx), encode_public_extras(game, player_idx)])


def encode_for(network, game: Game, player_idx: int) -> np.ndarray:
    """Whichever encoding `network` consumes: the full one if it was built
    with public-extras inputs, the base one otherwise."""
    if getattr(network, "extra_dim", 0):
        return encode_full_observation(game, player_idx)
    return encode_observation(game, player_idx)
