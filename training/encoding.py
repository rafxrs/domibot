"""The network's inputs and outputs: fixed-size encodings of a game and of actions.

`CARD_NAMES` holds all 33 base-set cards; `ACTION_VOCAB` every (verb, card)
pair any effect can produce plus the verb-only actions, so
`Game.legal_actions()` is always a subset of it. An observation is
`encode_observation` (hidden information respected), then the public extras,
then the player's own zones; each network reads the prefix it was built for.
"""
from __future__ import annotations

import numpy as np

from domibot import ALL_CARDS, Action, DecisionKind, Game, Phase

CARD_NAMES: list[str] = sorted(ALL_CARDS)
NUM_CARDS = len(CARD_NAMES)
CARD_INDEX = {name: i for i, name in enumerate(CARD_NAMES)}

_CARD_VERBS = ("PLAY", "BUY", "GAIN", "TRASH", "DISCARD", "TOPDECK")
_VERB_ONLY_ACTIONS = (Action("DONE"), Action("YES"), Action("NO"), Action("END_ACTIONS"), Action("END_BUY"),
                      Action("REVEAL_MOAT"), Action("NO_REVEAL"), Action("NONE"))
ACTION_VOCAB: list[Action] = [Action(v, c) for v in _CARD_VERBS for c in CARD_NAMES] + list(_VERB_ONLY_ACTIONS)
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
NUM_ACTIONS = len(ACTION_VOCAB)

MAX_OPPONENTS = 3
_DECISION_KINDS = (DecisionKind.PHASE_ACTION, DecisionKind.SELECT_CARD, DecisionKind.YES_NO, DecisionKind.REACT)

# supply, trash, my hand, my cards, my set-aside, the source card;
# per opponent: discard, play area, [hand size, draw pile size, set-aside size, present];
# phase (2), actions/buys/coins (3), my turn number, is my turn, decision kind (4)
OBS_DIM = NUM_CARDS * 6 + MAX_OPPONENTS * (NUM_CARDS * 2 + 4) + 11
# kingdom membership, empty piles, my VP; per opponent: its cards and VP; my VP lead
EXTRA_DIM = NUM_CARDS + 1 + 1 + MAX_OPPONENTS * (NUM_CARDS + 1) + 1
# my draw pile, discard pile, play area (counts; the draw pile's order is unknown), draw pile and discard sizes
ZONES_DIM = NUM_CARDS * 3 + 2
FULL_OBS_DIM = OBS_DIM + EXTRA_DIM + ZONES_DIM


def action_to_index(action: Action) -> int:
    return ACTION_INDEX[action]


def index_to_action(index: int) -> Action:
    return ACTION_VOCAB[index]


def legal_action_mask(game: Game) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in game.legal_actions():
        mask[ACTION_INDEX[a]] = True
    return mask


def _card_counts(names) -> np.ndarray:
    counts = np.zeros(NUM_CARDS, dtype=np.float32)
    for name in names:
        counts[CARD_INDEX[name]] += 1
    return counts


def encode_observation(game: Game, player_idx: int) -> np.ndarray:
    """`player_idx`'s view: its own hand and cards exactly, but only each opponent's
    discard pile, play area and zone sizes. Includes the card whose effect raised
    the pending choice (a Chapel trash and a Remodel trash otherwise look alike)."""
    me = game.players[player_idx]
    decision = game.pending_decision
    source = _card_counts([decision.source_card] if decision is not None and decision.source_card else [])
    supply = np.zeros(NUM_CARDS, dtype=np.float32)
    for name, count in game.supply.items():
        supply[CARD_INDEX[name]] = count
    parts = [supply, _card_counts(game.trash), _card_counts(me.hand), _card_counts(me.all_cards()),
             _card_counts(me.set_aside), source]
    opponents = game.other_players_in_order(player_idx)
    for slot in range(MAX_OPPONENTS):
        if slot < len(opponents):
            opp = game.players[opponents[slot]]
            parts += [_card_counts(opp.discard), _card_counts(opp.play_area),
                      np.array([len(opp.hand), len(opp.deck), len(opp.set_aside), 1.0], dtype=np.float32)]
        else:
            parts += [np.zeros(NUM_CARDS * 2 + 4, dtype=np.float32)]
    kind = decision.kind if decision is not None else DecisionKind.PHASE_ACTION
    parts.append(np.array([game.phase == Phase.ACTION, game.phase == Phase.BUY, me.actions, me.buys, me.coins,
                           me.turns_taken + 1, game.current_player == player_idx]
                          + [kind == k for k in _DECISION_KINDS], dtype=np.float32))
    obs = np.concatenate(parts)
    assert obs.shape == (OBS_DIM,)
    return obs


def encode_public_extras(game: Game, player_idx: int) -> np.ndarray:
    """Public facts the base encoding leaves out: every gain and trash is announced,
    so each opponent's card ownership and score are public."""
    scores = game.get_scores()
    in_game = _card_counts(game.supply)
    empty_piles = sum(1 for count in game.supply.values() if count == 0)
    parts = [in_game, np.array([empty_piles, scores[player_idx]], dtype=np.float32)]
    opponents = game.other_players_in_order(player_idx)
    for slot in range(MAX_OPPONENTS):
        if slot < len(opponents):
            parts += [_card_counts(game.players[opponents[slot]].all_cards()),
                      np.array([scores[opponents[slot]]], dtype=np.float32)]
        else:
            parts.append(np.zeros(NUM_CARDS + 1, dtype=np.float32))
    parts.append(np.array([scores[player_idx] - max(scores[o] for o in opponents)], dtype=np.float32))
    extras = np.concatenate(parts)
    assert extras.shape == (EXTRA_DIM,)
    return extras


def encode_own_zones(game: Game, player_idx: int) -> np.ndarray:
    me = game.players[player_idx]
    return np.concatenate([_card_counts(me.deck), _card_counts(me.discard), _card_counts(me.play_area),
                           np.array([len(me.deck), len(me.discard)], dtype=np.float32)])


def encode_full_observation(game: Game, player_idx: int) -> np.ndarray:
    return np.concatenate([encode_observation(game, player_idx), encode_public_extras(game, player_idx),
                           encode_own_zones(game, player_idx)])


def encode_for(network, game: Game, player_idx: int) -> np.ndarray:
    """The full encoding for a network with any inputs past the base ones, else the base one."""
    if getattr(network, "extra_dim", 0) or getattr(network, "zones_dim", 0):
        return encode_full_observation(game, player_idx)
    return encode_observation(game, player_idx)
