"""Exploration that doesn't die out once the policy stops buying a card.

On-policy PPO only learns about a card from games where the policy itself
bought it. A card bought before the policy can play it well loses to
Silver and Gold, its buy probability falls toward zero, and after that no
game ever holds it again, so nothing can find out what it's worth.
`domibot2.2` lost Throne Room, Village and Chapel this way (see
training/README.md, "Card use"). Two remedies, both used only while
collecting training games (evals and real play are unaffected):

- **Steered players** (`ExploreConfig`, `steer`): in a share of the games,
  one player gets a focus plan -- one or two of the kingdom's cards and how
  many copies of each it wants -- and during its first turns, some of its
  buys are replaced by a focus card it can afford. Everything else that
  player does, including how it plays those cards, is its own policy, so
  the value head learns what decks holding them are worth under the
  current policy. A replaced buy isn't the policy's choice, so it trains
  neither the policy nor the value head, and the GAE trace is cut there
  (`gae.compute_gae`) so earlier decisions aren't credited or blamed for it.
- **A buy floor** (`buy_floor_loss`): a penalty whenever the policy gives
  an affordable kingdom card less than a minimum probability. The policy
  keeps sampling every such card now and then, so its own buy decisions
  keep getting feedback, and once the value head learns a card is good
  those samples pull its probability up.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from domibot import KINGDOM_CARDS, Action, Game, Phase

from .. import encoding


@dataclass
class ExploreConfig:
    frac: float = 0.5  # share of self-play games with one steered player
    override_prob: float = 0.5  # chance each eligible buy is replaced by a focus card
    max_cards: int = 2  # focus cards per plan: 1 to this many, from the game's kingdom
    max_count: int = 3  # copies wanted of each: 1 to this many
    turn_limit: int = 12  # steer only during the player's first this many turns


@dataclass
class FocusPlan:
    seat: int
    wanted: dict[str, int]  # focus card -> copies wanted


def make_plan(kingdom: list[str], num_players: int, config: ExploreConfig, rng: random.Random) -> FocusPlan | None:
    """A focus plan for one game, or None (a normal game) with probability
    1 - `config.frac`. Cards are drawn uniformly from the kingdom, so over
    many games every card gets steered toward."""
    if rng.random() >= config.frac:
        return None
    cards = rng.sample(kingdom, rng.randint(1, config.max_cards))
    return FocusPlan(seat=rng.randrange(num_players), wanted={c: rng.randint(1, config.max_count) for c in cards})


def steer(game: Game, plan: FocusPlan | None, config: ExploreConfig, rng: random.Random) -> Action | None:
    """The focus card to buy instead of the policy's choice at this
    decision, or None to let the policy decide."""
    if plan is None or game.pending_decision is not None or game.phase != Phase.BUY:
        return None
    if game.current_player != plan.seat:
        return None
    me = game.players[plan.seat]
    if me.turns_taken >= config.turn_limit:
        return None
    owned = Counter(me.all_cards())
    legal = game.legal_actions()
    options = [Action("BUY", c) for c, n in plan.wanted.items() if owned[c] < n and Action("BUY", c) in legal]
    if not options or rng.random() >= config.override_prob:
        return None
    return rng.choice(options)


_KINGDOM_BUYS = np.zeros(encoding.NUM_ACTIONS, dtype=bool)
for _card in KINGDOM_CARDS:
    _KINGDOM_BUYS[encoding.action_to_index(Action("BUY", _card))] = True


def buy_floor_loss(masked_logits: torch.Tensor, legal: torch.Tensor, floor: float) -> torch.Tensor:
    """Mean over decisions where a kingdom card can be bought of
    sum(max(0, log(floor) - log pi(buy card))) over those cards: zero once
    every affordable kingdom card has at least probability `floor`."""
    kingdom_buys = legal & torch.from_numpy(_KINGDOM_BUYS).to(legal.device)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    deficit = torch.clamp(math.log(floor) - log_probs, min=0.0) * kingdom_buys
    decisions = kingdom_buys.any(dim=-1).sum().clamp(min=1)
    return deficit.sum() / decisions
