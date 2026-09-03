# domibot

A Dominion (base set) game engine, built as the foundational layer for later
training self-play RL agents (AlphaZero-style: policy/value network + MCTS).
Inspired by [pyminion](https://github.com/evanofslack/pyminion), but the API
is shaped around the needs of a search/self-play loop rather than a
human-playable CLI: at every point there is exactly one player who must make
exactly one choice from an explicit list of legal actions.

## Quickstart

```python
import random
from domibot import Game

kingdom = [
    "Village", "Smithy", "Market", "Laboratory", "Festival",
    "Witch", "Moat", "Council Room", "Chapel", "Throne Room",
]
game = Game(kingdom, num_players=2, seed=0)

while not game.is_game_over():
    actions = game.legal_actions()
    game.step(random.choice(actions))  # replace with a policy/agent

print(game.get_scores(), game.winners())
```

Run `python examples/random_playout.py [n]` to play `n` random games across
random kingdoms and player counts as a smoke test.

Every action taken is recorded in `game.action_log`. Save it to a file with:

```python
game.save_log("game_logs/my_game.log")             # human-readable text (default)
game.save_log("game_logs/my_game.json", fmt="json")  # structured, e.g. for a training pipeline
```

`save_log` creates any missing parent directories, so a path like
`game_logs/...` just works. `examples/play_vs_random.py` saves into
`game_logs/` (next to the project root, one per session, named by
timestamp + seed) when you opt in at the end of a game; that directory is
gitignored since its contents are generated, not source.

## Design: decisions as a generator, actions as a flat, typed choice

Dominion's rules text is full of nested, variable-length choices ("trash up
to 4 cards", "each opponent discards down to 3", "play an Action card
twice"). Modeling that with a hand-written state machine per phase gets
unwieldy fast. Instead, every card effect that needs player input is a
Python generator:

```python
def chapel_effect(game, player_idx):
    ...
    to_trash = yield from choose_cards(game, player_idx, player.hand, "TRASH", ...)
    for c in to_trash:
        trash_from(game, c, player.hand)
```

`Game` drives whichever generator is currently suspended: `legal_actions()`
returns the options attached to the pending `Decision`, and `step(action)`
resumes the generator with that choice (`generator.send(action)`), which
runs until the next `yield` or until the effect finishes. Cards with
sub-effects (Throne Room, Vassal replaying a discarded Action) compose with
plain `yield from` on another card's effect. Attacks share one helper,
`attack_each_opponent`, that walks opponents in turn order and inserts the
Moat "reveal to block?" decision automatically.

This keeps the action space small and uniform for a future policy network:
variable-length selections (Cellar, Chapel, Militia, Sentry's trash/discard
steps) are modeled as a *sequence* of single-card-or-DONE decisions rather
than one combinatorial "choose a subset" decision, so the legal-action count
at any single step is bounded by hand size, not by 2^hand_size.

`Action` is a flat `(verb, card_or_None)` pair (e.g. `PLAY(Village)`,
`BUY(Silver)`, `TRASH(Copper)`, `DONE`) — hashable and directly usable as a
policy-network output token. `Decision.kind` (`PHASE_ACTION` / `SELECT_CARD`
/ `YES_NO` / `REACT`) gives a future encoder context for *why* a given verb
is legal, since e.g. `YES`/`NO` is reused across several unrelated card
effects (Moneylender's optional trash, Library's draw-or-skip, Vassal's
play-or-not).

## Layout

Repo root: `src/domibot/` (the engine, below), `tests/`, `examples/` (CLI
scripts), `training/` (the RL layer — see its own README), `game_logs/`
(gitignored, generated).

- `enums.py` — `CardType`, `Phase`, `DecisionKind`
- `models.py` — `Action`, `Decision`
- `card.py` — `Card` definition (cost, types, flat bonuses, optional effect)
- `player.py` — `PlayerState`: deck/hand/discard/play_area/set_aside zones
- `effects.py` — shared decision primitives (`choose_cards`, `choose_one`,
  `choose_from_supply`, `yes_no`, `attack_each_opponent`, `peek_top`, ...)
- `cards/basic.py` — Copper, Silver, Gold, Estate, Duchy, Province, Curse
- `cards/kingdom.py` — all 26 base-set kingdom cards and their effects
- `game.py` — `Game`: supply/turn setup, `legal_actions()`, `step()`,
  scoring, game-end conditions
- `gamelog.py` — format/save a game's `action_log` to a text or JSON file

## What's implemented

- All 26 base-set kingdom cards plus the 7 basic cards,
  full Action/Buy/Cleanup turn structure, Moat reactions, supply setup and
  scaling for 2-4 players, and both game-end conditions (Provinces empty,
  or any 3 supply piles empty).
- A regression suite (`tests/test_game.py`) covering the trickier cards
  (Moat blocking, Throne Room composition, Bandit/Sentry/Library reveal
  order, Merchant's turn-scoped bonus, Gardens' deck-size VP) plus a sweep
  that random-plays every kingdom card to completion.

## The training layer: Domibot

[`training/`](training/README.md) is the RL-facing side, kept separate from
the engine: a fixed-size observation/action encoding, a Gym-shaped
`DominionEnv`, baseline agents (`RandomAgent`, `BigMoneyAgent`), and
**Domibot** — a policy/value network (PyTorch, GPU-ready) trained via
AlphaZero-style MCTS self-play (`training/mcts.py`, `training/self_play.py`,
`training/train.py`). `python -m training.train` runs the self-play loop;
`python -m training.evaluate` and `training.agents.DomibotAgent` let you
measure it against the baselines. See `training/README.md` for the full
picture, including the one real scope tradeoff worth knowing before
extending it: MCTS searches only the play/buy decisions, not card-effect
sub-decisions like Chapel's trashes.

## What's not here yet (next layers)

- Domibot's training loop is a first working version, not a tuned one —
  see "What's still missing" in `training/README.md`.
- Anything beyond the base set (no other expansions, no >4 players).
