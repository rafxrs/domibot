# domibot

A Dominion (base set) game engine, built as the foundational layer for later
training self-play RL agents: at every point there is exactly one player who must make
exactly one choice from an explicit list of legal actions. Play against domibot from the CLI or on a PyGame GUI.

## Setup

```bash
git clone https://github.com/rafxrs/domibot.git
cd domibot
pip install -e ".[dev,train,gui]"
```

Python 3.10+. `train` pulls in `torch` (CPU build by default — see
`training/README.md`'s GPU section to install a CUDA build instead);
`gui` pulls in `pygame`. Drop either extra you don't need.

## Play

- Play a game yourself against a random-move bot (text CLI):
  ```bash
  python examples/play_vs_random.py [seed]
  ```
- Play against a trained Domibot checkpoint (text CLI, or `--gui` for a
  pygame window):
  ```bash
  python examples/play_vs_domibot.py
  python examples/play_vs_domibot.py --gui
  ```
- Watch Domibot play itself (N games, M agents, 2-4 players):
  ```bash
  python examples/play_domibot.py --games 20 --players 2
  ```
- Get move recommendations while playing a real game yourself (e.g. on
  dominion.games):
  ```bash
  python examples/domibot_relay.py
  ```
- Train a new Domibot from scratch — MCTS self-play or PPO (see
  `training/README.md` for the difference and why both exist):
  ```bash
  python -m training.train
  python -m training.ppo.train
  ```
- A longer PPO run, resumed from the current strongest checkpoint and
  backgrounded with its output logged to a file — the reference-checkpoint
  eval uses real MCTS search and dominates wall-clock time, so
  `--eval-reference-every` lets it run far less often than the cheap
  BigMoney evals (see `training/README.md`'s PPO section):
  ```bash
  python -m training.ppo.train \
      --iterations 4000 --games-per-iter 64 \
      --checkpoint checkpoints/domibot2/domibot2.1.pt \
      --eval-every 20 --eval-games 20 \
      --eval-reference-checkpoint checkpoints/domibot_v4.4.pt --eval-reference-every 100 \
      > logs/domibot2/my_run.log 2>&1 &
  ```
- Evaluate a checkpoint against the baseline agents:
  ```bash
  python -m training.evaluate 200
  ```

See `training/README.md` for flags, checkpoint layout, GPU setup, and the
full checkpoint lineage.

## Using the engine directly

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

Every card effect that needs player input is a Python generator; `Game`
drives whichever one is currently suspended (`legal_actions()` returns the
pending `Decision`'s options, `step(action)` resumes it via
`generator.send`). Sub-effects (Throne Room, Vassal replaying a card)
compose with plain `yield from`; `attack_each_opponent` walks opponents in
turn order and inserts Moat's block-or-not automatically.

Variable-length selections (Cellar/Chapel/Militia/Sentry's trash/discard
steps) are a *sequence* of single-card-or-DONE decisions, not one
combinatorial "choose a subset" decision — keeps the legal-action count
bounded by hand size instead of 2^hand_size.

`Action` is a flat `(verb, card_or_None)` pair (`PLAY(Village)`,
`BUY(Silver)`, `TRASH(Copper)`, `DONE`) — hashable, usable directly as a
policy-network output token. `Decision.kind` (`PHASE_ACTION` /
`SELECT_CARD` / `YES_NO` / `REACT`) gives an encoder context for *why* a
verb is legal, since e.g. `YES`/`NO` is reused across unrelated effects
(Library's draw-or-skip, Vassal's play-or-not).

## Layout

Repo root: `src/domibot/` (the engine, below), `tests/`, `examples/` (CLI
scripts), `training/` (the RL layer — see its own README), `gui/` (the
pygame front-end behind `--gui`), `game_logs/` (gitignored, generated).

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
MCTS self-play (`training/train.py`) or PPO (`training/ppo/train.py`). See
`training/README.md` for the full picture.

## What's not here yet (next layers)

- Domibot's training loop is a first working version, not a tuned one —
  see "What's still missing" in `training/README.md`.
- Anything beyond the base set (no other expansions, no >4 players).
