# training

The RL-facing layer on top of the `domibot` game engine. Kept separate from
the `domibot` package itself: the engine has no opinion on how it gets
trained against, and this folder is where that opinion lives.

## What's here

- **`encoding.py`** — the fixed vocabularies everything else is built on:
  - `CARD_NAMES` (33) / `CARD_INDEX` — every base + kingdom card, sorted.
  - `ACTION_VOCAB` (206) / `ACTION_INDEX` — every `Action` any card effect
    can ever produce (6 card-targeted verbs x 33 cards, plus 8 verb-only
    actions). `Game.legal_actions()` is always a subset of this.
  - `encode_observation(game, player_idx)` — a fixed-size (350,) float32
    vector from one player's point of view. Respects hidden information:
    your own hand/deck is fully known, opponents only expose what's
    actually public in Dominion (discard pile, play area, hand/deck
    *sizes* — never hand or deck *contents*).
  - `legal_action_mask(game)` — boolean mask over `ACTION_VOCAB`.
- **`env.py`** — `DominionEnv`, a Gym-shaped (`reset`/`step`) wrapper around
  `Game` using the standard masked-discrete-action pattern (fixed action
  space + an `action_mask` every observation carries, since which actions
  are legal changes constantly). Turn-based multi-agent: each `step` acts
  for whoever `Game.current_decider()` currently is, and returns the next
  observation from *that* player's perspective — a self-play loop just
  routes each observation to whichever policy controls that seat. Reward is
  sparse: 0 until the game ends, then +1/-1/0 (win/loss/tie) from the
  perspective of whoever just moved.
- **`agents.py`** — the `Agent` protocol (`act(game) -> Action`) plus two
  baselines that operate directly on `Game` (no encoding needed — that's
  only for a network-based policy): `RandomAgent`, and `BigMoneyAgent` (buys
  Treasures/Victory on a fixed priority, never buys Action cards, and falls
  back to simple defaults for any reactive decision an opponent's attack
  forces on it).
- **`evaluate.py`** — `play_match(agent_a, agent_b, n_games)` runs a
  head-to-head series across random kingdoms (alternating who goes first)
  and reports win counts. `python -m training.evaluate [n_games] [seed]`
  runs BigMoney vs Random as a sanity check of the whole stack.

## Quickstart

```bash
python -m training.evaluate 200
```

```python
from training.env import DominionEnv

env = DominionEnv(num_players=2)
obs, info = env.reset(seed=0)
while True:
    legal = obs["action_mask"].nonzero()[0]
    action_idx = legal[0]  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action_idx)
    if terminated or truncated:
        break
```

## Domibot: the trained agent

- **`network.py`** — `DomibotNet`, a small residual MLP (350 → 256 ×4 blocks
  → policy logits (206) + value (tanh, [-1,1])). `get_device()` picks CUDA
  automatically when available. `.save(path)` / `DomibotNet.load(path)`
  handle checkpointing.
- **`mcts.py`** — PUCT search (`run_mcts`), same family of algorithm as
  AlphaZero/AlphaGo. **Scope decision, read this before extending it**: MCTS
  only searches phase-action decisions (what to play, what to buy, when to
  end a phase). Card-effect sub-decisions (Chapel's trashes, Militia's
  forced discard, a Moat reveal, ...) are resolved immediately by the fixed
  heuristic in `heuristics.py`, never searched. Two reasons: (1) *hard
  constraint* — card effects are Python generators that capture a live
  reference to the `Game` they were created against (see `effects.py` in
  the engine), so `Game.clone()` refuses to run while one is suspended;
  MCTS nodes only ever exist at the safe boundaries where cloning works.
  (2) *scope* — the strategic weight of a turn is overwhelmingly in what to
  play/buy, so this gets a working v1 without the sub-decision branching
  factor. Extending search to specific sub-decisions later is possible but
  not done here.
- **`heuristics.py`** — `heuristic_reaction(game)`, the fixed fallback both
  `BigMoneyAgent` and `DomibotAgent` use for sub-decisions, plus
  `advance_to_next_phase_action(game, action)` (apply an action, then keep
  auto-resolving forced sub-decisions until the next phase-action boundary
  or game end) — the one primitive MCTS, self-play, and `DomibotAgent` all
  share for "take one strategic step."
- **`self_play.py`** — `play_self_play_game(network, num_simulations, ...)`
  plays one game with MCTS-guided moves (Dirichlet noise at the root,
  temperature-based sampling for the first ~15 moves, then near-greedy),
  and returns one training `Example` per phase-action decision: the
  encoded state, the legal mask, the MCTS visit-count distribution (policy
  target), and — filled in once the game ends — the actual outcome from
  that decision's perspective (value target). `ReplayBuffer` is a fixed-
  capacity FIFO of these.
- **`train.py`** — the loop: self-play games → add to buffer → gradient
  steps (policy = cross-entropy vs. visit counts, value = MSE vs. outcome)
  → checkpoint → every `--eval-every` iterations, play the current network
  (via MCTS) against `BigMoneyAgent` and report the score, so you can watch
  it actually improve over time. Pass `--reference-checkpoint <path>` to
  also eval against a fixed past checkpoint (loaded once, frozen for the
  whole run) alongside BigMoney — useful for comparing a new network
  lineage against an older one on equal footing.
- `agents.DomibotAgent` wraps a trained network + MCTS behind the same
  `act(game)` interface as the baselines, so it drops straight into
  `evaluate.py` against `RandomAgent`/`BigMoneyAgent` (or another
  checkpoint) exactly like any other agent.

### Training

```bash
python -m training.train
```

Key flags (see `python -m training.train --help`): `--iterations`,
`--games-per-iter`, `--simulations` (MCTS sims/move during self-play —
bigger is stronger but slower), `--action-bias` (self-play-only nudge to
keep chaining Action cards instead of ending the phase early — see
`mcts._apply_action_continuation_bias`; most effective when set from a
fresh network rather than added mid-training), `--eval-every`,
`--eval-games` (per opponent), `--reference-checkpoint <path>` (adds a
second, fixed eval opponent), `--checkpoint <path>` to resume. Checkpoints
land in `checkpoints/` (gitignored) as `latest.pt` plus a snapshot every
eval.

### GPU

`get_device()` uses CUDA automatically whenever `torch.cuda.is_available()`
— no code changes needed on a CUDA machine, but `pip install torch` alone
grabs the CPU-only build, so you need torch installed against a CUDA index.

Verified working on an RTX 5080 (Blackwell, compute capability 12.0,
driver supporting CUDA 13.3): the `cu128`/`cu124`-tagged builds people
often see recommended online predate Blackwell-generation wheels catching
up to the latest torch release, and resolved to an *older* torch version
here. The `cu130` index had the exact same torch version as the CPU build
it replaced, with full CUDA support:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

Confirm it actually works (not just `is_available()`, which can be True
even when a build lacks kernels for a brand-new architecture — the failure
mode shows up at actual kernel launch, not at that check):

```bash
python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); print((x @ x).sum().item())"
```

If your card/driver combo resolves to a different tag, list what's on
each index without installing anything: `pip install --force-reinstall
--dry-run torch --index-url https://download.pytorch.org/whl/<tag>` for
whichever `cuNNN` tags exist at https://download.pytorch.org/whl/ , and
pick whichever gives you the newest torch version with CUDA.

Self-play batches leaf evaluations across `--parallel-games` concurrent
games (`mcts.run_mcts_batch` / `self_play.play_self_play_games_batch`), so
raising `--parallel-games` toward your card's real throughput sweet spot is
the lever for affording higher `--simulations` counts. The Python game
engine driving move selection is still real, unbatched CPU work though, so
past some point it — not the network — becomes the bottleneck again.

### Playing against / evaluating Domibot

```python
import torch
from training.agents import DomibotAgent, BigMoneyAgent
from training.evaluate import play_match
from training.network import DomibotNet, get_device

device = get_device()
network = DomibotNet.load("checkpoints/latest.pt", map_location=device).to(device)
domibot = DomibotAgent(network, num_simulations=200, device=device)

print(play_match(domibot, BigMoneyAgent(), n_games=50))
```

## What's still missing

Everything above is a first working version, not a tuned one. Likely next
steps: extending search to at least the highest-value sub-decisions,
hyperparameter tuning (network size, simulation count, `--parallel-games`,
replay buffer size), and longer training runs than anything validated here
(this was only smoke-tested for a couple of iterations to confirm the
pipeline runs end to end without crashing — actual strength after real
training time is unverified).
