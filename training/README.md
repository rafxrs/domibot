# training

The RL-facing layer on top of the `domibot` game engine. Kept separate from
the `domibot` package itself: the engine has no opinion on how it gets
trained against, and this folder is where that opinion lives.

## What's here

- **`encoding.py`** — the fixed vocabularies everything else is built on:
  `CARD_NAMES`/`CARD_INDEX` (every base+kingdom card), `ACTION_VOCAB`/
  `ACTION_INDEX` (206 actions any card effect can produce — `Game.legal_actions()`
  is always a subset), `encode_observation(game, player_idx)` (fixed-size
  (350,) float32 vector, respects hidden info: opponents expose only what's
  actually public — discard, play area, hand/deck *sizes*, never contents),
  `legal_action_mask(game)`.
- **`env.py`** — `DominionEnv`, a Gym-shaped (`reset`/`step`) masked-discrete-
  action wrapper around `Game`. Turn-based multi-agent: each `step` acts for
  whoever `Game.current_decider()` is and returns the next observation from
  *their* perspective. Reward is sparse: 0 until the game ends, then
  +1/-1/0 from whoever just moved.
- **`agents.py`** — the `Agent` protocol (`act(game) -> Action`) plus
  `RandomAgent` and `BigMoneyAgent` (fixed-priority Treasures/Victory, never
  buys Action cards).
- **`evaluate.py`** — `play_match(agent_a, agent_b, n_games)`, a head-to-head
  series across random kingdoms. `python -m training.evaluate [n_games] [seed]`
  runs BigMoney vs Random as a sanity check.

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
  automatically. `.save(path)` / `DomibotNet.load(path)` for checkpointing.
- **`mcts.py`** — PUCT search (`run_mcts`), AlphaZero-style. Searches and
  learns every decision uniformly — phase actions *and* card-effect
  sub-decisions (Chapel's trashes, Militia's forced discard, ...). A node's
  position is `(boundary, path)` rather than a raw `Game` (card effects are
  suspended generators, which `Game.clone()` can't safely copy) —
  `materialize()` replays `path` from `boundary` on demand to reach the real
  state; see the module docstring for why replay is always deterministic.
- **`heuristics.py`** — `heuristic_reaction(game)`, the fixed non-learned
  fallback `BigMoneyAgent` always uses for sub-decisions (and
  `DomibotAgent`/self-play use only as an ablation,
  `search_sub_decisions=False` — by default they search sub-decisions via
  MCTS instead).
- **`self_play.py`** — `play_self_play_game(network, num_simulations, ...)`
  plays one MCTS-guided game and returns one training `Example` per
  decision (encoded state, legal mask, visit-count policy target, and the
  eventual outcome as the value target). `ReplayBuffer` is a fixed-capacity
  FIFO of these.
- **`train.py`** — the loop: self-play → buffer → gradient steps (policy
  cross-entropy vs. visit counts, value MSE vs. outcome) → checkpoint →
  eval vs `BigMoneyAgent` every `--eval-every` iterations.
  `--reference-checkpoint <path>` adds a fixed past checkpoint as a second
  eval opponent.
- `agents.DomibotAgent` wraps a trained network + MCTS behind the same
  `act(game)` interface as the baselines.

### Training

```bash
python -m training.train
```

Key flags (see `python -m training.train --help`): `--iterations`,
`--games-per-iter`, `--simulations` (MCTS sims/move, stronger but slower),
`--action-bias` (self-play-only nudge toward chaining Action cards instead
of ending the phase early — most effective set from a fresh network, not
added mid-training), `--eval-every`, `--eval-games`,
`--reference-checkpoint <path>` (a second, fixed eval opponent),
`--checkpoint <path>` to resume. Checkpoints land in `checkpoints/`
(gitignored) as `latest.pt` plus a snapshot every eval.

### GPU

`get_device()` uses CUDA automatically once torch is installed against a
CUDA index (plain `pip install torch` grabs the CPU-only build):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

If that tag doesn't have kernels for your GPU, `torch.cuda.is_available()`
can still say `True` and only fail at actual kernel launch — confirm with:

```bash
python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); print((x @ x).sum().item())"
```

Self-play batches leaf evaluations across `--parallel-games` concurrent
games, so raising it toward your card's throughput sweet spot affords
higher `--simulations`. Past some point the unbatched Python game engine,
not the network, becomes the bottleneck.

### Playing against / evaluating Domibot

```python
import torch
from training.agents import DomibotAgent, BigMoneyAgent
from training.evaluate import play_match
from training.network import DomibotNet, get_device

device = get_device()
network = DomibotNet.load("checkpoints/domibot2/domibot2.1.pt", map_location=device).to(device)
domibot = DomibotAgent(network, num_simulations=200, device=device)

print(play_match(domibot, BigMoneyAgent(), n_games=50))
```

### Testing against real people: `examples/domibot_relay.py`

A move advisor for playing a real game yourself (e.g. on dominion.games)
while asking Domibot what it would do at each decision -- no clicks
automated, you play every move. `training/relay.py` reconstructs a `Game`
from exactly what's visible at the table (your own hand/total ownership
exactly; the opponent's discard and hand/deck *sizes*, never contents),
filling in what's genuinely hidden via determinization. Only covers
phase-action decisions, not sub-decisions.

`training/log_parser.py` does a full turn-by-turn replay of a pasted
dominion.games log to derive everything automatically -- your hand,
discard, play area, phase, actions/buys/coins, and the opponent's hand
size, draw-pile size, and discard -- so the common case skips straight to
the recommendation with no manual entry. Best-effort: dominion.games
occasionally renders a card as a bare, unnamed "a card", which falls back
to manual entry (pre-filled with whatever it did derive).

```bash
python examples/domibot_relay.py --checkpoint checkpoints/domibot2/domibot2.1.pt --simulations 400
```

## Checkpoint lineage and results

`checkpoints/` (gitignored) holds each promoted checkpoint: `domibot_vX.Y.pt`
(v1-v4, MCTS) or `domibotN.M.pt` (domibot 2+, PPO). Full narrative history
is in git log; this table is the durable summary.

| Checkpoint | Change from previous | Key measured result |
|---|---|---|
| v1.1-v1.4 | original network, resumed across sessions | 68% vs BigMoney by v1.4; never chains actions |
| v2.1 | fresh net, action-continuation bias from iter 1, 500 sims | 71-27-2 vs v1.4; real multi-card engine turns seen |
| v2.2 | +100 iters, same settings | round-robin: beat v2.1 57-36-7 |
| v3.1 | fresh net, sub-decisions searched via MCTS (not fixed heuristic) | far behind v2.2, expected for a harder, fresh lineage |
| v3.2 | +200 iters | ~17% vs v2.2; spams Militia (needs no sub-decision), avoids chaining cards |
| v3.3 | +200 iters | ~20% vs v2.2, decelerating |
| v3.4 | +100 iters, curriculum forcing sub-decision-heavy kingdoms | 27.5% vs v2.2 |
| v3.5 | +300 iters, same curriculum | 31.25% vs v2.2 |
| v3.6 | +300 iters, same curriculum | regressed (flat LR across 1200+ iters); not promoted |
| v3.7 | +300 iters from v3.5, determinization ensembles (imperfect-info fix) | 35% vs v2.2, best of v3.x |
| v4.1 | fresh net: 3 engine bugs fixed, `OBS_DIM` 350→419, LR decay, ensembles carried forward | 11/60 vs BigMoney -- a regime fix, not a power play |
| v4.2 | +200 iters, warm-restart LR cycle | 28/60 peak |
| v4.3 | +200 iters, warm-restart LR cycle | 35/60 peak |
| v4.4 | `action_bias` x2.75 (no effect); TD-bootstrap + opponent pool | 67%/60% vs BigMoney/+terminal; still never chains actions |
| (unpromoted) | same TD+pool recipe, fresh network from scratch | 35%/25%, still no chaining -- rules out entrenchment as the cause |
| **domibot2.1** | **new algorithm: PPO + GAE, no tree search** (see below) | **95-100% vs BigMoney on fixed kingdoms; real multi-action engine turns** |

**The handful of decisions that actually mattered**, in order:
1. **v1→v2**: training the action-continuation bias in from iteration 1 on
   a fresh network, instead of retrofitting it onto a converged one (which
   had zero effect -- PUCT's exploitation term dominates the prior once
   value estimates are already confident).
2. **v2→v3**: sub-decisions (trash/discard/gain/topdeck) switched from a
   fixed heuristic to being searched and learned via MCTS -- a much harder
   decision space, hence v3.1's weak start.
3. **v3.6→v3.7**: fixed the "strategy fusion" problem (every self-play
   search only ever explored the *one* concrete hidden deal that game
   actually had) via multi-determinization PIMC
   (`mcts.redeal_hidden_info`/`run_mcts_ensemble`); LR decay followed.
4. **v3→v4**: a full-codebase audit found three real engine bugs (game-end
   timing, indistinguishable same-shape decisions, cards resolving
   invisibly instead of staged in `set_aside`) that changed `OBS_DIM`,
   forcing a fresh network.
5. **v4.2 strategy check**: `domibot_v4.2.pt` never played two action cards
   in the same turn on a kingdom stocked with engine pieces. This became
   the real question for the rest of the project: not win rate, but
   whether it ever discovers multi-step strategy.
6. **v4.4 diagnostics**: `action_bias` x2.75, TD-bootstrapped targets, and
   an opponent pool produced a large strength jump but *still* zero
   chaining, including from a fresh network. Five conditions, one
   conclusion: needed a different algorithm, not another patch.
7. **domibot 2**: see below.

**Naming, from domibot 2 on**: promoted checkpoints are `domibotN.M.pt`
(no `v`, no underscore). Logs/checkpoints are split by lineage:
`logs/domibot1/` / `checkpoints/` for MCTS, `logs/domibot2/` /
`checkpoints/domibot2/` for PPO.

## domibot 2: PPO self-play

Built alongside (not replacing) the MCTS lineage, once five conditions
(plain self-play, `action_bias` x2.75, TD-bootstrapped targets, an
opponent pool, TD+pool from scratch) all failed to produce durable engine
play. Diagnosis: every MCTS value target is a Monte-Carlo return, and pure
self-play only has to beat itself, so a half-built engine reliably loses
to tuned Big Money with nothing rewarding the climb to a well-executed
one. PPO's GAE fixes credit assignment structurally and needs no tree
search at data-generation time.

**Reused unchanged**: the `domibot` engine, `encoding.py`,
`network.DomibotNet` (nothing MCTS-specific about it), `env.DominionEnv`
(gained one addition: an optional `reward_fn`), `evaluate.py`/`agents.py`.
`mcts.py` stays too, for a future inference-time search layer on a
PPO-trained network (not yet wired up).

**New `training/ppo/` subpackage**: `gae.py` (`compute_gae`, generalizing
`self_play._backfill_value_targets`'s per-decider pattern to full GAE),
`rollout.py` (`collect_rollouts`, N `DominionEnv` instances stepped side
by side sharing one batched forward pass per round -- no MCTS tree, so no
`boundary`/`path`/`materialize` needed), `train.py` (clipped surrogate,
value MSE, entropy bonus, advantage normalization). 7 tests in
`tests/test_ppo.py`.

**Running it**:

```bash
python -m training.ppo.train
```

Key flags (see `python -m training.ppo.train --help`): `--iterations`,
`--games-per-iter`, `--lr`/`--lr-final-frac` (cosine LR decay),
`--entropy-coef` (PPO's exploration driver, the closest analogue to MCTS's
`--action-bias`), `--opponent-pool-size`/`--opponent-pool-frac`,
`--eval-every`/`--eval-games`, `--eval-reference-checkpoint <path>` (a
fixed MCTS checkpoint as a second eval opponent, via real search --
`--eval-reference-every` lets it run less often than the cheap BigMoney
evals, since it dominates wall-clock time otherwise), `--checkpoint <path>`
to resume. Checkpoints land in `checkpoints/domibot2/` as
`domibot2_latest.pt` plus a snapshot every eval.

A longer run, resumed from the current strongest checkpoint and
backgrounded with its output logged to a file:

```bash
python -m training.ppo.train \
    --iterations 4000 --games-per-iter 64 \
    --checkpoint checkpoints/domibot2/domibot2.1.pt \
    --eval-every 20 --eval-games 20 \
    --eval-reference-checkpoint checkpoints/domibot_v4.4.pt --eval-reference-every 100 \
    > logs/domibot2/my_run.log 2>&1 &
```

**Training arc** (fresh network through 8000 iterations, all resumed
continuations of the same lineage):
- **1-400**: default hyperparameters. Completed in under an hour (~250x
  faster than MCTS's per-iteration cost, no simulation budget to pay
  for). `iter_400` chains actions on 28% of turns on a Witch-free
  engine-rich test kingdom (up to 5 plays deep) -- something no MCTS
  checkpoint ever showed. Beat `domibot_v4.4.pt` 30-28-2 head to head.
- **401-4400**: plateaued by 2400 (win-rate trend flat, entropy decaying).
  A warm restart (LR decay + `entropy_coef` 0.01→0.03) broadened chaining
  specifically on the kingdom where it was weakest (Witch: 7%→20%). Final
  eval: 85%/75%/80% vs BigMoney/BigMoney+terminal/`domibot_v4.4.pt`.
- **4401-8000**: an opponent pool (30% of games vs. a frozen recent
  checkpoint) was a wash, not a repeat win -- chaining moved in opposite
  directions on the two test kingdoms (20%→13% / 24%→28%), likely because
  the pool's snapshots were too recent to add real diversity. Final eval:
  100%/65%/80%.

Promoted `iter_8000` as `domibot2.1.pt`, marking the end of this first
training arc (superseding an earlier interim promotion of `iter_400`
under the same name).

**Not yet done**: Stage 2 (privileged critic), Stage 4 (inference-time
search for the relay tool), a wider/older opponent pool, further
hyperparameter tuning, longer runs.

## What's still missing

The relay tool only recommends sub-decisions for one case -- Militia's
forced discard -- when a pasted log ends with the opponent having just
played it. Every other sub-decision (Bureaucrat/Bandit's forced reactions,
your own mid-turn choices like an unresolved Chapel trash) falls back to
manual entry. Also missing: `mcts.run_mcts_batch`'s root-parallel batching
doesn't support heterogeneous per-root simulation budgets; hyperparameter
tuning and longer training runs than anything validated so far.
