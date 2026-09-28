# training

The RL-facing layer on top of the `domibot` game engine, kept separate from
it: the engine has no opinion on how it gets trained against.

Two training approaches were built here, in order. **Phase 1** (MCTS
self-play, AlphaZero-style) never learned to chain action cards despite
five separate fix attempts, and was abandoned. **Phase 2** (PPO) is the
current approach and the strongest bot in the project. Both share the same
engine, encoding, and network.

## What's here

- **`encoding.py`** — fixed vocabularies: `CARD_NAMES`/`ACTION_VOCAB` (206
  actions any card effect can produce), `encode_observation` (a (419,)
  float32 vector that respects hidden info — opponents expose only their
  discard, play area, and hand/deck *sizes*), `encode_public_extras` (138
  more public features: scores, opponent card ownership, kingdom, empty
  piles), `encode_own_zones` (101 more: the player's own draw pile, discard
  pile and play area), `legal_action_mask`.
- **`env.py`** — `DominionEnv`, a Gym-shaped masked-discrete-action wrapper
  around `Game`. Each `step` acts for whoever `Game.current_decider()` is;
  reward is sparse (0 until game end, then +1/-1/0).
- **`agents.py`** — the `Agent` protocol (`act(game) -> Action`) plus
  `RandomAgent`, `BigMoneyAgent`, and `DomibotAgent` (a network + MCTS).
- **`evaluate.py`** — `play_match(agent_a, agent_b, n_games)`, a head-to-head
  series across random kingdoms, and `wilson` for its confidence interval.
- **`strategy_profile.py` / `strategy_bots.py` / `gauntlet.py`** — what a
  checkpoint actually plays (which cards it buys and plays), scripted bots
  for specific strategies it doesn't play, and the gauntlet that pits it
  against them (see "Card use" below).
- **`network.py`** — `DomibotNet`, a residual MLP (observation → 256 wide ×
  4 blocks by default → policy logits (206) + tanh value), shared by both
  phases. Width, depth, and which extra inputs it reads are saved in each
  checkpoint.
- **`mcts.py`** — PUCT search, built for Phase 1's self-play loop; still
  used today as an inference-time search layer on top of Phase 2's network
  (the relay tool runs it that way).
- **`self_play.py` / `train.py` / `heuristics.py`** — Phase 1's self-play
  loop, training loop, and non-learned sub-decision fallback.
- **`ppo/`** — Phase 2's rollout collection, GAE, training loop, and
  distillation into a larger network.
- **`relay.py` / `log_parser.py`** — the real-game move advisor (below).

## GPU

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

## Phase 1: MCTS self-play (AlphaZero-style) — abandoned

**What it was.** `mcts.py`'s PUCT search driving `self_play.py`'s self-play
loop and `train.py`'s training loop: the standard AlphaZero recipe (policy
trained against MCTS visit counts, value against game outcomes). It
searched and learned *every* decision uniformly — phase actions and
card-effect sub-decisions (Chapel's trashes, Militia's forced discard, ...)
alike. A node's position is a `(boundary, path)` pair rather than a raw
`Game`, since card effects are suspended generators that `Game.clone()`
can't safely copy; `materialize()` replays `path` from `boundary` on
demand, which is deterministic because every random draw goes through
`Game.rng`.

**Checkpoint lineage and results** (`checkpoints/domibot1/domibot_vX.Y.pt`):

| Checkpoint | What changed | Key measured result |
|---|---|---|
| v1.x | original network | ~68% vs BigMoney in its in-training eval; never chains actions |
| v2.x | fresh net, action-continuation bias (nudges self-play toward chaining instead of ending the phase) from iter 1 — retrofitting it onto v1.4 had zero effect | v2.1 beat v1.4 71-27-2 with real multi-card engine turns; v2.2 beat v2.1 57-36-7 |
| v3.x | sub-decisions searched via MCTS instead of a fixed heuristic (much harder decision space), curriculum kingdom sampling, then multi-determinization search to fix "strategy fusion" (each self-play search only explored the *one* hidden deal a game actually had) | ~17% → 35% (28/80 games) vs v2.2 over 7 versions; spams Militia (needs no sub-decision), avoids chaining |
| v4.1 | fresh net after an audit found 3 engine bugs (game-end timing, indistinguishable same-shape decisions, invisible mid-effect cards), plus LR decay | 11/60 vs BigMoney — a regime fix, not a power play |
| v4.2-v4.3 | warm-restart LR cycles | 35/60 peak; v4.2 never plays two action cards in a turn |
| v4.4 | `action_bias` x2.75 (no effect); TD-bootstrapped value targets + opponent pool | 40/60 vs BigMoney, 36/60 vs BigMoney+terminal; still never chains |
| (unpromoted) | same TD+pool recipe, fresh network | 21–36–3 vs BigMoney, 15–44–1 vs +terminal (60 games each); still no chaining — rules out entrenchment as the cause |
| domibot2.1 | Phase 2: PPO + GAE, no tree search | 351–43–6 vs BigMoney (400 games), 159–36–5 vs v4.4 (200 games); real multi-action engine turns |
| **domibot2.2** | **PPO resumed with 4x the games per update and a quarter of the learning rate** | **1130–797–73 vs domibot2.1 (2000 games); 367–26–7 vs BigMoney (400), 169–30–1 vs v4.4 (200)** |

**Why it failed.** It plateaued at Big Money + Witch: from v4.2 on, the
question wasn't win rate but whether it ever discovers multi-step strategy,
and nothing tried made it, even fixes aimed squarely at the diagnosis
below. Every MCTS value target is a Monte-Carlo return (or a short
TD-bootstrap of one), which buries a deferred-payoff card's credit under a
whole game's noise; and pure self-play only has to beat itself, so a
half-built engine reliably loses to tuned Big Money with nothing rewarding
the climb to a well-executed one. Five separate conditions, one
conclusion: it needed a different algorithm, not another patch.

**Running it** (still works, no longer the recommended path):

```bash
python -m training.train
```

Key flags (see `--help`): `--iterations`, `--games-per-iter`,
`--simulations` (MCTS sims/move), `--action-bias`, `--eval-every`,
`--eval-games`, `--reference-checkpoint <path>`, `--checkpoint <path>` to
resume. Checkpoints land in `checkpoints/`. `--parallel-games` batches leaf
evaluations across concurrent games; past some point the unbatched Python
game engine, not the network, becomes the bottleneck.

## Phase 2: PPO — the current bot

**Why PPO.** GAE gives dense, bootstrapped credit to *every* decision from a
real value function, fixing Phase 1's credit-assignment problem
structurally, and needs no tree search at data-generation time — so it's
also ~250x faster per iteration.

**Reused from Phase 1 unchanged**: the `domibot` engine, `encoding.py`,
`network.DomibotNet`, `env.DominionEnv` (gained one addition, an optional
`reward_fn`), `evaluate.py`/`agents.py`, and `mcts.py` (now an
inference-time search layer, used by the relay tool on top of PPO-trained
networks).

**New `ppo/` subpackage**: `gae.py` (`compute_gae`, generalizing
`self_play._backfill_value_targets`'s per-decider pattern to full GAE),
`rollout.py` (`collect_rollouts`: N `DominionEnv` instances stepped side by
side sharing one batched forward pass per round — no MCTS tree, so no
`boundary`/`path`/`materialize` needed; `collect_cross_play_rollouts` for an
opponent pool), `train.py` (clipped surrogate, value MSE, entropy bonus,
advantage normalization, cosine LR decay), `distill.py` (copies a trained
network into a larger one, below), `league.py` (a pool of varied opponents,
below). Tests: `tests/test_ppo.py`, `tests/test_ppo_league.py`.

**Running it**:

```bash
python -m training.ppo.train
```

Key flags (see `--help`): `--iterations`, `--games-per-iter`,
`--lr`/`--lr-final-frac` (cosine LR decay), `--entropy-coef` (PPO's
exploration driver), `--league-*` (varied opponents, below),
`--opponent-pool-size`/`--opponent-pool-frac` (this run's own snapshots),
`--eval-every`/`--eval-games`, `--eval-reference-checkpoint <path>` (a fixed
MCTS checkpoint as a second eval opponent, via real search;
`--eval-reference-every` lets it run less often than the cheap BigMoney
evals, since it otherwise dominates wall-clock time), `--checkpoint <path>`
to resume (a fresh network's size is `--hidden-dim`/`--num-blocks`).
Checkpoints land in `checkpoints/domibot2/` as `<run-name>_latest.pt` plus
a `<run-name>_iter_N.pt` snapshot every eval; `--run-name` defaults to
`domibot2`.

A longer run, resumed from the current strongest checkpoint with the
settings that produced it and backgrounded with its output logged to a
file. Resuming a trained policy at a higher learning rate or with smaller
batches erodes it first (see "Restarting PPO without losing strength"
below):

```bash
python -m training.ppo.train \
    --checkpoint checkpoints/domibot2/domibot2.2.pt --start-iteration 14001 \
    --iterations 2000 --games-per-iter 256 --minibatch-size 1024 \
    --lr 5e-5 --lr-final-frac 0.1 --target-kl 0.02 \
    --eval-every 25 --eval-games 200 \
    --eval-reference-checkpoint checkpoints/domibot1/domibot_v4.4.pt --eval-reference-every 250 \
    > logs/domibot2/my_run.log 2>&1 &
```

`python -m training.ppo.plot_eval logs/domibot2/my_run.log` graphs eval win
rate vs iteration (needs `pip install -e ".[plot]"`). Pass several logs to
merge a resumed run into one history; `--smooth N` plots a rolling mean,
`--vline ITER:LABEL` marks an event. `docs/training_curve.png` is:

```bash
python -m training.ppo.plot_eval logs/domibot2/domibot2_stage1_run{1,2,3,3b}.log \
    logs/domibot2/domibot2_stage3_pool_run1.log logs/domibot2/domibot2.2_run1.log \
    logs/domibot2/domibot2_256x4_control_run1.log --no-trend --smooth 10 \
    --vline "2400:warm restart" --vline "4400:opponent pool" \
    --vline "8000:domibot2.1" --vline "12000:gentler restart" \
    --title "domibot2 (PPO) eval win rate, fresh network -> domibot2.2 (rolling mean of 10 evals)" \
    --out docs/training_curve.png
```

**Training arc** (fresh network through 14000 iterations; promoted
checkpoints are named `domibotN.M.pt`, with logs/checkpoints under
`logs/domibot2/` and `checkpoints/domibot2/`). In-training evals are 20
games each through iteration 8000 (single points noisy, ±~20pp) and 200
after, on different kingdoms each time; the curve below is a rolling mean.
Promotion decisions use larger evals on fixed seeds, given below.

![Training curve](../docs/training_curve.png)

- **1-400**: default hyperparameters, under an hour. `iter_400` chains
  actions on 62/218 turns (28%) on a Witch-free engine-rich test kingdom
  (up to 5 plays deep) — something no MCTS checkpoint ever showed — and
  beat `domibot_v4.4.pt` 30–28–2 over 60 games.
- **401-4400**: plateaued by 2400 (flat win-rate trend, entropy decaying).
  A warm restart (LR decay + `entropy_coef` 0.01→0.03) broadened chaining
  specifically where it was weakest (Witch kingdom: 7% → 20% of turns,
  40/198). Final eval: 17/20, 15/20, 16/20 games vs BigMoney,
  BigMoney+terminal, `domibot_v4.4.pt`.
- **4401-8000**: an opponent pool (30% of games vs. a frozen recent
  checkpoint) was a wash — chaining moved in opposite directions on the two
  test kingdoms (Witch: 20% → 13%, 32/244 turns; Witch-free: 24% → 28%,
  58/209), likely because the pool's snapshots were too recent to add real
  diversity. Final eval: 20/20, 13/20, 16/20.

`iter_8000` is promoted as **`domibot2.1.pt`**, the strongest checkpoint
until `domibot2.2` (superseding an earlier interim promotion of `iter_400` under
the same name). A larger post-hoc eval (raw policy, no search, paired random
kingdoms): 351–43–6 vs BigMoney and 286–103–11 vs BigMoney+terminal over
400 games each, and 159–36–5 vs `domibot_v4.4.pt` (100-sim MCTS) over 200. Download it from the
[Releases page](https://github.com/rafxrs/domibot/releases).

**domibot2.2 attempt (8001-12000): no measurable gain.** Resumed from
`domibot2.1.pt` with four changes, all still in the code:

- *Win-weighted reward* (`--reward-win-weight 0.8`, now the default): 0.8 ×
  win/loss + 0.2 × the old tanh margin. Margin alone scored a 1-point win
  and a 1-point loss as +0.1 and -0.1, so the win/loss line barely
  registered, and 61% of `domibot2.1`'s losses to BigMoney+terminal were
  within 6 VP.
- *Public score features* (`--public-features`, default on):
  `encoding.encode_public_extras` adds both players' VP and the lead,
  opponent card ownership, kingdom membership, and empty piles — all
  visible at a real table, but left for the network to reconstruct before.
  They enter a trained network through a zero-initialized input layer
  (`DomibotNet.with_extra_inputs`), so its outputs are unchanged until it
  learns to use them.
- *Forced-move masking*: ~42% of decisions have exactly one legal action,
  hence no policy gradient. They're left out of the policy and entropy
  terms and the advantage normalization, but still train the value head.
- *KL logging and `--target-kl`*: approximate KL and clip fraction per
  update, and an optional per-epoch early stop.

It ran 4000 iterations at `--lr 2e-4 --lr-final-frac 0.1 --entropy-coef
0.01 --target-kl 0.02` with 200-game evals (`logs/domibot2/domibot2.2_run1.log`;
a first try at `--entropy-coef 0.02` was aborted when entropy doubled and
win rate fell). Win rate vs BigMoney+terminal fell from 74% at the first
eval to ~60% for the next ~1500 iterations and only recovered as the
learning rate decayed; the tuning below explains why. The final
checkpoint, `domibot2_iter_12000.pt`, is no stronger than `domibot2.1`:
1005–921–74 head-to-head over 2000 games (52.1%, 95% CI 49.9–54.3%), and
72.7% vs 72.0% against BigMoney+terminal on the same 2000 games. Not
promoted.

**Larger network, via distillation.** The 2.2 result suggested the 256×4
network had plateaued, so the next experiment is a 512 wide × 6 block
network (3.7M parameters, up from 0.76M). Training that from scratch would
repeat the 8000 iterations it took to reach `domibot2.1` before the extra
capacity could matter, so `ppo/distill.py` first trains it to copy
`domibot2_iter_12000.pt` — as strong as `domibot2.1`, already trained on
the current reward, and reads the public features:

- Data is DAgger-style: the teacher plays the first 30% of iterations;
  after that the student plays and the teacher labels the positions the
  student reaches, which is where an imitator's mistakes compound.
- Loss: KL(teacher ‖ student) over legal actions on non-forced decisions,
  plus MSE to the teacher's value.
- Progress is measured on each fresh batch *before* training on it
  (held-out KL, top-1 agreement with the teacher), and every 50 iterations
  by 200 games vs the teacher and vs BigMoney+terminal.

```bash
python -m training.ppo.distill \
    --teacher checkpoints/domibot2/domibot2_iter_12000.pt \
    --hidden-dim 512 --num-blocks 6 --iterations 400 \
    --out checkpoints/domibot2/domibot2_512x6_distilled.pt
```

It took 400 iterations, ~20 minutes
(`logs/domibot2/domibot2_512x6_distill.log`). Held-out top-1 agreement
with the teacher rose from 41% before any training to 94% by iteration 50
and 97.7% by the end. The student drew the teacher 99–96–5 over the final
200 games, and won 139/200 vs BigMoney+terminal (the teacher scores ~72%).
PPO then resumes from it (the size is read from the checkpoint). The
tradeoff: the student starts inside the teacher's strategy and only
leaves it if PPO finds something better.

**Restarting PPO without losing strength.** The first two attempts to
resume from the distilled network failed the same way: policy entropy
climbed and win rate fell. At the 2.2 run's `--lr 2e-4` the larger
network's policy moved ~3x further per update than the 256×4 one's had
(approximate KL 0.041 vs 0.013). At 1e-4, entropy went 0.15 → 0.42 and
BigMoney+terminal fell to 128/200 within 50 iterations. Both were stopped
(their logs weren't kept).

The 2.2 run had the same signature: entropy rose from ~0.18 to ~0.30 over
its first few hundred iterations while win rate fell, and both recovered
only as the learning rate decayed. A policy that finished its last run at
a low learning rate is sharp. Restarting it at a high one adds parameter
noise, which pushes a near-deterministic policy's entropy up whatever the
entropy bonus is, and its greedy play gets worse until the decay sharpens
it again. The 2.2 run spent most of its 4000 iterations getting back to
where it started.

Six short probes from the distilled network separated the causes
(`logs/domibot2/domibot2_512x6_probe_{A..F}.log`). A–D ran 100 iterations
of 64 games; E–F ran 50 of 256, with 4x the minibatch size. Entropy is the
mean over the first and last quarter of each probe; evals are 200 games at
the midpoint and the end.

| probe | lr | games/iter | entropy coef | entropy | vs BigMoney+terminal |
|---|---|---|---|---|---|
| A | 1e-4 | 64 | 0.003 | 0.22 → 0.32 | 116, 126 |
| B | 5e-5 | 64 | 0.01 | 0.24 → 0.43 | 126, 106 |
| C | 5e-5 | 64 | 0.003 | 0.19 → 0.35 | 116, 123 |
| D | 2e-5 | 64 | 0.01 | 0.17 → 0.23 | 133, 139 |
| E | 1e-4 | 256 | 0.01 | 0.20 → 0.34 | 130, 122 |
| F | 5e-5 | 256 | 0.01 | 0.16 → 0.26 | 135, 143 |

A smaller entropy bonus barely helps (B vs C), so the bonus isn't what
drives the drift. A lower learning rate slows it (B → D). A 4x larger
batch helps at 5e-5 (B → F) but not at 1e-4 (E). F held strength over its
50 iterations, so the first long run used it.

**First long run: the drift came back.** F's settings with cosine decay to
10%, for 2000 iterations of 256 games. Since the recipe changed along with
the network, the same recipe also ran on the 256×4
`domibot2_iter_12000.pt` as a control (`--run-name` keeps their checkpoint
files apart). By iteration ~12650 the two had split:

| | 512×6 | 256×4 control |
|---|---|---|
| entropy | 0.22 → 0.55–0.64 | ~0.12, flat |
| vs BigMoney+terminal (last 15 evals) | ~57% | 73.4% |
| vs `domibot_v4.4` (40 games at 12250, 12500) | 33, then 24 | 35, then 33 |
| approximate KL per update | 0.018 | 0.001 |

The 512×6 run was stopped there (`logs/domibot2/domibot2_512x6_run1.log`).

**The cause: refitting each batch.** PPO makes several passes over each
batch (`--epochs-per-update`, default 4), and every decision in it carries
a noisy advantage. The larger network can fit that noise position by
position (its surrogate loss went ~6x lower than the control's), and
fitting noise moves the policy in random directions, which shows up as
rising entropy. The 256×4 network can't fit it as well and averages it
out instead. Two more probes varied only the number of passes, at F's
settings for 100 iterations (`logs/domibot2/domibot2_512x6_probe_{G,H}.log`):

| passes per batch | entropy, iteration 25 → 100 | vs BigMoney+terminal (evals at 25/50/75/100) |
|---|---|---|
| 4 (the stopped run) | 0.22 → 0.41 | 141, 135, 128, 121 |
| 2 | 0.15 → 0.20 | 136, 133, 148, 144 |
| 1 | 0.13, flat | 147, 141, 134, 147 |

**Second long run.** One pass per batch, otherwise the same; the control
keeps running unchanged:

```bash
python -m training.ppo.train \
    --checkpoint checkpoints/domibot2/domibot2_512x6_distilled.pt --run-name domibot2_512x6e1 \
    --iterations 2000 --games-per-iter 256 --minibatch-size 1024 --epochs-per-update 1 \
    --start-iteration 12001 --lr 5e-5 --lr-final-frac 0.1 --entropy-coef 0.01 --target-kl 0.02 \
    --eval-every 25 --eval-games 200 \
    --eval-reference-checkpoint checkpoints/domibot1/domibot_v4.4.pt \
    --eval-reference-every 250 --eval-reference-games 40 \
    > logs/domibot2/domibot2_512x6_run2_1epoch.log 2>&1 &
```

The control uses 4 passes, `--checkpoint
checkpoints/domibot2/domibot2_iter_12000.pt --run-name domibot2_256x4`,
and logs to `logs/domibot2/domibot2_256x4_control_run1.log`. It also
differs from the 2.2 run (4x the games per update, a quarter of the
learning rate), so it tests whether a gentler restart alone helps the
256×4 network.

**Results.** Both runs finished 2000 iterations. Their final checkpoints
were evaluated on the same seeds as `domibot2.1`, raw policy with no
search (`logs/domibot2/*_iter_14000_eval.log`; 95% CIs in parentheses):

| | 256×4 control | 512×6, one pass | `domibot2.1` |
|---|---|---|---|
| vs `domibot2.1` (2000 games) | 1130–797–73, 58.3% (56.2–60.5%) | 1137–776–87, 59.0% (56.9–61.2%) | — |
| vs BigMoney+terminal (2000) | 76.8% (75.0–78.6%) | 76.1% (74.2–78.0%) | 72.0% (70.0–74.0%) |
| vs BigMoney (400) | 92.6% | 91.4% | 88.5% |
| vs `domibot_v4.4` (200) | 84.8% | 85.2% | 80.8% |

Head-to-head, the two drew: 963–957–80 over 2000 games (50.1%, 95% CI
48.0–52.3%). So:

- **The gain came from the gentler restart, not the size.** The 256×4
  control, resumed from `domibot2_iter_12000.pt` with 4x the games per
  update and a quarter of the 2.2 attempt's learning rate, beat
  `domibot2.1` 58–42. The 2.2 attempt's higher learning rate spent its
  4000 iterations recovering from its own restart instead.
- **5x the parameters bought nothing measurable**, and the larger network
  was harder to train: it needed one pass per batch to stop fitting
  advantage noise.

The control's final checkpoint, `domibot2_256x4_iter_14000.pt`, is
promoted as **`domibot2.2.pt`**: as strong as the 512×6 network, a fifth
the size, and faster for the relay tool's search. Download it from the
[Releases page](https://github.com/rafxrs/domibot/releases).

**Continuing 2.2 (14001-16000): plateaued.** Its last 500 iterations were
still gaining against BigMoney+terminal (72.6% -> 74.9% over four
500-iteration blocks) as its learning rate ran out, so the same recipe
ran for another 2000 iterations from `domibot2.2.pt` with a fresh
learning-rate cycle (`logs/domibot2/domibot2_256x4_run2.log`, same
command as above with `--checkpoint checkpoints/domibot2/domibot2.2.pt
--start-iteration 14001`). In-training evals stayed flat (~75% vs
BigMoney+terminal, ~92% vs BigMoney), and no checkpoint beat
`domibot2.2` head-to-head over 2000 games: 49.3%, 50.2%, 50.1% at
iterations 14500/15000/15500, and 51.5% (95% CI 49.3-53.7%) at 16000,
whose baselines (77.5% vs BigMoney+terminal over 2000, 94.5% vs BigMoney
over 400, 86.8% vs `domibot_v4.4` over 200) are also within noise of 2.2's.
Not promoted. With a 5x larger network landing at the same strength too,
more of the same self-play is unlikely to help; varying the opponents is
the next lever.

**Opponent league: no gain either.** `ppo/league.py` plays part of every
iteration's games against a pool of fixed, genuinely different opponents
instead of the current network. Only the learner's seat produces training
data, and network opponents play their own raw policy. The pool:

- past checkpoints from across the arc (`domibot2_iter_{2000..12000}`, every
  2000 iterations, `iter_8000` being `domibot2.1`): the strategies it
  played on the way here;
- `domibot2_512x6e1_iter_14000`: 2.2's strength, but a separately trained
  network;
- `domibot_v4.4`'s network (the MCTS lineage's Big Money + Witch player);
- scripted Big Money and Big Money + terminal -- the latter still takes
  ~23% of games off 2.2;
- a frozen copy of the learner every 250 iterations (the last 4 kept).

Opponents are drawn by prioritized fictitious self-play (PFSP, as in
AlphaStar): weight (1 - p)^2, where p is the learner's recent score against
that opponent (wins + half of ties, decayed 5% per iteration so it tracks
the current learner), with 20% of the weight spread evenly so no opponent
disappears. Each iteration draws 4 opponents and splits the league games
between them; the log shows each one's result (`league=name:score/games`)
and, at every eval, the whole pool's scores and weights.

The run keeps 2.2's recipe, with half of its 256 games per iteration going
to the league, and evals against `domibot2.2`'s raw policy every 25
iterations (`--eval-rival-checkpoint`) to track the promotion bar as it
goes:

```bash
python -m training.ppo.train \
    --checkpoint checkpoints/domibot2/domibot2.2.pt --run-name domibot2_league \
    --iterations 2000 --games-per-iter 256 --minibatch-size 1024 --start-iteration 14001 \
    --lr 5e-5 --lr-final-frac 0.1 --entropy-coef 0.01 --target-kl 0.02 \
    --league-frac 0.5 --league-opponents-per-iter 4 \
    --league-snapshot-every 250 --league-max-snapshots 4 \
    --league-checkpoints checkpoints/domibot2/domibot2_iter_{2000,4000,6000,8000,10000,12000}.pt \
        checkpoints/domibot2/domibot2_512x6e1_iter_14000.pt checkpoints/domibot1/domibot_v4.4.pt \
    --league-scripted bigmoney bigmoney_terminal \
    --eval-every 25 --eval-games 200 --eval-rival-checkpoint checkpoints/domibot2/domibot2.2.pt \
    --eval-reference-checkpoint checkpoints/domibot1/domibot_v4.4.pt \
    --eval-reference-every 250 --eval-reference-games 40 \
    > logs/domibot2/domibot2_league_run1.log 2>&1 &
```

It ran all 2000 iterations (`logs/domibot2/domibot2_league_run1.log`)
and stayed level with 2.2 throughout: the in-training eval against it
averaged 47.7-50.6% in every 250-iteration block, and the final checkpoint
went 974–954–72 over 2000 games (50.5%, 95% CI 48.3-52.7%). Not promoted.
BigMoney+terminal (~75%) and BigMoney (~92%) didn't move either. Against
the league itself it gained a few points on the older opponents (e.g.
`iter_8000`, i.e. 2.1: 60% -> 68%; `iter_10000`: 68% -> 74%;
BigMoney+terminal: 74% -> 78%; 32-game samples, so noisy) but none against
the ones at its own level (the 512×6 network: 48% -> 47%; its own
snapshots: ~50%).

Three different changes -- more self-play, a 5x larger network, and a
varied league -- all land exactly at 2.2's strength, so what the raw policy
can learn this way looks exhausted. Meanwhile the relay tool (2.2 with
400 simulations of search per move) went 5-0 against human players rated
around and above the relay account's own dominion.games rating (37-45),
which suggested search as the next lever.

**What search adds: nothing measurable.** `ppo/search_eval.py` plays 2.2
with search against 2.2's raw policy on paired random kingdoms. A fair
search (`agents.DeterminizedSearchAgent`) searches only what a player could
know, like the relay: the opponent's hand and deck reshuffled together, its
own deck order reshuffled, a fresh random seed. Within one search every
simulation replays the same copy, so it sees one sampled future;
`--determinizations K` searches K independent copies and adds up their
visits instead. Card-effect choices use the raw policy. Results
(`logs/domibot2/search_eval/`):

| search | vs raw 2.2 |
|---|---|
| 100 simulations | 480–479–41 over 1000, 50.0% (95% CI 47.0-53.1%) |
| 400 (the relay's setting) | 906–982–112 over 2000, 48.1% (45.9-50.3%) |
| 1600 | 235–244–21 over 500, 49.1% (44.7-53.5%) |
| 8 copies x 50 | 974–940–86 over 2000, 50.8% (48.7-53.0%) |
| 16 copies x 100 | 479–474–47 over 1000, 50.2% (47.2-53.3%) |
| 400, seeing the true game (hidden hand, deck order) | 498–457–45 over 1000, 52.0% (49.0-55.1%) |

Against BigMoney+terminal, 400-simulation search scored 77.8% (764–208–28
over 1000), level with the raw policy's 76.8%.

Why: the search almost never changes the move. Over 24 games of 2.2
against itself, 400-simulation search picked the policy's own move on
811/815 action-phase decisions and 988/1015 buys; the 8-copy search on
815/815 and 1013/1015. 2.2's policy is sharp (entropy ~0.1), so the search
follows its prior unless the value head sees a clear difference, and the
value head isn't accurate enough to see one -- even searching the true
game, with no hidden information at all, gains only ~2 points. The rare
single-copy disagreements are mostly noise from its one sampled future.

So the 5-0 against humans was the network's own play, and expert
iteration (training the policy toward search results) has nothing to
learn from while the search agrees with the policy. A better value
estimate is what search, and anything built on it, would need first.

`--perfect-info` in `ppo/search_eval.py` uses `agents.DomibotAgent`, which
with its default settings searches the true game state. That is also how
`domibot_v4.4` searched in every "vs `domibot_v4.4` (100-sim MCTS)" eval
in this README, including the in-training ones: it saw its opponent's hand
and every deck's order.

**Card use: 2.2 has stopped buying a third of the kingdom.**
`strategy_profile.py` plays a checkpoint's raw policy against itself on
random kingdoms and reports, for each kingdom card, how often it's in a
deck at the end of games where it was available, and how often it's
bought and played:

```bash
python -m training.strategy_profile checkpoints/domibot2/domibot2.2.pt --games 600 --workers 6
```

Over 600 games (`logs/domibot2/gauntlet/domibot2.2_profile.log`), eight
cards end up in fewer than 1% of 2.2's decks: Throne Room, Chapel,
Workshop, Artisan, Remodel, Mine, Moneylender and Vassal, with Village in
6%. What it does use: Sentry (87% of decks), Witch (81%), Militia (78%),
Gardens (70%), Bandit, Market and Poacher. With Throne Room, Village and
Smithy put in every kingdom (400 games,
`domibot2.2_profile_throne_village_smithy.log`) it never played a single
Throne Room. 44% of its turns play no Action and 1.5% play eight or more.

Earlier checkpoints show when the cards went (200 games each,
`logs/domibot2/gauntlet/domibot2_iter_N_profile.log`; share of decks
holding the card at the end):

| checkpoint | Throne Room | Village | Chapel | Workshop | Sentry | Throne Room on Throne Room |
|---|---|---|---|---|---|---|
| iter 400 | 1% | 1% | 2% | 0% | 2% | never |
| iter 1000 | 0% | 6% | 15% | 0% | 0% | never |
| iter 2000 | 27% | 27% | 4% | 14% | 0% | 3 times |
| iter 4000 | 2% | 1% | 2% | 4% | 16% | never |
| iter 8000 (`domibot2.1`) | 0% | 7% | 0% | 1% | 61% | never |
| iter 12000 | 1% | 1% | 4% | 0% | 83% | never |
| `domibot2.2` (600 games) | 0% | 6% | 0% | 0% | 87% | never |

So Throne Room and Village were in use around iteration 2000 (Throne Room
most often on Witch, three times on another Throne Room) and gone by 4000,
while Sentry took over; Chapel peaked at 15% around iteration 1000. The
likely mechanism: bought before the policy could play them well, these
cards lost to Silver and Gold, their buy probability fell toward zero, and
from then on on-policy PPO never produced a deck holding them, so neither
the value head nor the policy could learn what they're worth. With the
policy's entropy at ~0.1 and its per-update KL at 0.0001 by the end of the
last run, they don't come back on their own. Everything tried since
inherits this: more self-play, a bigger network, a league of 2.2's own
ancestors (none of which play these cards either) and search (which uses
the same policy and value estimate).

**The strategy gauntlet.** `strategy_bots.py` scripts three well-known
strategies built on cards 2.2 doesn't use, and `gauntlet.py` plays a
checkpoint's raw policy against each, on kingdoms containing the cards the
strategy needs (plus random others), from both seats. BigMoney+terminal
plays the same kingdoms and seeds for comparison:

```bash
python -m training.gauntlet checkpoints/domibot2/domibot2.2.pt --kingdoms 400 --workers 6
```

Against 2.2, 400 kingdoms x 2 seats each
(`logs/domibot2/gauntlet/domibot2.2_gauntlet.log`; the bot's own strength
is its score against BigMoney+terminal on the same kind of kingdom):

| bot | its cards | bot vs BigMoney+terminal | 2.2 vs bot | 2.2 vs BigMoney+terminal, same kingdoms |
|---|---|---|---|---|
| Workshop/Gardens | Workshop, Gardens | 58.6% | 65.8% (95% CI 62-69%) | 83.2% |
| Chapel/Witch | Chapel, Witch | 56.1% | 72.8% (70-76%) | 72.9% |
| Throne Room engine | Chapel, Throne Room, Village, Smithy, Market, Militia | 28.6% | 90.3% (88-92%) | 73.0% |

The Workshop/Gardens rush, about twenty lines of buy priorities, takes 34%
of its games off 2.2, twice what BigMoney+terminal manages on the same
kingdoms (17%): 2.2 neither plays that strategy nor has learned to stop
it. Chapel/Witch does exactly as well against 2.2 as BigMoney+terminal.
The Throne Room engine chains Throne Rooms but is the weakest bot (it
loses to BigMoney+terminal too), so its row doesn't test much yet. The
bots are untuned priority lists: a bot scoring badly doesn't prove 2.2
has no gap there. The gauntlet's worst case is a yardstick for any
change aimed at the card-use problem, alongside head-to-head results
against 2.2.

**Own-zone inputs.** The network couldn't see its own deck.
`encode_observation` gives the opponent's discard pile and play area, but
for the player itself only its hand and everything it owns: not how the
rest splits between draw pile, discard pile and play area, nor the draw
pile's size. A player with perfect memory knows all of that (every card it
draws, gains, discards and shuffles is shown to it), and engine play
depends on it: whether to play Village or Smithy first depends on what's
left to draw, and when to buy depends on when the reshuffle comes.
`encoding.encode_own_zones` adds it: the count of each card in the
player's draw pile, discard pile and play area, plus the draw pile and
discard sizes (101 inputs, order within the draw pile still unknown). It
enters after the public extras through its own norm and a
zero-initialized projection (`DomibotNet.with_zone_inputs`), so 2.2
resumes with unchanged outputs, and it's on by default
(`--zone-features`). The relay tool reconstructs these zones already (it
tracks your discard and play area), so it needs no change.

The first run keeps 2.2's recipe exactly, so the only difference from the
2.2 continuation above (51.5% against 2.2 after 2000 iterations) is the new
inputs:

```bash
python -m training.ppo.train \
    --checkpoint checkpoints/domibot2/domibot2.2.pt --run-name domibot2_zones \
    --iterations 2000 --games-per-iter 256 --minibatch-size 1024 --start-iteration 14001 \
    --lr 5e-5 --lr-final-frac 0.1 --entropy-coef 0.01 --target-kl 0.02 \
    --eval-every 25 --eval-games 200 --eval-rival-checkpoint checkpoints/domibot2/domibot2.2.pt \
    --eval-reference-checkpoint checkpoints/domibot1/domibot_v4.4.pt \
    --eval-reference-every 250 --eval-reference-games 40 \
    > logs/domibot2/domibot2_zones_run1.log 2>&1 &
```

Results: in progress.

**Playing against / evaluating it**:

```python
import torch
from training.agents import DomibotAgent, BigMoneyAgent
from training.evaluate import play_match
from training.network import DomibotNet, get_device

device = get_device()
network = DomibotNet.load("checkpoints/domibot2/domibot2.2.pt", map_location=device).to(device)
domibot = DomibotAgent(network, num_simulations=200, device=device)

print(play_match(domibot, BigMoneyAgent(), n_games=50))
```

**Testing against real people** — `examples/domibot_relay.py` is a move
advisor for a real game you play yourself (e.g. on dominion.games); no
clicks are automated. `relay.py` reconstructs a `Game` from exactly what's
visible at the table (your own hand/total ownership exactly; the opponent's
discard and hand/deck *sizes*, never contents), filling in what's genuinely
hidden via determinization, then runs MCTS on top of the network.
`log_parser.py` does a full turn-by-turn replay of a pasted dominion.games
log to derive everything automatically, so the common case skips straight
to the recommendation. Anything it can't follow exactly (a bare, unnamed
"a card" in your own lines, or a line it doesn't recognize) stops the
replay, and the tool falls back to manual entry pre-filled with whatever it
did derive; so does a derived state that doesn't add up. Besides phase
decisions it covers the choices a log can leave pending:
- **your own card, mid-effect** (what Chapel trashes, what Throne Room
  plays, whether to play what Vassal discarded, Remodel/Mine/Workshop/
  Artisan gains, Sentry, Harbinger, Poacher, Cellar): `relay.replay_open_play`
  replays the card through the engine with your deck stacked with the cards
  the log shows you drawing, plus every choice the log already shows. A
  choice made in several steps with nothing new revealed in between is
  shown as the whole sequence.
- **the opponent's attack**: Militia's discard, Bureaucrat's topdeck,
  Bandit's trash, and whether to reveal Moat.

Cards known to be on top of a deck (Sentry/Harbinger/Artisan topdecks,
Bureaucrat's Silver) are placed there rather than shuffled in, and cards
known to be in the opponent's hand (a Moat they reacted with, a hand
Bureaucrat revealed) are dealt to them rather than left to chance.
`tests/test_log_parser_edge_cases.py` replays thirteen real games (eight
kept whole in `tests/fixtures/dominion_logs/`) and requires every point a
paste could end at to parse and reconstruct.

Still unverified against real logs, so handled defensively: your own
Library (its set-aside lines stop the replay). Not covered: a
reaction to a Throne-Roomed attack or to one played after the opponent's
Council Room, a card that reshuffles your deck twice, the order two Sentry
cards go back in, and games with more than two players.

```bash
python examples/domibot_relay.py --checkpoint checkpoints/domibot2/domibot2.2.pt --simulations 400
```

**Not yet done**: getting the unused cards back into play -- games where
one player is steered into buying a given card so the value head learns
what decks holding it are worth, plus a minimum buy probability for every
affordable kingdom card during training so the policy's own buys keep
getting feedback -- judged by the card-use profile, the gauntlet and
head-to-head results. Then a better value estimate (a privileged,
full-information critic during training), measured by how well it
predicts outcomes, and whether search on top of it finally helps.
