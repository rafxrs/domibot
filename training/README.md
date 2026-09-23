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
  AlphaZero/AlphaGo. Searches and learns *every* Dominion decision — phase
  actions (what to play, what to buy, when to end a phase) *and*
  card-effect sub-decisions (Chapel's trashes, Militia's forced discard,
  Sentry's trash/discard/reorder, ...) — uniformly. A node's position is a
  `(boundary, path)` pair rather than a raw `Game`: `boundary` is the
  nearest ancestor `Game` at a true phase-action boundary (always safely
  clonable — card effects are Python generators that capture a live
  reference to the `Game` they were created against, see `effects.py` in
  the engine, so `Game.clone()` refuses to run while one is suspended), and
  `path` is the sub-decision `Action`s taken since that boundary.
  `materialize()` reconstructs the actual position on demand by cloning
  `boundary` and replaying `path` — safe because `Game.rng`'s state is
  fully captured by `clone()` and every random draw inside a card effect
  goes exclusively through it, so replay always reaches bit-identical
  state. See the module docstring for the full reasoning.
- **`heuristics.py`** — `heuristic_reaction(game)`, the fixed, non-learned
  fallback `BigMoneyAgent` always uses for sub-decisions, and that
  `DomibotAgent`/self-play fall back to only as an ablation
  (`search_sub_decisions=False`) — by default they search and learn
  sub-decisions via MCTS instead. Also `advance_to_next_phase_action(game,
  action)` (apply an action, then keep auto-resolving forced sub-decisions
  via the heuristic until the next phase-action boundary or game end),
  used by that ablation path and by `BigMoneyAgent`.
- **`self_play.py`** — `play_self_play_game(network, num_simulations, ...)`
  plays one game with MCTS-guided moves (Dirichlet noise at the root,
  temperature-based sampling for the first ~15 moves, then near-greedy),
  and returns one training `Example` per decision — phase action or
  sub-decision alike: the encoded state, the legal mask, the MCTS
  visit-count distribution (policy target), and — filled in once the game
  ends — the actual outcome from that decision's perspective (value
  target). `ReplayBuffer` is a fixed-capacity FIFO of these.
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

### Testing against real people: `examples/domibot_relay.py`

A move advisor for playing a real game yourself (e.g. on dominion.games)
while asking Domibot what it would do at each of your decisions, without
actually automating any clicks -- you play every move, this only tells you
what it recommends. `training/relay.py` reconstructs a `Game` from exactly
what's visible to a player at the table (your own hand/total ownership
exactly; the opponent's discard pile and hand/deck *sizes*, never their
contents), filling in what's genuinely hidden (the opponent's hand/deck
contents, your own deck's order) via determinization -- see that module's
docstring for the details. Only covers phase-action decisions: state is
always reconstructed at a phase-action boundary, so a sub-decision can't be
represented here even though `mcts.py` itself searches those too now when
driving self-play/`DomibotAgent` directly.

In practice this is closer to zero-effort than that description suggests:
`training/log_parser.py` does a full turn-by-turn replay of a pasted
dominion.games log (plays, buys/gains, trashes/discards/topdecks including
Sentry/Bandit-style reveals and Throne-Room replays, explicit "+N
Action/Buy/$" lines, and Cleanup) to derive *everything* -- your hand,
discard, play area, phase, actions, buys, coins, and the opponent's hand
size, draw-pile size, and discard. When that fully succeeds (the common
case), the CLI skips straight to the recommendation with no further
prompts at all. It's still best-effort -- dominion.games occasionally
renders a card as a bare, unnamed "a card", which breaks exact replay from
that point on -- and gracefully falls back to manual entry (pre-filled
with whatever it did derive) when that happens.

```bash
python examples/domibot_relay.py --checkpoint checkpoints/domibot_v2.2.pt --simulations 400
```

## Checkpoint lineage and results

`checkpoints/` (gitignored) holds each promoted checkpoint as
`domibot_vX.Y.pt` (v1-v4 lineage, MCTS self-play) or `domibot2_vN.pt`
(domibot 2, PPO self-play), plus that run's `iter_N.pt` snapshots under
`vX.Y_run/` or `v4_run/`. Full narrative history for anything below is in
git log / prior commit messages; this table is the durable summary.

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
| **domibot2_v1** | **new algorithm: PPO + GAE, no tree search** (see below) | **95-100% vs BigMoney on fixed kingdoms; real multi-action engine turns** |

**The handful of decisions that actually mattered**, in order:
1. **v1→v2**: `_apply_action_continuation_bias` tested at various
   strengths against a converged v1.4 had zero effect -- PUCT's
   exploitation term dominates the prior once a network already has
   confident (anti-chaining) value estimates. Motivated training the
   bias in from iteration 1 on a fresh network instead of retrofitting it.
2. **v2→v3**: sub-decisions (trash/discard/gain/topdeck choices) switched
   from a fixed heuristic to being searched and learned via MCTS like any
   other decision -- a much harder, larger decision space, hence v3.1's
   weak start.
3. **v3.6→v3.7**: `train.py`'s flat learning rate across 1200+ cumulative
   iterations was one real problem, but discussing *why* the whole v3.x
   lineage improved so slowly surfaced a deeper one -- every self-play
   search explored hypothetical continuations against the *one* concrete
   hidden deal that game actually had, a "strategy fusion" problem in
   imperfect-info game AI. `mcts.redeal_hidden_info`/`run_mcts_ensemble`
   (multi-determinization PIMC) fixed it; LR decay followed in v4.1.
4. **v3→v4**: a full-codebase audit found three real engine bugs (game-end
   timing, same-shape decisions from different source cards being
   indistinguishable, popped cards resolving invisibly instead of being
   staged in `set_aside`) that changed `OBS_DIM`, forcing a fresh network.
5. **v4.2 strategy check**: `domibot_v4.2.pt` never played two action
   cards in the same turn -- Big Money + Witch only, zero engine buys, on
   a kingdom stocked with engine pieces. This became the actual question
   for the rest of the project: not "is it winning more," but "does it
   ever discover multi-step strategy."
6. **v4.4 diagnostics**: `action_bias` raised 2.75x (a mechanism already
   built specifically to force self-play to try chaining) produced zero
   change. TD-bootstrapped value targets + an opponent pool -- targeting
   two specific, different mechanisms (noisy credit assignment over a
   whole-game Monte-Carlo return; self-play only ever needing to beat
   itself) -- produced a large strength jump (67%/60%) but *still* zero
   chaining, including from a fresh network with no prior entrenchment
   to blame. Five separate conditions, one conclusion: this needed a
   different algorithm, not another patch.
7. **domibot 2**: see below.

## domibot 2: PPO self-play

Built alongside (not replacing) `training/train.py`'s MCTS lineage, once
five separate conditions (plain self-play, `action_bias` x2.75,
TD-bootstrapped targets, an opponent pool, and TD+pool from a fresh
network) all failed to produce durable engine play -- see the table above
and the approved design plan for the full reasoning. The diagnosis: every
MCTS value target is a Monte-Carlo return (or a short TD-bootstrap of
one), and pure self-play only ever has to beat itself, so a half-built
engine reliably loses to tuned Big Money with nothing rewarding the climb
to a well-executed one. PPO's GAE fixes credit assignment structurally
(dense, bootstrapped credit to *every* decision from a real value
function) and needs no tree search at data-generation time at all.

**Reused unchanged**: the `domibot` engine, `training/encoding.py`,
`network.DomibotNet` (a plain `(obs) -> (policy_logits, value)` residual
MLP -- nothing MCTS-specific, PPO uses the class as-is), `env.DominionEnv`
(already existed, already unused by the MCTS pipeline, exactly the
Gym-shaped interface PPO needs -- gained one small addition, an optional
`reward_fn` so `mcts.terminal_value`'s margin-based reward can replace
plain +1/-1/0), and `evaluate.py`/`agents.py` for eval, so every number
is directly comparable to the MCTS lineage's. `mcts.py` stays too, for a
future inference-time search layer on top of a PPO-trained network (the
network signature never changed) -- not yet wired up.

**New `training/ppo/` subpackage**: `gae.py` (`compute_gae`, reusing the
exact per-decider-subsequence pattern `self_play._backfill_value_targets`
proved out, generalized to full GAE -- a truncated episode's tail
bootstraps from its own value instead of being discarded), `rollout.py`
(`collect_rollouts`, `N` `DominionEnv` instances stepped side by side
sharing one batched forward pass per round, the same root-parallel idea
as `run_mcts_batch` minus the tree -- no `boundary`/`path`/`materialize`
needed anywhere, since PPO only ever advances the one real game), `train.py`
(the PPO loop: clipped surrogate, value MSE, entropy bonus, advantage
normalization; CLI mirrors `train.py`'s conventions). 7 tests in
`tests/test_ppo.py`.

**First real run**: 400 iterations x 50 games (20,000 total games,
matching `domibot_v4.4.pt`'s cumulative lineage volume), default
hyperparameters (`lr=3e-4`, `gae_lambda=0.95`, `clip_eps=0.2`,
`entropy_coef=0.01`). Completed in **under an hour** -- rollout+update
per iteration averaged ~2s, roughly 250x faster than MCTS's ~500-600s/
iteration, since there's no simulation budget to pay for at all. Eval vs
`domibot_v4.4.pt` (via `DomibotAgent`, real MCTS search) climbed from
2/20 to a peak of 17/20, settling in a noisy 9-17/20 band. A 4-way,
100-game round-robin among the best late checkpoints (iter_300/320/380/
400) found `iter_300` and `iter_320` on top by win rate (50.7%/49.7%,
`iter_400` and `iter_380` behind at ~45%) -- but win rate among four
checkpoints from a one-hour run isn't the actual goal here, and it
buried the more important difference: `iter_400` (the final checkpoint)
chains actions far more than `iter_300` does (see below). Promoted
`iter_400` on that basis instead, confirmed with a direct 60-game match
against `domibot_v4.4.pt` (its real MCTS search, 100 sims, vs.
`domibot2_v1.pt`'s raw policy, no search at all): **`domibot2_v1.pt` won
30-28-2** -- a narrower margin than `iter_300` would have given (which
beat `domibot_v4.4.pt` 33-26-1 in the same test), but still a real win,
on the checkpoint that actually shows the behavior this project has been
chasing.

The result that mattered: on the fixed engine-rich kingdom used
throughout this project's diagnostics, `domibot2_v1.pt` (raw policy, no
search) goes 100% single-action on the Witch kingdom (**19-1 vs
BigMoney**; Witch alone judged good enough there, the same call every
strong MCTS checkpoint made) but on the same kingdom with Witch removed,
**20-0 vs BigMoney with 62/218 turns (28%) multi-action, up to 5 plays
deep**, buying Laboratory 26 times and visibly chaining it into further
plays -- something no MCTS checkpoint ever showed after `domibot_v4.2.pt`.
`iter_300` shows the same pattern far more weakly (17/254 and 22/117
turns respectively) -- real, but a fraction of `iter_400`'s, which is
exactly why it's the better pick despite the lower win-rate-among-
siblings: on the actual objective, `iter_400` is doing more of what
matters, more clearly.

**Not yet done**: Stage 2 (privileged critic), Stage 3 (opponent pool
ported to PPO), Stage 4 (inference-time search for the relay tool);
hyperparameter tuning (`entropy_coef` especially -- PPO's exploration
driver, likely to matter a lot for how reliably it finds chaining);
longer runs to see whether strength and chaining both keep improving.

## What's still missing

Card-effect sub-decisions are now searched and learned (see `mcts.py`
above), not just play/buy. What's still missing: `examples/domibot_relay.py`/
`training/relay.py` only recommend sub-decisions for one case so far --
Militia's forced discard (`reconstruct_opponent_turn_boundary`), when a
pasted log ends with the opponent having just played it. Every other
sub-decision (Bureaucrat/Bandit's own forced reactions, and your own
mid-turn choices like an unresolved Chapel trash) still falls back to
manual entry. Also missing: extending `mcts.run_mcts_batch`'s root-parallel
batching to
heterogeneous per-root simulation budgets (sub-decisions now consume
search budget that used to be free, so a kingdom with lots of them needs
more total decision points for the same amount of real game); and
hyperparameter tuning (network size, simulation count, `--parallel-games`,
replay buffer size) and longer training runs than anything validated so
far.
