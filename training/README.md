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

`checkpoints/` (gitignored) holds two separate lineages, each archived as
`domibot_vX.Y.pt` plus a `vX.Y_run/` folder with that run's `iter_N.pt`
snapshots:

- **v1.1 - v1.4**: the original network, resumed across several training
  sessions (`v1.1_run` through `v1.4_run`). BigMoney-relative win rate
  climbed roughly 0% → 46% (v1.1) → 52-55% (v1.2) → 59% (v1.3) → 68%
  (v1.4) over its training history. Play was generally sound (good
  Province timing, sensible attack usage) but self-play games essentially
  never showed genuine multi-card "engine" turns -- confirmed by directly
  testing `_apply_action_continuation_bias` at various strengths against
  the converged v1.4 network and finding zero effect, since PUCT's
  exploitation term dominates the prior once a network already has
  confident (anti-chaining) value estimates. That finding motivated v2.
- **v2.1**: a *fresh* network (not resumed from v1.x) trained with the
  action-continuation bias active from iteration 1 instead of retrofitted
  later, `--simulations 500` (vs. v1.x's 100), and root-parallel self-play
  batching (`mcts.run_mcts_batch`) to make that simulation count
  affordable. One 100-iteration run (`v2.1_run/`), self-play on CPU /
  training+eval on CUDA. Final in-training eval: **35/40 vs BigMoney,
  28/40 vs domibot_v1.4** (iteration 100); a separate 100-game, non-
  alternating-seat head-to-head against v1.4 gave **71-27-2**. Directly
  inspecting its self-play games (not just win rate) found real,
  repeated Village/Laboratory/Merchant-enabled multi-action turns v1.4
  never showed -- e.g. stacking 3-4 Laboratories in one turn to draw deep
  into the deck, or `Village -> Bureaucrat` (a pure enabler spending its
  extra action on a genuine payoff card, not just another cantrip). Not
  perfect: an early checkpoint (iter 30) showed a real pathology --
  trashing its entire deck down to a single Chapel via over-aggressive
  trashing, then stalling turn after turn with nothing to draw or buy --
  gone by iter 90/100, though later checkpoints lean toward avoiding
  Chapel altogether rather than clearly having learned moderate use of it.
- **v2.2**: resumed from v2.1 for 100 more iterations (101-200), same
  settings, self-play on CPU / training+eval on CUDA. Training loss
  plateaued almost immediately (policy_loss oscillating ~0.51-0.54,
  value_loss ~0.04-0.05 from iter 110 onward, no real downward trend),
  and the periodic 40-game in-training evals were too noisy on their own
  to tell whether any given checkpoint was actually better -- e.g. iter
  180 looked like the run's peak (34/40 vs BigMoney, 32/40 vs v1.4) while
  the final iter 200 looked worse (28/40, 25/40). A proper 100-game
  round-robin between v2.1, iter_180, and iter_200 cleared this up: **iter
  200 beat v2.1 57-36-7**, a real margin, while iter_180 landed close to
  even against both v2.1 (49-46-5) and iter_200 (50-46-4) -- the
  in-training numbers were mostly noise, and the final checkpoint (the
  one promoted to `domibot_v2.2.pt`) is genuinely the strongest of the
  three despite its worse-looking final eval line.
- **v3.1**: a *fresh* network, first lineage trained with sub-decision
  search on (see `mcts.py` above) -- every trash/discard/gain/topdeck
  choice searched and learned via MCTS, not resolved by the fixed
  heuristic. Same settings as v2.1/v2.2 otherwise (500 sims, self-play on
  CPU / training+eval on CUDA), `--max-moves 1000` (see `self_play.py`'s
  `DEFAULT_MAX_MOVES` -- the original 400 was calibrated before
  sub-decisions counted against it and made self-play games fail to
  finish naturally almost every time at this run's early checkpoints,
  which was caught and fixed before this run). One 100-iteration run
  (`v3.1_run/`). `policy_loss` declined steadily throughout (0.86 -> 0.71,
  never plateaued), confirming real learning -- unlike v2.2's plateau,
  though not directly comparable since v3.1 is also learning a much
  larger decision space (every sub-decision, not just play/buy) from a
  fresh network in the same 100 iterations. Playing strength is
  correspondingly far behind v2.2 at this stage: eval win rate vs
  BigMoney never exceeded 6/50 (12%) across the whole run, finishing at
  2/50. Expected for iteration 1-100 of a new, harder-to-learn lineage
  starting from scratch (compare v1.1's own 0% start) rather than a sign
  of anything broken -- the fixes verified during this run's own
  diagnosis (the two `heuristics.py` bugs, the move-cap recalibration)
  removed the confounds that would have made a weak result here
  ambiguous. Promoted as `domibot_v3.1.pt`, expecting a `v3.2` (and
  likely more) continuation before this lineage is competitive.
- **v3.2**: resumed from v3.1 for 200 more iterations (101-300), same
  settings (`--max-moves 1000`). `policy_loss` kept declining (0.71 ->
  ~0.65, still no plateau) and eval vs BigMoney climbed sharply from
  v3.1's 2/50 up into a noisy 34-54% range (17-27/50) from iteration 170
  on -- a real jump, confirmed by a clean 100-game (non-noisy) match
  against BigMoney-caliber play. But a 100-game round-robin against
  `domibot_v2.2.pt` (the actual bar that matters) told a very different
  story: **v2.2 beat both iter_260 (the in-training peak) and iter_300
  (the final iteration) about 80% of the time** (80-17-3 and 83-16-1),
  with iter_260 and iter_300 statistically indistinguishable from each
  other (17/100 vs 16/100 against v2.2) -- so v3.2 is genuinely still
  far behind v2.2 despite its encouraging trend against BigMoney.
  Directly inspecting its actual games (not just the score) explains
  why: across a 6-game self-play sample with a kingdom covering every
  sub-decision type, it converged hard on spamming Militia (197 plays,
  next-highest was Smithy at 21) -- the one card here needing *no*
  sub-decision from whoever plays it, since the opponent does the
  discarding -- while essentially avoiding Village/Market/Festival
  (chaining enablers) and Artisan (a two-step gain-then-topdeck
  decision) entirely. The sub-decision machinery itself looks sound
  where it *does* get used: every Workshop-triggered gain went to
  Silver or Village, never Curse, and 17/18 trashes across the sample
  were genuine junk (Estate/Copper), not good cards -- both of today's
  fixed bugs stayed fixed under real play. It just hasn't learned to
  reach for the harder, more valuable lines yet. Promoted as
  `domibot_v3.2.pt` (the final iteration, per the tie-breaker above);
  `domibot_v2.2.pt` remains the strongest checkpoint for actual play.
- **v3.3**: resumed from v3.2 for 200 more iterations (301-500), same
  settings. `policy_loss` stopped declining and settled around ~0.64-0.67
  for most of the run -- v3.2's steady improvement has leveled off. Eval
  vs BigMoney stayed in a healthy, stable 44-68% range throughout. A
  4-way, 80-game round-robin between `domibot_v2.2.pt` and the three
  most promising in-training checkpoints (iter_450, iter_480, iter_500)
  found a clear (not tied, unlike v3.2's) internal winner -- **iter_480
  beat both iter_450 (42-33-5) and iter_500 (42-36-2)** -- but all three
  still lost to v2.2 roughly 80-85% of the time (64-16-0 for iter_480,
  the best showing). That puts v3.3's best candidate at roughly 20%
  against v2.2, only a modest gain over v3.2's ~17% -- a decelerating
  improvement curve after v3.1's initial ~2%, across three successive
  200-iteration continuations (500 total). Consistent with the earlier
  gameplay inspection: without learning to actually chain the enabler
  cards it's been avoiding, more iterations of the same self-play regime
  buys diminishing returns. Promoted `iter_480` (not the final iteration
  this time -- the round-robin gave a clear, non-tied answer) as
  `domibot_v3.3.pt`; `domibot_v2.2.pt` remains the checkpoint to actually
  use. Worth considering before another straight continuation: whether
  something in the setup itself (network capacity, simulation count,
  self-play diversity) needs to change rather than just running more
  iterations of the same regime.
- **v3.4**: resumed from v3.3 for 100 more iterations (501-600), but with
  `--min-sub-decision-cards 6` -- a curriculum change, not just more of
  the same regime, aimed directly at a miscalibration found by inspecting
  MCTS search internals: with judgment-heavy cards (Chapel especially)
  sparse across random kingdoms, the *visit-count* distribution that
  becomes each iteration's training target had drifted to favor
  `TRASH(Gold)` above every other Chapel option -- worse the more such
  cards co-occurred in one kingdom, a joint event too rare under plain
  random sampling to get corrected by ordinary training volume.
  `min_sub_decision_cards` forces at least that many sub-decision cards
  into every sampled kingdom, directly inflating how often the
  self-play buffer actually contains examples of this failure mode being
  searched. `policy_loss` held flat around ~0.65-0.68 (no regression from
  the forced kingdom shift); in-training eval vs `domibot_v2.2.pt` ranged
  6-13/50 across the run, peaking at iter_560. A 4-way, 80-game
  round-robin between `domibot_v2.2.pt` and the three highest-scoring
  in-training checkpoints (iter_550, iter_560, iter_590) found no clean
  internal winner this time -- a genuine three-way cycle (iter_560 beat
  iter_550 42-34, iter_550 beat iter_590 50-30, iter_590 beat iter_560
  47-31), unlike v3.3's clear iter_480. But against the fixed benchmark
  that actually matters, **iter_560 had the best showing of the three:
  22/80 (27.5%) against v2.2**, versus iter_590's 20/80 (25.0%) and
  iter_550's 16/80 (20.0%) -- a real, if modest, improvement over v3.3's
  best (20%) and v3.2's (~17%), consistent with (not proof of) the
  curriculum change addressing the miscalibration it targeted. Promoted
  `iter_560` as `domibot_v3.4.pt`, picked on that v2.2 showing rather than
  the ambiguous internal cycle, since performance against the fixed
  benchmark is what every prior promotion in this lineage has actually
  been decided on. `domibot_v2.2.pt` remains the checkpoint to actually
  use. Worth another continuation at this same curriculum setting before
  concluding much either way -- 100 iterations is a short run to separate
  a real effect from noise (the internal round-robin's own cycle is a
  reminder of how much variance 80 games still carries).
- **v3.5**: resumed from v3.4 for 300 more iterations (601-900), same
  `--min-sub-decision-cards 6` curriculum setting. First launch of this
  run silently defaulted to `--simulations 100` instead of `500` (the
  setting every checkpoint since v2.1 has actually used) -- a copy-paste
  omission, caught by comparing self-play throughput against v3.4's log
  (buffer growth per wall-clock second was ~5x faster, matching a 500->100
  simulations drop almost exactly) before any checkpoint from that bad
  run was promoted or even seriously looked at. Killed after 20 iterations
  and relaunched correctly from `domibot_v3.4.pt`; nothing from the bad
  run persisted (only the disposable `latest.pt`/`iter_610.pt`/`iter_620.pt`
  scratch files, since `domibot_v3.4.pt` itself is never overwritten by
  `train.py`). The corrected run's `policy_loss` held flat around
  ~0.62-0.65, similar to v3.4; in-training eval vs `domibot_v2.2.pt`
  trended upward in the back half, peaking at iter_870 (19/50), iter_810
  (18/50), and iter_830 (17/50). A 4-way, 80-game round-robin between
  those three and `domibot_v2.2.pt` found a clean internal winner this
  time (unlike v3.4's cycle): **iter_830 beat both iter_810 (44-35-1) and
  iter_870 (43-35-2)** -- and also had the best showing against v2.2:
  **25/80 (31.25%)**, versus iter_870's 23/80 (28.75%) and iter_810's
  15/80 (18.75%). That's a real continuation of the upward trend across
  the curriculum-sampling lineage (20% -> 27.5% -> 31.25%), on top of a
  3x longer run than v3.4's. Promoted `iter_830` as `domibot_v3.5.pt`;
  `domibot_v2.2.pt` remains the checkpoint to actually use, though the
  gap is visibly narrowing.
- **v3.6**: resumed from v3.5 for 300 more iterations (901-1200), same
  curriculum setting -- no round-robin, not promoted. `policy_loss`
  climbed steadily for the entire back half of the run (~0.622 at iter 920
  to ~0.658 at iter 1200), not noise settling, a real regression rather
  than a plateau; in-training eval vs `domibot_v2.2.pt` stayed flat on
  average (~26.9%, no better than v3.5's ~25.1%) despite the extra
  iterations. Most likely cause: `train.py` uses a flat `lr=1e-3` with no
  decay schedule across what's now 1200+ cumulative iterations. This
  diagnostic finding is what motivated the bigger architectural question
  addressed next, rather than just adding LR decay and continuing --
  discussing *why* the whole v3.x lineage improves so slowly against v2.2
  surfaced a deeper mismatch: every self-play MCTS search explores
  hypothetical continuations against the one concrete hidden deal that
  game actually has (see `mcts.py`'s new module content below), a known
  "strategy fusion" problem in imperfect-info game AI.
- **v3.7**: resumed from v3.5 (not v3.6 -- not worth building on a run
  that regressed) for 300 more iterations (1201-1500), with the new
  `--determinization-ensemble-size 4` (`mcts.redeal_hidden_info`/
  `run_mcts_ensemble`): instead of every self-play tree searching the one
  true hidden deal, each phase-action decision now searches 4 independently
  redealt hidden-info samples and merges their visit counts -- the
  standard multi-determinization form of PIMC. Same curriculum setting and
  `--simulations 500` as before; `policy_loss` still climbed similarly to
  v3.6 (~0.606 to ~0.651), confirming the LR-decay issue is real and
  independent of this fix -- not addressed here. Despite that, in-training
  eval vs `domibot_v2.2.pt` averaged noticeably higher than any prior run
  (~30.7% vs v3.6's ~26.9%) and hit a new peak (22/50 = 44% at iter 1400).
  A 4-way, 80-game round-robin between `domibot_v2.2.pt` and the three
  best in-training checkpoints (iter_1400, iter_1470, iter_1480) found a
  clean winner: **iter_1480 beat both iter_1400 (44-33-3) and iter_1470
  (40-39-1)**, and had the best showing against v2.2: **28/80 (35.0%)** --
  the best result anywhere in the v3.x lineage (20% -> 27.5% -> 31.25% ->
  **35.0%**), on a run with a known-unaddressed optimization problem still
  present. Promoted `iter_1480` as `domibot_v3.7.pt`; `domibot_v2.2.pt`
  remains the stronger checkpoint for actual play, but the gap keeps
  narrowing, now with a plausible causal story (not just more iterations)
  for the last jump. Worth doing next: add the LR decay schedule and run
  another ensemble continuation -- the two fixes address independent
  problems and neither has been tried with the other in place yet.
- **v4.1**: a *fresh* network from scratch, not a v3.x continuation --
  forced, not discretionary, because this pass fixed the observation
  encoding itself (`OBS_DIM` 350 -> 419: source-card one-hot + set-aside
  card counts, see `encoding.py`), which changes the input layer's shape
  and makes every prior checkpoint structurally incompatible. Bundles
  everything found in a full-codebase audit: three engine correctness
  bugs (game-end now evaluated at Cleanup instead of mid-turn the instant
  a pile empties; `Decision.source_card` + a `_resolving` card stack so
  same-shape decisions from different source cards are distinguishable;
  Vassal/Bandit/Library/Sentry now stage popped cards in `set_aside`
  instead of resolving them invisibly), a normalized network (pre-norm
  residual blocks, `LayerNorm` on the input and before the heads),
  and a rebalanced training signal (`MARGIN_SCALE` 20 -> 10,
  `value_loss_weight=2.0`, `grad_clip=1.0`, cosine LR decay 1e-3 -> 1e-4
  across the run instead of v3.6/v3.7's flat rate, and `value_known`
  masking so a truncated game's fabricated value target doesn't corrupt
  the loss). Per the user's call, the curriculum
  (`--min-sub-decision-cards`) is dropped entirely for this lineage --
  every sub-decision is searchable from iteration 1, uniform kingdoms
  throughout. `--determinization-ensemble-size 2` throughout (v3.7's
  fix, carried forward).

  First launch used `--games-per-iter 20 --train-steps-per-iter 24`,
  chosen to target v2.2's replay ratio via two independent levers without
  checking that cutting gradient steps adds no independent-label
  diversity -- the same root cause (over-replay of a thin, correlated
  data pool) this run was meant to fix. Caught before it went far
  (~20 iterations in): measured buffer staleness was ~49 iterations vs
  v2.2's proven ~19, and only 20 independent games/iter vs v2.2's ~100.
  Restarted clean at **`--games-per-iter 50 --train-steps-per-iter 40`**
  (~15,000 fresh examples/iter, replay ratio <1, buffer capacity 200k
  reached by iteration 15) -- matching or beating v2.2 on every replay
  axis except independent games/iter (50 vs ~100), left as a lever for a
  future continuation if the value head stalls again.

  200 iterations, self-play on CPU / training+eval on CUDA, ~500s/iter.
  In-training eval vs BigMoney (60 games/check) climbed from 0/60
  (iterations 10-40) to a noisy but real back-half cluster: 7, 6, 2, 1,
  6, 6, 4, 5, **9**, 7, 4, **11**, **9** (iterations 80-200) -- the four
  best scores of the whole run land in iterations 160-200. `value_loss`
  dropped cleanly from 0.30 to a minimum of 0.08 by iteration 34, then
  climbed to ~0.17 and *plateaued* there (oscillating 0.15-0.18) for the
  entire back half rather than continuing to diverge -- read as benign
  (self-play games getting more contested as the policy improves raises
  the intrinsic difficulty of value prediction) rather than a training
  problem, since `policy_loss` and the BigMoney eval kept improving
  concurrently rather than degrading. A 4-way, 100-game round-robin
  between the four best late checkpoints (iter_130, iter_160, iter_190,
  iter_200) found a clean winner with no internal contradictions:
  **iter_200 beat all three others in direct play** (55-43-2 vs iter_130,
  55-39-6 vs iter_160, 50-42-8 vs iter_190), taking the standings at
  53.3% combined win rate vs iter_190's 51.0%, iter_130's 44.0%, and
  iter_160's 41.0% -- the final checkpoint, after full LR decay, is
  genuinely the strongest, unlike v2.2's iter_180 in-training-eval red
  herring. Promoted `iter_200` as `domibot_v4.1.pt`.

  Not yet directly comparable to `domibot_v2.2.pt` in a round-robin --
  the old checkpoint's 350-dim input is incompatible with the fixed
  419-dim encoder (this is what `train.py`'s fail-fast guard on
  `--reference-checkpoint` now catches instead of crashing). In absolute
  terms 11/60 vs BigMoney is still far below v2.1's fresh-network
  baseline (35/40), so this checkpoint is likely still behind v2.2 in
  real play -- the point of this run was fixing the underlying regime
  (game-end bug, sub-decision representability, over-replay, missing LR
  decay, imperfect-info search) that every prior lineage trained through,
  not beating v2.2 in one 200-iteration shot. Worth doing next: build a
  comparison shim so v2.2 can play against the new encoding, to get a
  real read on how much of the gap these fixes closed.
- **v4.2**: resumed from `domibot_v4.1.pt` for 200 more iterations
  (201-400), same settings, but with a **fresh warm-restart LR cycle**
  (`--lr 1e-3 --lr-final-frac 0.1` again instead of continuing flat at
  v4.1's terminal 1e-4) rather than a literal continuation -- v4.1's
  `value_loss` had plateaued for its entire back half, and `train.py`'s
  cosine scheduler restarts its `T_max` window from whatever `--iterations`
  is passed on each launch, so this is a deliberate SGDR-style warm
  restart, not an oversight. It worked: in-training eval vs BigMoney rose
  from v4.1's best of 11/60 to a run peak of **28/60 at iteration 370**,
  with the back half (iterations 340-400: 22, 16, 19, 28, 17, 22, 22)
  clearly and durably above the first half's typical 10-16 band, not just
  a single spike. `value_loss` again plateaued around 0.12 for most of
  the run before drifting back up to 0.148 by iteration 400 -- the same
  late-run pattern as v4.1, and read the same way (harder, more
  contested self-play games as the policy improves, not divergence),
  since win rate held up at the same time rather than degrading.

  A 4-way, 100-game round-robin between the four best late checkpoints
  (iter_340, iter_370, iter_390, iter_400) again found a clean winner,
  and again contradicted the in-training numbers: iter_370 had the
  *highest* single eval score (28/60) but placed **third** in the
  round-robin, while **iter_400 beat all three others in direct play**
  (68-30 vs iter_340, 54-42 vs iter_370, 47-46-7 vs iter_390) for 56.3%
  combined -- vs iter_390's 48.3%, iter_370's 44.0%, iter_340's 42.3%.
  Same lesson as v2.2's iter_180 and v4.1's own round-robin: never
  promote off the in-training number alone. Promoted `iter_400` as
  `domibot_v4.2.pt`.

  A qualitative look at `domibot_v4.2.pt`'s actual play (20 games vs
  BigMoney on a fixed kingdom stocked with engine pieces, trashing, an
  attack, and defense -- Village/Laboratory/Market/Festival/Smithy/
  Council Room/Chapel/Witch/Militia/Moat) found a genuinely strong but
  narrow strategy: Big Money + Witch, and *nothing* else. Zero buys of
  any of the six engine pieces across all 20 games, and the
  action-cards-played-per-turn histogram has exactly one bucket -- it
  has never once played two action cards in the same turn. Chapel was
  bought 8 times but played only 3, trashing just 1-2 cards each time
  rather than the aggressive early-game trash-down that makes it strong.
  Likely explanation: multi-step engine payoffs are a long-horizon
  strategy that's hard for MCTS to stumble into via self-play unless the
  search is regularly deep/wide enough to see the payoff several turns
  out, whereas Big Money + Witch is a short, locally-greedy strategy
  that's easy to reinforce. Worth checking again on a later checkpoint --
  if it never starts chaining actions, that's a stronger signal of a
  real ceiling than the loss curves alone.
- **v4.3**: resumed from `domibot_v4.2.pt` for 200 more iterations
  (401-600), same settings and another warm-restart LR cycle
  (`--lr 1e-3 --lr-final-frac 0.1`). The usual post-restart dip appeared
  (eval fell from 28/60 to a low of 21/60 by iteration 440) and fully
  recovered, then went on to set a new run peak of **35/60 at iteration
  490** -- clearing v4.2's best (28/60) by a real margin -- and settled
  into a 27-33/60 band for the back half, clearly above v4.2's 22-28
  band. Both loss curves improved alongside the eval number rather than
  just cycling in place: `value_loss` bottomed at 0.086 (iteration 509,
  the best of the whole v4.x lineage so far) before its usual late-cycle
  climb back up to 0.135 by iteration 600 -- still below v4.1/v4.2's
  ~0.15-0.18 plateau, i.e. the ceiling itself is dropping across legs,
  not just cycling. `policy_loss` similarly settled a touch lower
  (~0.57-0.59 vs the earlier ~0.60-0.65 floor).

  A 4-way, 100-game round-robin between the four best late checkpoints
  (iter_490, iter_530, iter_590, iter_600) was, for the first time, a
  genuine toss-up -- all four landed within 47.7%-50.0%, essentially
  indistinguishable at this sample size (contrast v4.1's and v4.2's
  round-robins, where one checkpoint had a clear double-digit-point
  lead). **iter_590 narrowly took the standings at 50.0%** with two
  direct wins (56-43 vs iter_600, 51-46 vs iter_530) and a near-tie loss
  to iter_490 (48-49). Promoted `iter_590` as `domibot_v4.3.pt` on that
  basis, though given how close the standings were, this pick is weaker
  evidence than the last two promotions -- any of the four would have
  been a defensible choice.

- **v4.4 (diagnostics, then `domibot_v4.4.pt`)**: v4.4 proper (601-800)
  was paused at iteration ~645 after a strategy check found the network
  *never plays two action cards in one turn* at `iter_640` (Big Money +
  Witch only, no engine buys). Two cheap diagnostics followed, each
  ~40 iterations resumed from `iter_640` under out-of-band iteration
  numbers (so they can't collide with the real lineage's files):
  1. `--action-bias 0.55` (up from 0.2): no change in behavior.
  2. `--td-lambda 0.5 --opponent-pool-size 5 --opponent-pool-frac 0.3`
     (new this pass, all off by default): `self_play._backfill_value_targets`
     blends the true outcome with the network's own search-improved value
     the next time the same decider acts, so a deferred-payoff card isn't
     judged only by a game-final outcome dozens of turns later (and
     truncated games stop being wasted); `play_cross_play_games` plays
     30% of games against a frozen recent checkpoint, whose decisions
     produce no examples. Multi-action turns still did not appear, but
     playing strength jumped.

  **Measured on random kingdoms (60 paired games, 100 sims):**
  v4.1 20% vs BigMoney / 7% vs BigMoney+terminal; v4.3 48% / 38%;
  **`iter_91040` 67% / 60%**. Promoted `iter_91040` as `domibot_v4.4.pt`.
  `BigMoneyTerminalAgent` (Big Money plus the best terminal Action the
  kingdom offers; needs no particular card) was added as the harder
  yardstick, with `--eval-bm-terminal` to log it during training.

  **Caveats, in the order they were discovered:** (a) my earlier claim
  that v4.1 "never showed engine play" was wrong -- v4.1 (iter 200) does
  chain actions (2-5 plays in 22 turns; Market/Village/Council Room buys)
  but loses 1-19 to BigMoney doing so, and later checkpoints abandon it,
  which supports "engines were found, then trained out because a
  half-built engine loses to Big Money" over "engines can't be found".
  (b) The fixed test kingdom contained Witch, one of the strongest Big
  Money enablers, so abandoning engines there may be correct play. On a
  Witch-free engine kingdom `iter_91040` goes 2-18 vs BigMoney (no action
  buys, ~107 Duchies vs 50 Provinces -- it greens too early) and v4.1
  0-20. (c) The 90% vs BigMoney seen on the Witch kingdom is not a
  general strength number. The reduced `value_loss` (~0.02-0.03) under
  TD targets is not comparable to earlier legs' (partly self-referential
  targets). A fresh-network run with the same TD + pool settings
  (`--start-iteration 92001`) is in progress to test whether the
  resumed network's entrenchment was hiding an engine effect.

## domibot 2: PPO self-play

The v4.4 entrenchment question was closed with a from-scratch,
200-iteration run under identical TD+pool settings
(`--start-iteration 92001`): still zero multi-action turns on either the
Witch or Witch-free fixed kingdom, and its random-kingdom strength (35%
vs BigMoney / 25% vs BigMoney+terminal) landed *below* plain-MC v4.3 (48%
/ 38%) despite the better recipe -- TD-bootstrapping needs a reasonably
competent value function to bootstrap *against*; it isn't a good way to
train one from nothing. `domibot_v4.4.pt` remains the strongest
checkpoint, confirmed by direct head-to-head against the rest of the
lineage: beats v4.3 34-23-3, v4.2 42-18-0.

That's five separate conditions now (plain MCTS self-play, `action_bias`
raised 2.75x, TD-bootstrapped value targets, an opponent pool, and TD+pool
from scratch) without ever producing durable engine play. The diagnosis:
every value target in the MCTS pipeline is a Monte-Carlo return (or a
short TD-bootstrap of one), and pure self-play only ever has to beat
itself -- a half-built engine reliably loses to tuned Big Money, so
nothing rewards crossing that valley to reach a well-executed one. Rather
than keep patching the MCTS lineage, **domibot 2 is a PPO-based training
algorithm**, built alongside (not replacing) `training/train.py` -- see
the approved design plan for the full reasoning and staging. GAE fixes
the credit-assignment problem structurally (dense, bootstrapped credit to
every decision from a real value function, not just a 1-turn TD hop's
reach) and needs no tree search at data-generation time at all.

**What's reused unchanged**: the `domibot` engine, `training/encoding.py`
(same `OBS_DIM`/`NUM_ACTIONS` vocab), `training/network.py`'s
`DomibotNet` (a plain `(obs) -> (policy_logits, value)` residual MLP,
nothing MCTS-specific about it -- PPO reuses the class as-is),
`training/env.py`'s `DominionEnv` (already existed, already unused by the
MCTS pipeline, and is exactly the Gym-shaped interface PPO needs -- it
gained one small, backward-compatible addition, an optional `reward_fn`
so `mcts.terminal_value`'s margin-based reward can be used instead of
plain +1/-1/0), and `training/evaluate.py`/`training/agents.py` for
eval, so every number is directly comparable to the MCTS lineage's.
`training/mcts.py` stays too, repurposed as an *inference-only* search
layer for the relay tool once a PPO checkpoint is strong enough (Stage
4) -- since the network signature never changed, `DomibotAgent`/
`run_mcts` can point at a PPO-trained network with zero code changes.

**Stage 1 (core PPO loop) is implemented**, in a new `training/ppo/`
subpackage:
- `gae.py` -- `compute_gae` reuses the exact "extract this decider's own
  ordered subsequence from the interleaved trajectory" pattern
  `self_play._backfill_value_targets` proved out for TD-bootstrapping,
  generalized to full GAE. A truncated episode's last transition
  bootstraps from its own value estimate rather than being discarded, the
  same "recover signal from a capped game" idea `td_lambda` introduced.
- `rollout.py` -- `collect_rollouts` steps `N` `DominionEnv` instances
  side by side, one batched network forward pass per round (the same
  root-parallel idea as `run_mcts_batch`, minus the tree -- no
  `boundary`/`path`/`materialize` needed anywhere, since PPO only ever
  advances the one real game). Every decision, phase action and
  sub-decision alike, produces one `Transition`.
- `train.py` -- the PPO loop (clipped surrogate, value MSE, entropy
  bonus, advantage normalization), CLI mirroring `training/train.py`'s
  conventions, eval against `BigMoneyAgent`/`BigMoneyTerminalAgent` and
  optionally a reference checkpoint (e.g. `domibot_v4.4.pt`) via the
  existing `DomibotAgent`. Checkpoints land in `checkpoints/` as
  `ppo_latest.pt` / `ppo_iter_N.pt`.

7 new tests in `tests/test_ppo.py` (full suite: 158 passed). Verified
with a real, if tiny, end-to-end smoke run (`--iterations 3
--games-per-iter 8 --max-moves 40`) -- rollout, GAE, PPO update, eval,
and checkpointing all completed without error. **Not yet done**: a real
training run at a comparable compute budget to `domibot_v4.4.pt`, the
fixed-kingdom multi-action-turn check, and the random-kingdom baseline
comparison that would actually answer whether PPO solves the engine-
building problem -- plus Stages 2-4 (privileged critic, opponent pool,
inference-time search), each gated on Stage 1 producing a real result
first.

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
