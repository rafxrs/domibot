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

## What's still missing

Card-effect sub-decisions are now searched and learned (see `mcts.py`
above), not just play/buy. What's still missing: extending
`examples/domibot_relay.py`/`training/relay.py` to also recommend
sub-decisions (`reconstruct_game` only ever produces boundary states
today); extending `mcts.run_mcts_batch`'s root-parallel batching to
heterogeneous per-root simulation budgets (sub-decisions now consume
search budget that used to be free, so a kingdom with lots of them needs
more total decision points for the same amount of real game); and
hyperparameter tuning (network size, simulation count, `--parallel-games`,
replay buffer size) and longer training runs than anything validated so
far.
