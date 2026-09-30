# training

The RL side of domibot, kept separate from the `domibot` rules engine. Two
approaches were built, in order. **Phase 1**, AlphaZero-style MCTS
self-play, was abandoned after it never learned to chain Action cards.
**Phase 2**, PPO, produced the current bot, `domibot2.3`. Both use the same
engine, encoding and network.

## What's here

- **`encoding.py`** — the network's inputs and outputs. `ACTION_VOCAB` (206
  actions) and `legal_action_mask`; `encode_observation` (419 values,
  hidden information respected: an opponent shows only its discard pile,
  play area and hand/deck sizes); `encode_public_extras` (138 more public
  values: scores, opponent card ownership, kingdom, empty piles);
  `encode_own_zones` (101 more: the player's own draw pile, discard pile
  and play area).
- **`network.py`** — `DomibotNet`, a residual MLP (256 wide × 4 blocks by
  default) with a policy head (206 logits) and a tanh value head. Size and
  extra inputs are saved in each checkpoint. New inputs join a trained
  network through zero-initialized weights, so its play is unchanged until
  they're trained.
- **`env.py`** — `DominionEnv`, a Gym-style masked-action wrapper around `Game`.
- **`ppo/`** — Phase 2: rollouts, GAE, the training loop, the opponent
  league, distillation into a larger network, search evaluation, and
  training-curve plots.
- **`agents.py` / `evaluate.py`** — the `Agent` protocol (`act(game) ->
  Action`), Big Money baselines and MCTS agents; `play_match` for paired
  head-to-head series, `wilson` for their confidence intervals.
- **`strategy_profile.py` / `strategy_bots.py` / `gauntlet.py` /
  `plan_search.py`** — what a checkpoint actually plays, scripted
  strategies to test it against, and a search for the buy plan that beats
  it on a given board (see "Card use" and "Plan search").
- **`mcts.py`** — PUCT search from Phase 1, now an optional search layer
  on top of a trained network.
- **`self_play.py` / `train.py` / `heuristics.py`** — Phase 1's loop, and a
  fixed fallback for card-effect decisions.
- **`relay.py` / `log_parser.py`** — the move advisor for real games (see
  "The relay tool").

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

## Phase 2: PPO — the current bot

PPO with GAE credits every decision through a learned value function, and
needs no tree search to generate data: rollouts step many games side by
side with one batched forward pass per round.

- Every decision is a training example, card-effect choices included. The
  ~40% with a single legal action train only the value head.
- The reward at game end is 0.8 × win/loss + 0.2 × a tanh of the VP margin
  (`--reward-win-weight`).
- Evals use the raw greedy policy, with no search.

**From scratch** (defaults: learning rate 3e-4, 4 passes per batch, entropy
bonus 0.01, GAE λ 0.95):

```bash
python -m training.ppo.train --iterations 2400 --games-per-iter 50
```

**Continuing a trained network.** A trained policy resumed at a high
learning rate or with small batches first gets worse: its entropy climbs
and its greedy play degrades until the learning rate decays again. Resume
with large batches and a low learning rate, as 2.2 and 2.3 were trained:

```bash
python -m training.ppo.train \
    --checkpoint checkpoints/domibot2/domibot2.3.pt --start-iteration 16001 --run-name my_run \
    --iterations 2000 --games-per-iter 256 --minibatch-size 1024 \
    --lr 5e-5 --lr-final-frac 0.1 --target-kl 0.02 \
    --eval-every 25 --eval-games 200 --eval-rival-checkpoint checkpoints/domibot2/domibot2.3.pt \
    > logs/domibot2/my_run.log 2>&1 &
```

That's about 7 seconds per iteration on one GPU. Checkpoints go to
`checkpoints/domibot2/<run-name>_latest.pt`, plus `<run-name>_iter_N.pt` at
every eval. See `--help` for the rest: league opponents (`--league-*`,
including the gauntlet's bots), exploration (`--explore-*`,
`--buy-floor`), a fresh network's size (`--hidden-dim`, `--num-blocks`),
`--epochs-per-update`, `--no-zone-features`, and an eval against
`domibot_v4.4`. `python -m training.ppo.plot_eval <logs>` plots eval win
rate against iteration (needs `pip install -e ".[plot]"`).

### How domibot2.3 was trained

![Training curve](../docs/training_curve.png)

| iterations | what changed | result |
|---|---|---|
| 1–2400 | from scratch, 50 games per iteration, learning rate 3e-4 | multi-Action turns by iteration 400 |
| 2401–4400 | entropy bonus 0.01 → 0.03, cosine learning-rate decay | |
| 4401–8000 | 30% of games against recent snapshots of itself | `iter_8000` is **`domibot2.1`** |
| 8001–12000 | win-weighted reward, public extras, single-move masking, learning rate 2e-4 | level with 2.1 (52.1%) |
| 12001–14000 | 4× the games per update (256), a quarter of the learning rate (5e-5) | **`domibot2.2`**, 58.3% vs 2.1 |
| 14001–16000 | a quarter of the games against the gauntlet's strategy bots (below) | **`domibot2.3`**, level with 2.2, beats the rush that exploited it |

The two releases against their benchmarks (raw policy, paired random
kingdoms, each kingdom played from both seats):

| opponent | games | `domibot2.3` | `domibot2.2` |
|---|---|---|---|
| the previous release | 2000 | 50.7% vs 2.2 (965–938–97) | 58.3% vs 2.1 (1130–797–73) |
| BigMoney + terminal | 2000 | 79.2% | 76.8% |
| BigMoney | 400 | 95.4% (380–17–3) | 92.6% (367–26–7) |
| `domibot_v4.4`, 100-simulation MCTS | 200 | 88.0% (175–23–2) | 84.8% (169–30–1) |

`domibot_v4.4`'s search sees the true game state, including its opponent's
hand and every deck's order.

### What didn't beat 2.2

These attempts to improve on 2.2 all landed level with it (head-to-head,
2000 games):

| attempt | vs 2.2 |
|---|---|
| 2000 more iterations with the same recipe | 51.5% |
| a 5× larger network (512 × 6), distilled from 2.2's predecessor, then trained with the same recipe | 50.1% |
| an opponent league: past checkpoints, a separately trained network, `domibot_v4.4` and scripted bots, sampled by PFSP | 50.5% |
| the player's own draw pile, discard pile and play area as inputs | 51.4% |
| search on top of 2.2, with hidden information resampled: 100–1600 simulations, up to 16 resamples | 48.1–50.8% |

Two lessons carry over:

- The larger network drifted (entropy up, strength down) until it made one
  pass over each batch instead of four (`--epochs-per-update 1`): it was
  fitting each batch's noisy advantages.
- Search picks the policy's own move on 97–100% of decisions, and even
  search that sees the true game state gains only 2 points (52.0%). The
  value head isn't accurate enough for search to overrule the policy.
  `ppo/search_eval.py` measures this.

### Card use: the current bottleneck

`strategy_profile.py` reports which kingdom cards a checkpoint's raw policy
puts in its deck, buys and plays in self-play:

```bash
python -m training.strategy_profile checkpoints/domibot2/domibot2.3.pt --games 600 --workers 6
```

2.2 has stopped using nine of the 26 kingdom cards. Throne Room, Chapel,
Workshop, Artisan, Remodel, Mine, Moneylender and Vassal are in 1% of its
decks or fewer, Village in 6%. It never played a Throne Room in 400 games
with Throne Room, Village and Smithy in every kingdom. Earlier checkpoints
show when they went (share of decks holding the card at the end):

| checkpoint | Throne Room | Village | Chapel | Sentry |
|---|---|---|---|---|
| iter 1000 | 0% | 6% | 15% | 0% |
| iter 2000 | 27% | 27% | 4% | 0% |
| iter 4000 | 2% | 1% | 2% | 16% |
| iter 8000 (`domibot2.1`) | 0% | 7% | 0% | 61% |
| `domibot2.2` | 0% | 6% | 0% | 87% |

Bought before the policy could play them well, these cards lost to Silver
and Gold, and their buy probability fell toward zero. After that PPO never
saw a deck holding them. Fresh runs of today's code didn't keep them either
(4000 iterations each, same recipe and seed; each reached 25–29% against
2.2):

| fresh run | Throne Room in decks, iterations 1000–4000 | Throne Room played |
|---|---|---|
| with the own-zone inputs | 5–8% | rarely, never after iteration 2000 |
| without them | 2–9% | rarely, never after iteration 2000 |
| with exploration (below) | 0% | never |

**Exploration** first steered one player's early buys toward one or two
random kingdom cards, and kept every affordable kingdom card at a minimum
buy probability (`--buy-floor`). It taught the policy to play a Throne Room
it holds (51% of the time with a target in hand; 2.2: 20%). But a player
handed three early Throne Rooms, on boards with Village and Smithy, won 1
of 60 games against the same policy. Throne Room only pays inside a
coordinated engine, so steering in single cards teaches, correctly, that
they're bad buys in this policy's decks. Steering now uses whole plans (see
"Steering whole plans").

**The gauntlet** plays a checkpoint against scripted strategies built on
cards 2.2 doesn't use, each on kingdoms holding those cards, with BigMoney
+ terminal on the same kingdoms for comparison:

```bash
python -m training.gauntlet checkpoints/domibot2/domibot2.3.pt --kingdoms 400 --workers 6
```

| bot | its cards | bot vs BigMoney + terminal | 2.2 vs bot | 2.2 vs BigMoney + terminal, same kingdoms |
|---|---|---|---|---|
| Workshop/Gardens | Workshop, Gardens | 58.6% | 65.8% | 83.2% |
| Chapel/Witch | Chapel, Witch | 56.1% | 72.8% | 72.9% |
| Throne Room engine | Chapel, Throne Room, Village, Smithy, Market, Militia | 28.6% | 90.3% | 73.0% |

A Workshop/Gardens rush of about twenty lines takes 34% of its games off
2.2, twice what BigMoney + terminal manages on the same kingdoms. The Throne
Room engine loses to BigMoney + terminal too, so it doesn't test much yet.

### Training against the strategy bots: `domibot2.3`

`strategy_league` resumed 2.2 with its recipe and played a quarter of each
iteration's games against the gauntlet's bots and BigMoney + terminal,
each bot on kingdoms holding its cards (the 2.2 command, from
`domibot2.2.pt` at iteration 14001, plus `--league-frac 0.25
--league-scripted workshop_gardens chapel_witch throne_room_engine
bigmoney_terminal`). Its final checkpoint is `domibot2.3`. On the gauntlet,
with kingdoms other than the ones in the table above (`--seed 1`):

| bot | `domibot2.3` | `domibot2.2` |
|---|---|---|
| Workshop/Gardens | 87.3% | 63.6% |
| Chapel/Witch | 75.9% | 74.1% |
| Throne Room engine | 93.6% | 89.9% |

It learned to beat the rush, gave up nothing head-to-head (50.7% vs 2.2)
and gained a few points against every baseline. Its card use barely
changed: it still skips the same nine cards.

### Plan search: is there a better strategy on this board?

The network can't play an engine it's handed. With an engine bot making
its buys on the engine boards, it lost all 300 games to 2.3, worse than
the fully scripted bot (6%). So whether engines would beat 2.3 needs a
different test. `plan_search.py` evolves a buy plan for one board against
a checkpoint, the way Provincial (an evolutionary Dominion AI) evolved
kingdom-specific strategies. A plan is a buy menu: an ordered list of
cards and copies wanted, plus when Provinces, Duchies and Estates take
over. A scripted player with generic rules for all 26 cards plays the
plan. The best plans are re-scored on fresh games, since the search
overrates its own winners.

```bash
python -m training.plan_search checkpoints/domibot2/domibot2.3.pt --boards 20 --home "Throne Room" Village \
    --generations 30 --population 40 --parents 10 --games 16 --final-games 300 --workers 7
```

A board takes about 2 minutes. Against 2.3 (`--seed 1` for the Throne Room
boards, `--seed 2` for the random ones):

| 20 boards | a plan beats 2.3 | best plan's score, median board |
|---|---|---|
| with Throne Room and Village | 1 | 39% |
| random | 1 | 33% |

Both winners are the same deck, on the two boards with Chapel, Village,
Witch and Laboratory: one Chapel, two Witches, two Villages, two
Laboratories, then money, with no Province before turn 12 (70.6% and 55.5%
over 600 fresh games). Everywhere else the best plan is Big Money plus one
or two cards, and loses to 2.3. No Throne Room plan won anywhere. But the
scripted player plays engines poorly, so this doesn't show that a
well-played engine would lose too.

### Steering whole plans

In a share of the self-play games (`--explore-frac`), every player's buys
follow a whole plan for its first 1–16 turns (`--explore-turn-limit`), and
the policy makes every other decision. Half the time it's an engine
(`plan_search.engine_plans`) when the board has a village and a draw card;
otherwise Big Money with one or two copies of one kingdom card, or a
Gardens rush. Both players are steered because a scripted engine wins only
about 11% against 2.3's own buying, which says little about how well it was
played. Against another steered deck, how it's played decides the game.
The plan's buys train no policy. The log's `steered_won` counts how often
an engine beat a non-engine plan (45% for 2.3). `--explore-plans` steers one
player with plan search's winners instead, each on its own board.

Running now: 2.3 resumed with the strategy-bot league plus steering in a
quarter of the games (`--explore-frac 0.35`).

### Not yet done

- **A better value estimate**, e.g. a critic that sees hidden information
  during training. Search needs one before it can help.

## The relay tool

`examples/domibot_relay.py` recommends moves in a real game you play
yourself, e.g. on dominion.games; it clicks nothing. Paste the game log:
`log_parser.py` replays it turn by turn, and `relay.py` rebuilds the game
from what's visible at the table. Your own cards are known exactly; for the
opponent it has the discard pile and the hand and deck sizes, and fills in
the rest at random.

```bash
python examples/domibot_relay.py --checkpoint checkpoints/domibot2/domibot2.3.pt --simulations 400
```

Besides phase decisions, it covers the choices a log can leave pending:

- **your own card, mid-effect** (Chapel, Throne Room, Vassal, the
  Remodel/Mine/Workshop/Artisan gains, Sentry, Harbinger, Poacher,
  Cellar): the card is replayed through the engine with your deck stacked
  from the draws the log shows.
- **the opponent's attack**: Militia's discard, Bureaucrat's topdeck,
  Bandit's trash, and whether to reveal Moat.

Cards known to be on top of a deck, and cards known to be in the
opponent's hand (a revealed Moat), are placed exactly. When the replay
can't follow the log (an unrecognized line, or an unnamed card of yours),
the tool falls back to manual entry, pre-filled with what it did derive.
`tests/test_log_parser_edge_cases.py` replays 13 real games and checks
every point a paste could end at.

Not covered yet: your own Library (unverified against real logs), a
reaction to a Throne-Roomed attack or to one played after the opponent's
Council Room, a card that reshuffles your deck twice, the order two Sentry
cards go back in, and games with more than two players.

## Phase 1: MCTS self-play (AlphaZero-style) — abandoned

`mcts.py`'s PUCT search drove `self_play.py` and `train.py`: the policy
trained on search visit counts, the value on game outcomes, for every
decision including card effects. Card effects are suspended generators
that `Game.clone()` can't copy, so a search node is a `(boundary, path)`
pair that gets replayed on demand.

| checkpoint (`checkpoints/domibot1/`) | what changed | result |
|---|---|---|
| v1.x | original network | ~68% vs BigMoney; never chains Actions |
| v2.x | fresh network, biased toward continuing the Action phase | first multi-card turns; v2.2 beat v2.1 57–36–7 |
| v3.x | search on card-effect decisions, curriculum kingdoms, multiple determinizations | ~35% vs v2.2; avoids chaining |
| v4.1–v4.3 | three engine bugs fixed, fresh network, learning-rate restarts | 35/60 vs BigMoney at best |
| v4.4 | TD-bootstrapped value targets and an opponent pool | 40/60 vs BigMoney, 36/60 vs BigMoney + terminal; Big Money + Witch, never chains |

It never got past Big Money + Witch. Its value targets were whole-game
outcomes, which bury the credit for a buy that pays off turns later under
a game's worth of noise. And self-play only had to beat itself, so a
half-built engine always lost to tuned Big Money. PPO's GAE addresses the
first by construction. It still runs: `python -m training.train` (see
`--help`).
