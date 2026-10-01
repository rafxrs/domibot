# training

The RL side of domibot, kept separate from the `domibot` rules engine.
**Phase 1**, AlphaZero-style MCTS self-play, was abandoned after it never
learned to chain Action cards. **Phase 2**, PPO, produced the current bot,
`domibot2.4`. Both use the same engine, encoding and network.

## What's here

- **`encoding.py`**: the network's inputs and outputs. `ACTION_VOCAB` (206
  actions); `encode_observation` (419 values, hidden information
  respected), then public extras (138: scores, opponent card ownership,
  kingdom, empty piles) and the player's own zones (101: draw pile,
  discard pile, play area).
- **`network.py`**: `DomibotNet`, a residual MLP (256 × 4 by default) with
  policy and value heads. New inputs join a trained network through
  zero-initialized weights.
- **`env.py`**: `DominionEnv`, a Gym-style wrapper around `Game`, and the
  reward functions.
- **`ppo/`**: Phase 2: rollouts, GAE, the training loop, the opponent
  league, steering (`explore.py`), search evaluation, plots.
- **`agents.py` / `evaluate.py`**: Big Money baselines and MCTS agents,
  `play_match` for paired series, `wilson` for confidence intervals.
- **`strategy_profile.py` / `strategy_bots.py` / `gauntlet.py` /
  `plan_search.py`**: what a checkpoint plays, scripted strategies to test
  it against, and a search for the buy plan that beats it on a board.
- **`mcts.py`**: PUCT search over a trained network, for the examples, the
  relay and evals; **`heuristics.py`**: a fixed rule for card-effect choices.
- **`relay.py` / `log_parser.py`**: the move advisor for real games.

## GPU

`get_device()` uses CUDA once torch is installed against a CUDA index
(plain `pip install torch` is CPU-only):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); print((x @ x).sum().item())"
```

## Phase 2: PPO

Rollouts step many games side by side with one batched forward pass, and
GAE credits every decision, card-effect choices included. Moves with a
single legal action train only the value head. The reward at game end is
0.8 × win/loss + 0.2 × a tanh of the VP margin. Evals use the raw greedy
policy, with no search.

```bash
# from scratch (learning rate 3e-4, 4 passes per batch, entropy bonus 0.01, GAE λ 0.95)
python -m training.ppo.train --iterations 2400 --games-per-iter 50

# continuing a trained network: large batches and a low learning rate, as 2.2–2.4 were trained
python -m training.ppo.train --checkpoint checkpoints/domibot2/domibot2.4.pt --start-iteration 20001 \
    --run-name my_run --iterations 2000 --games-per-iter 256 --minibatch-size 1024 \
    --lr 5e-5 --lr-final-frac 0.1 --target-kl 0.02 \
    --eval-every 25 --eval-games 200 --eval-rival-checkpoint checkpoints/domibot2/domibot2.4.pt
```

A resumed policy at a high learning rate or with small batches gets worse
before the learning rate decays. About 7 seconds per iteration on one GPU.
See `--help` for league opponents (`--league-*`), steering (`--explore-*`,
`--imitate-steered`) and the rest; `python -m training.ppo.plot_eval <logs>`
plots eval win rates.

### How domibot2.4 was trained

![Training curve](../docs/training_curve.png)

| iterations | what changed | result |
|---|---|---|
| 1–2400 | from scratch, 50 games per iteration | multi-Action turns by iteration 400 |
| 2401–4400 | entropy bonus 0.03, cosine learning-rate decay | |
| 4401–8000 | 30% of games against recent snapshots of itself (now `--league-snapshot-every`) | **`domibot2.1`** |
| 8001–12000 | win-weighted reward, public extras, single-move masking | level with 2.1 |
| 12001–14000 | 256 games per update, learning rate 5e-5 | **`domibot2.2`**, 58.3% vs 2.1 |
| 14001–16000 | a quarter of the games against the gauntlet's bots | **`domibot2.3`**, level with 2.2, beats the Gardens rush |
| 16001–18000 | steering: whole buy plans in a quarter of the games | plays engines better, level with 2.3 |
| 18001–20000 | plan search's winning plans, with self-imitation | **`domibot2.4`**, 55.6% vs 2.3, opens Chapel |

Raw policy, paired random kingdoms, each played from both seats:

| opponent | games | `domibot2.4` | `domibot2.3` |
|---|---|---|---|
| the previous release | 2000 | 55.6% (1077–851–72) | 50.7% (965–938–97) |
| BigMoney + terminal | 2000 | 80.2% | 79.2% |
| BigMoney | 400 | 95.4% | 95.4% |
| `domibot_v4.4`, 100-simulation MCTS on the true game state | 200 | 90.8% | 88.0% |

### What didn't work

| attempt | result |
|---|---|
| 2000 more iterations of 2.2's recipe | 51.5% vs 2.2 |
| a 5× larger network (512 × 6), distilled, same recipe | 50.1% vs 2.2; drifted until one pass per batch (`--epochs-per-update 1`) |
| a league of past checkpoints, `domibot_v4.4` and scripted bots (PFSP) | 50.5% vs 2.2 |
| the player's own draw pile, discard pile and play area as inputs | 51.4% vs 2.2 |
| search on top of 2.2: 100–1600 simulations, hidden information resampled | 48.1–50.8% vs 2.2; search keeps the policy's move 97–100% of the time, the value head isn't accurate enough to overrule it |
| fresh runs of today's code, with or without the own-zone inputs | Throne Room dropped by iteration 2000 |
| steering one or two random cards into early buys | taught Throne Room play, not buying: three early Throne Rooms won 1 of 60 games |
| a buy floor: every affordable kingdom card at 1% or more | collapsed a trained policy within 25 iterations (20% vs 2.3) |
| steering with plan search's winners, without self-imitation | the plans kept winning 71% for 500 iterations |

### Card use

`strategy_profile.py` reports which kingdom cards a checkpoint puts in its
deck, buys and plays in self-play:

```bash
python -m training.strategy_profile checkpoints/domibot2/domibot2.4.pt --games 600 --workers 6
```

Share of decks holding the card at game end:

| checkpoint | Throne Room | Village | Chapel | Sentry |
|---|---|---|---|---|
| iter 1000 | 0% | 6% | 15% | 0% |
| iter 2000 | 27% | 27% | 4% | 0% |
| `domibot2.1` | 0% | 7% | 0% | 61% |
| `domibot2.2` | 0% | 6% | 0% | 87% |
| `domibot2.3` | 1% | 1% | 3% | 86% |
| `domibot2.4` | 0% | 25% | 53% | 89% |

Bought before the policy could play them well, these cards lost to Silver
and Gold, their buy probability fell toward zero, and on-policy PPO never
saw a deck holding them again. 2.4 got Chapel and Village back. It still
never buys Throne Room, nor Workshop, Artisan, Remodel, Mine, Moneylender
or Vassal.

Over 5000 self-play games, 2.4 plays 1.73 Actions per turn (none in 36%
of turns, five or more in 12%). Its buys are 41% Victory cards (Province
18%, Duchy 11%, Estate 8%, Gardens 4%), 27% Treasures (Silver 13%, Gold 10%,
Copper 4%) and 32% Actions, Sentry the most bought at 5%.

### Finding what it's missing

**The gauntlet** plays a checkpoint against scripted strategies, each on
kingdoms holding its cards (`--seed 1`):

```bash
python -m training.gauntlet checkpoints/domibot2/domibot2.4.pt --kingdoms 400 --workers 6 --seed 1
```

| bot | its cards | bot vs BigMoney + terminal | 2.2 | 2.3 | 2.4 |
|---|---|---|---|---|---|
| Workshop/Gardens | Workshop, Gardens | 58.6% | 63.6% | 87.3% | 92.2% |
| Chapel/Witch | Chapel, Witch | 56.1% | 74.1% | 75.9% | 89.1% |
| Throne Room engine | Chapel, Throne Room, Village, Smithy, Market, Militia | 28.6% | 89.9% | 93.6% | 98.2% |

**Plan search** evolves a buy plan for one board against a checkpoint, the
way Provincial (an evolutionary Dominion AI) did. A plan is a buy menu
(cards and copies wanted, in priority order) plus when Provinces, Duchies
and Estates take over. A scripted player plays its cards, or with
`--network-plays` the checkpoint's own policy does, so the search finds
where the network's buying is wrong given its own play. The best plans are
re-scored on fresh games. About 2–3 minutes per board:

```bash
python -m training.plan_search checkpoints/domibot2/domibot2.4.pt --boards 20 --home "Throne Room" Village \
    --generations 30 --population 40 --parents 10 --games 16 --final-games 300 --workers 7 [--network-plays]
```

On the same 40 boards (20 with Throne Room and Village, `--seed 1`; 20
random, `--seed 2`):

| against | the plan's cards played by | boards where a plan wins | best plan, median board |
|---|---|---|---|
| `domibot2.3` | the script | 2 | 33% |
| 2.3 after steering (iteration 18000) | the network | 3 | 38% |
| `domibot2.4` | the script | 0 | 26% |
| `domibot2.4` | the network | 0 | 30% |

Every plan that won before 2.4 opened Chapel, on boards with Chapel and
Laboratory. Against 2.4 no plan wins on any of the 40 boards; the best
scores 45%. No Throne Room plan won anywhere.

### Steering and self-imitation

On-policy PPO can't learn a buy it never makes. **Steering**
(`ppo/explore.py`) makes the buys for it in a share of the self-play
games, while the policy plays the cards. With `--explore-frac`, every
player's buys follow a plan for its first 1–16 turns: an engine half the
time when the board has a village and a draw card, otherwise Big Money
plus one kingdom card or a Gardens rush. Both players are steered because
a scripted engine wins only 11% against the policy's own buying, which
teaches little about playing it. With `--explore-plans`, one player
follows plan search's winners on their own boards instead.
**Self-imitation** (`--imitate-steered`, after Oh et al. 2018) then trains
the policy toward a steered buy wherever it did better than the value head
expected.

2.4 came from two runs on 2.3's recipe and league: steering in a quarter
of the games (`--explore-frac 0.35`), then the three winning plans in 15%
of them with `--imitate-steered 0.1`. In the second run the plans' win rate
against it fell from 71% to 28%. It now opens Chapel in every game on
those three boards and in half the games on other boards holding Chapel,
where it scores 61% against 2.3 (67% with Witch too).

### Not yet done

- **Throne Room**: no plan using it has beaten a checkpoint yet, so there
  is nothing to imitate.
- **A better value estimate**, e.g. a critic that sees hidden information
  during training. Search needs one before it can help.

## The relay tool

`examples/domibot_relay.py` recommends moves in a real game you play
yourself, e.g. on dominion.games; it clicks nothing. Paste the game log:
`log_parser.py` replays it, and `relay.py` rebuilds the game from what's
visible at the table, filling in the opponent's hidden cards at random.

```bash
python examples/domibot_relay.py --checkpoint checkpoints/domibot2/domibot2.4.pt --simulations 400
```

Besides what to play and buy, it answers choices a log leaves pending:
your own card mid-effect (Chapel, Throne Room, Vassal, gains, Sentry,
Harbinger, Poacher, Cellar) and the opponent's attacks (Militia, Bureaucrat,
Bandit, Moat). When the replay can't follow the log it falls back to manual
entry. `tests/test_log_parser_edge_cases.py` replays 13 real games. Not
covered: your own Library, reactions to a Throne-Roomed attack, a double
reshuffle, two Sentries' topdeck order, and more than two players.

The network misjudges some penultimate Provinces (taking the second-to-last
while not ahead lets the opponent win with the last), so the relay decides
those by 200 playouts each of Province and its best alternative
(`PPOAgent(province_playouts=...)`). Added to the greedy policy, that check
wins 1.5 points more against 2.4 itself and 1.0 more against BigMoney +
terminal (6000 games each).

## Phase 1: MCTS self-play (AlphaZero-style), abandoned

`mcts.py`'s PUCT search drove a self-play loop. Its best checkpoint,
`domibot_v4.4`, plays Big Money + Witch and never chains Actions (40/60 vs
BigMoney). Its value targets were whole-game outcomes, which bury a buy's
credit under a game's worth of noise, and self-play only had to beat itself.

## Archived code

Removed from the tree; `git show 8957e7c:<path>` restores any of it.

| code | what it was |
|---|---|
| `training/self_play.py`, `training/train.py`, `tests/test_mcts_and_training_loop.py` | Phase 1's self-play loop, with root noise, action bias, determinization ensembles and a kingdom curriculum (`--min-sub-decision-cards`) |
| `training/ppo/distill.py` | distillation into a larger network (the 512 × 6 attempt) |
| `--opponent-pool-size` / `--opponent-pool-frac` in `training/ppo/train.py` | games against this run's recent checkpoints; the league's snapshots do the same |
