"""Self-play training loop for Domibot: generate games with MCTS + the
current network, train on the resulting (state, MCTS policy, outcome)
examples, checkpoint, periodically measure progress against BigMoneyAgent,
repeat.

    python -m training.train
    python -m training.train --iterations 200 --games-per-iter 20 --simulations 150
    python -m training.train --reference-checkpoint checkpoints/domibot_v1.4.pt --eval-games 40

GPU note: self-play runs `--parallel-games` games at a time side by side
(`self_play.play_self_play_games_batch`), sharing one batched network
forward pass across all of them at every simulation instead of paying a
batch-size-1 forward pass per game per simulation (see `mcts.run_mcts_batch`
/ `mcts.evaluate_nodes_batch`). This is root-parallelism across independent
games, not parallelism inside a single game's tree, so the per-game move
sequence is unaffected -- only how the network gets called. Raise
`--parallel-games` toward your GPU's real batch-throughput sweet spot to
make higher `--simulations` counts affordable; the Python game engine
driving move selection is still the other half of the cost and doesn't
benefit from this. `get_device()` already picks CUDA automatically whenever
it's available.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import encoding
from .agents import BigMoneyAgent, DomibotAgent
from .evaluate import play_match
from .network import DomibotNet, get_device
from .self_play import DEFAULT_MAX_MOVES, ReplayBuffer, play_self_play_games_batch

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def train_step(network: DomibotNet, optimizer: torch.optim.Optimizer, batch, device: torch.device,
               value_loss_weight: float = 1.0, grad_clip: float = 0.0) -> tuple[float, float]:
    obs = torch.from_numpy(np.stack([e.obs for e in batch])).to(device)
    mask = torch.from_numpy(np.stack([e.mask for e in batch])).to(device)
    policy_target = torch.from_numpy(np.stack([e.policy_target for e in batch])).to(device)
    value_target = torch.tensor([e.value_target for e in batch], dtype=torch.float32, device=device)

    value_known = torch.tensor([e.value_known for e in batch], dtype=torch.bool, device=device)

    policy_logits, value_pred = network(obs)
    masked_logits = policy_logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
    # Examples from games that hit max_moves carry no real outcome, so they
    # are excluded from the value term (their policy target still counts).
    if value_known.any():
        value_loss = F.mse_loss(value_pred[value_known], value_target[value_known])
    else:
        value_loss = value_pred.sum() * 0.0
    # Weighted, because the two terms are not on comparable scales: policy
    # cross-entropy sits near its target-entropy floor (~0.6) while value
    # MSE on tanh-squashed margins is far smaller, so an unweighted sum
    # hands the value head only a few percent of the gradient.
    loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move during self-play")
    parser.add_argument("--parallel-games", type=int, default=32,
                         help="self-play games advanced side by side, sharing one batched network forward pass "
                              "per simulation round (see self_play.play_self_play_games_batch). Higher values "
                              "trade GPU memory for self-play throughput; --games-per-iter is split into chunks "
                              "of this size.")
    parser.add_argument("--action-bias", type=float, default=0.2,
                         help="self-play-only nudge toward continuing to play Action cards over ending the phase "
                              "early, applied at every search node (see mcts._apply_action_continuation_bias). "
                              "0 disables it.")
    parser.add_argument("--max-moves", type=int, default=None,
                         help="safety cap on decisions per self-play game, sub-decisions included, for an "
                              "undertrained policy that can otherwise stall indefinitely (default: "
                              "self_play.DEFAULT_MAX_MOVES)")
    parser.add_argument("--min-sub-decision-cards", type=int, default=0,
                         help="force at least this many of each self-play kingdom's 10 cards to come from "
                              "self_play.SUB_DECISION_CARDS (cards with a real trash/discard/gain/topdeck "
                              "judgment call), instead of plain uniform sampling -- boosts how often several "
                              "judgment-heavy cards land in the same kingdom together, a rarer joint event under "
                              "plain sampling than any one card's own frequency. 0 (default) disables this.")
    parser.add_argument("--determinization-ensemble-size", type=int, default=1,
                         help="for each plain phase-action decision during self-play, search this many "
                              "independently-redealt hidden-info samples (mcts.redeal_hidden_info) instead of the "
                              "one true (but actually hidden, from the deciding player's own point of view) deal, "
                              "and merge their visit counts -- a multi-determinization form of PIMC that averages "
                              "the search over several plausible opponent hands instead of committing the whole "
                              "tree to whichever one the self-play game actually has. --simulations is split "
                              "evenly across the ensemble (at least 1 each), so a bigger value trades search depth "
                              "per world for world diversity at a fixed simulation budget. 1 (default) disables "
                              "this, identical to today's behavior. Does not apply to sub-decision searches yet.")
    parser.add_argument("--train-steps-per-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final-frac", type=float, default=1.0,
                         help="cosine-decay the learning rate from --lr down to this fraction of it over the "
                              "run's --iterations (1.0, the default, keeps the old flat-LR behavior). E.g. 0.1 "
                              "decays 1e-3 -> 1e-4. On a resume the schedule restarts, so set --lr to wherever "
                              "the previous run left off rather than expecting it to continue the curve.")
    parser.add_argument("--value-loss-weight", type=float, default=1.0,
                         help="multiplier on the value MSE term in the joint loss. The two terms are not on "
                              "comparable scales (policy cross-entropy floors near the target distribution's own "
                              "entropy, ~0.6; value MSE on tanh-squashed margins is much smaller), so 1.0 gives "
                              "the value head only a few percent of the gradient.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                         help="clip gradient norm to this value (0 disables)")
    parser.add_argument("--eval-every", type=int, default=5, help="iterations between eval checks")
    parser.add_argument("--eval-games", type=int, default=20, help="games per eval opponent (BigMoney and, if set, --reference-checkpoint)")
    parser.add_argument("--eval-simulations", type=int, default=100, help="MCTS simulations per move during eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from this checkpoint file")
    parser.add_argument("--reference-checkpoint", type=str, default=None,
                         help="if set, also eval every --eval-every iterations against this fixed checkpoint "
                              "(loaded once, frozen for the whole run) in addition to BigMoney")
    parser.add_argument("--start-iteration", type=int, default=1,
                         help="iteration number to start counting from when resuming, so iter_N.pt snapshots "
                              "continue a previous run's numbering instead of overwriting it from iter_5.pt again")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                         help="device for training/eval (default: CUDA if available, else CPU)")
    parser.add_argument("--self-play-device", type=str, default=None,
                         help="device for self-play generation only (default: same as --device). Our benchmarks "
                              "found self-play is bottlenecked by the unbatched Python game-engine loop, not the "
                              "(tiny) network's forward pass, so CPU self-play alongside GPU training/eval is often "
                              "just as fast and frees the GPU for the training step -- pass 'cpu' here to do that.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    self_play_device = torch.device(args.self_play_device) if args.self_play_device else device
    max_moves = args.max_moves if args.max_moves is not None else DEFAULT_MAX_MOVES
    print(f"device: {device}  |  self-play device: {self_play_device}  |  max_moves: {max_moves}  |  "
          f"min_sub_decision_cards: {args.min_sub_decision_cards}  |  "
          f"determinization_ensemble_size: {args.determinization_ensemble_size}")
    print(f"games_per_iter: {args.games_per_iter}  |  train_steps_per_iter: {args.train_steps_per_iter}  |  "
          f"batch_size: {args.batch_size}  |  simulations: {args.simulations}  |  "
          f"lr: {args.lr} -> {args.lr * args.lr_final_frac:g}  |  value_loss_weight: {args.value_loss_weight}")

    network = DomibotNet.load(args.checkpoint, map_location=device).to(device) if args.checkpoint else DomibotNet().to(device)
    if args.checkpoint:
        print(f"resumed from {args.checkpoint}")
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.iterations, 1), eta_min=args.lr * args.lr_final_frac)
        if args.lr_final_frac < 1.0 else None)
    buffer = ReplayBuffer(args.buffer_capacity)
    rng = random.Random(args.seed)

    # Self-play runs off a separate copy of the weights when --self-play-device
    # differs from --device, since a single nn.Module lives on one device at a
    # time; kept in sync via load_state_dict (a plain tensor copy, which
    # transparently crosses devices) right before each iteration's self-play.
    self_play_network = network if self_play_device == device else DomibotNet().to(self_play_device)

    reference_agent = None
    if args.reference_checkpoint:
        stale = (f"--reference-checkpoint {args.reference_checkpoint} predates the observation/network "
                 f"change (this encoder produces {encoding.OBS_DIM} features) and cannot be played "
                 f"against the current code. Drop --reference-checkpoint and use the BigMoney eval for "
                 f"in-training progress; compare against the older lineage separately.")
        try:
            reference_net = DomibotNet.load(args.reference_checkpoint, map_location=device).to(device)
        except RuntimeError as e:  # state_dict shape/key mismatch
            raise SystemExit(f"{stale}\n  (underlying error: {e.args[0].splitlines()[0]})")
        if reference_net.obs_dim != encoding.OBS_DIM:
            raise SystemExit(stale)
        reference_net.eval()
        reference_agent = DomibotAgent(reference_net, num_simulations=args.eval_simulations, device=device)
        print(f"reference opponent: {args.reference_checkpoint}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    end_iteration = args.start_iteration + args.iterations - 1
    for iteration in range(args.start_iteration, end_iteration + 1):
        network.eval()
        if self_play_network is not network:
            self_play_network.load_state_dict(network.state_dict())
            self_play_network.eval()
        t0 = time.time()
        remaining = args.games_per_iter
        games_this_iter = truncated_this_iter = 0
        while remaining > 0:
            chunk = min(args.parallel_games, remaining)
            games_examples = play_self_play_games_batch(
                self_play_network, chunk, args.simulations, action_bias=args.action_bias,
                max_moves=max_moves, min_sub_decision_cards=args.min_sub_decision_cards,
                determinization_ensemble_size=args.determinization_ensemble_size,
                device=self_play_device, seed=rng.randrange(2**31),
            )
            for examples in games_examples:
                buffer.add_game(examples)
                games_this_iter += 1
                if examples and not examples[0].value_known:
                    truncated_this_iter += 1
            remaining -= chunk
        self_play_time = time.time() - t0

        network.train()
        policy_losses, value_losses = [], []
        for _ in range(args.train_steps_per_iter):
            if len(buffer) < args.batch_size:
                break
            batch = buffer.sample(args.batch_size, rng)
            pl, vl = train_step(network, optimizer, batch, device,
                                value_loss_weight=args.value_loss_weight, grad_clip=args.grad_clip)
            policy_losses.append(pl)
            value_losses.append(vl)
        if scheduler is not None:
            scheduler.step()

        network.save(CHECKPOINT_DIR / "latest.pt")

        msg = f"iter {iteration}/{end_iteration}  buffer={len(buffer)}  self_play={self_play_time:.1f}s"
        if policy_losses:
            msg += f"  policy_loss={sum(policy_losses) / len(policy_losses):.4f}  value_loss={sum(value_losses) / len(value_losses):.4f}"
        # Truncation rate is the one number that says whether the value
        # targets are real: a capped game has no outcome at all.
        msg += f"  truncated={truncated_this_iter}/{games_this_iter}"
        if scheduler is not None:
            msg += f"  lr={optimizer.param_groups[0]['lr']:.2e}"
        print(msg, flush=True)

        if iteration % args.eval_every == 0:
            network.eval()
            agent = DomibotAgent(network, num_simulations=args.eval_simulations, device=device)
            result = play_match(agent, BigMoneyAgent(), n_games=args.eval_games, seed=iteration)
            print(f"  eval vs BigMoney: {result['agent_a_wins']}/{result['games']} wins, {result['ties']} ties", flush=True)
            if reference_agent is not None:
                ref_result = play_match(agent, reference_agent, n_games=args.eval_games, seed=iteration)
                print(f"  eval vs {Path(args.reference_checkpoint).stem}: "
                      f"{ref_result['agent_a_wins']}/{ref_result['games']} wins, {ref_result['ties']} ties", flush=True)
            network.save(CHECKPOINT_DIR / f"iter_{iteration}.pt")


if __name__ == "__main__":
    main()
