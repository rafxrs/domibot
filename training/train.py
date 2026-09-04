"""Self-play training loop for Domibot: generate games with MCTS + the
current network, train on the resulting (state, MCTS policy, outcome)
examples, checkpoint, periodically measure progress against BigMoneyAgent,
repeat.

    python -m training.train
    python -m training.train --iterations 200 --games-per-iter 20 --simulations 150

GPU note: this network is small (a few hundred thousand parameters) and
each self-play move currently runs the network one board at a time inside
MCTS, so the actual bottleneck is the Python game engine driving those
simulations, not GPU throughput — a CUDA build of torch still helps (the
training step itself batches nicely), but don't expect it to make self-play
itself dramatically faster without further work (e.g. batching leaf
evaluations across simulations, which this first version doesn't do).
`get_device()` already picks CUDA automatically whenever it's available.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .agents import BigMoneyAgent, DomibotAgent
from .evaluate import play_match
from .network import DomibotNet, get_device
from .self_play import ReplayBuffer, play_self_play_game

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def train_step(network: DomibotNet, optimizer: torch.optim.Optimizer, batch, device: torch.device) -> tuple[float, float]:
    obs = torch.from_numpy(np.stack([e.obs for e in batch])).to(device)
    mask = torch.from_numpy(np.stack([e.mask for e in batch])).to(device)
    policy_target = torch.from_numpy(np.stack([e.policy_target for e in batch])).to(device)
    value_target = torch.tensor([e.value_target for e in batch], dtype=torch.float32, device=device)

    policy_logits, value_pred = network(obs)
    masked_logits = policy_logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, value_target)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move during self-play")
    parser.add_argument("--train-steps-per-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=5, help="iterations between eval-vs-BigMoney checks")
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-simulations", type=int, default=100, help="MCTS simulations per move during eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from this checkpoint file")
    parser.add_argument("--start-iteration", type=int, default=1,
                         help="iteration number to start counting from when resuming, so iter_N.pt snapshots "
                              "continue a previous run's numbering instead of overwriting it from iter_5.pt again")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    network = DomibotNet.load(args.checkpoint, map_location=device).to(device) if args.checkpoint else DomibotNet().to(device)
    if args.checkpoint:
        print(f"resumed from {args.checkpoint}")
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_capacity)
    rng = random.Random(args.seed)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    end_iteration = args.start_iteration + args.iterations - 1
    for iteration in range(args.start_iteration, end_iteration + 1):
        network.eval()
        t0 = time.time()
        for _ in range(args.games_per_iter):
            examples = play_self_play_game(network, args.simulations, seed=rng.randrange(2**31))
            buffer.add_game(examples)
        self_play_time = time.time() - t0

        network.train()
        policy_losses, value_losses = [], []
        for _ in range(args.train_steps_per_iter):
            if len(buffer) < args.batch_size:
                break
            batch = buffer.sample(args.batch_size, rng)
            pl, vl = train_step(network, optimizer, batch, device)
            policy_losses.append(pl)
            value_losses.append(vl)

        network.save(CHECKPOINT_DIR / "latest.pt")

        msg = f"iter {iteration}/{end_iteration}  buffer={len(buffer)}  self_play={self_play_time:.1f}s"
        if policy_losses:
            msg += f"  policy_loss={sum(policy_losses) / len(policy_losses):.4f}  value_loss={sum(value_losses) / len(value_losses):.4f}"
        print(msg, flush=True)

        if iteration % args.eval_every == 0:
            network.eval()
            agent = DomibotAgent(network, num_simulations=args.eval_simulations, device=device)
            result = play_match(agent, BigMoneyAgent(), n_games=args.eval_games, seed=iteration)
            print(f"  eval vs BigMoney: {result['agent_a_wins']}/{result['games']} wins, {result['ties']} ties", flush=True)
            network.save(CHECKPOINT_DIR / f"iter_{iteration}.pt")


if __name__ == "__main__":
    main()
