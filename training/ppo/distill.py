"""Policy distillation: train a (usually larger) student DomibotNet to
reproduce a trained teacher's policy and value, so a bigger network can
start PPO at the teacher's strength instead of relearning from scratch.

    python -m training.ppo.distill \
        --teacher checkpoints/domibot2/domibot2_iter_12000.pt \
        --hidden-dim 512 --num-blocks 6 \
        --out checkpoints/domibot2/domibot2_512x6_distilled.pt

Why: domibot2.1 took 8000 PPO iterations from a random network. A larger
network trained from scratch would repeat all of that before its extra
capacity could matter; distillation reaches the teacher's strength in a
few hundred iterations, so every PPO iteration afterwards goes toward
getting *past* it. The cost is that the student starts inside the
teacher's strategy and only leaves it if PPO finds something better.

Data is DAgger-style. For the first `--teacher-frac` of iterations the
teacher plays, which gives sensible states while the student is still
random. After that the student plays and the teacher labels the states the
*student* reaches, which is where an imitator's mistakes compound. Either
way every recorded decision is labeled by the teacher.

Loss: KL(teacher || student) over the legal actions, averaged over
non-forced decisions (a forced move's distribution is trivially the same),
plus `--value-weight` x MSE to the teacher's value on every decision.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..agents import BigMoneyTerminalAgent
from ..evaluate import play_match
from ..network import DomibotNet, get_device
from ..self_play import DEFAULT_MAX_MOVES
from .rollout import collect_rollouts
from .train import PPOAgent


def _masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)


@torch.no_grad()
def teacher_targets(teacher: DomibotNet, obs: torch.Tensor, mask: torch.Tensor,
                    batch_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """The teacher's masked log-probs (N, num_actions) and values (N,)."""
    log_probs, values = [], []
    for start in range(0, len(obs), batch_size):
        logits, v = teacher(obs[start:start + batch_size])
        log_probs.append(_masked_log_probs(logits, mask[start:start + batch_size]))
        values.append(v)
    return torch.cat(log_probs), torch.cat(values)


def distill_loss(student: DomibotNet, obs: torch.Tensor, mask: torch.Tensor, teacher_log_probs: torch.Tensor,
                 teacher_values: torch.Tensor, value_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    logits, values = student(obs)
    student_log_probs = _masked_log_probs(logits, mask)
    # Illegal actions have teacher log-prob ~ -1e9, so exp() zeroes their terms.
    kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1)
    free = (mask.sum(-1) > 1).float()
    policy_kl = (kl * free).sum() / free.sum().clamp(min=1.0)
    value_loss = F.mse_loss(values, teacher_values)
    with torch.no_grad():
        agree = ((logits.masked_fill(~mask, -1e9).argmax(-1) == teacher_log_probs.argmax(-1)).float() * free).sum() \
            / free.sum().clamp(min=1.0)
    return policy_kl + value_weight * value_loss, {
        "kl": policy_kl.item(), "value_mse": value_loss.item(), "top1_agree": agree.item()}


def distill_update(student: DomibotNet, optimizer: torch.optim.Optimizer, obs: torch.Tensor, mask: torch.Tensor,
                   teacher_log_probs: torch.Tensor, teacher_values: torch.Tensor, epochs: int = 2,
                   minibatch_size: int = 512, value_weight: float = 1.0, grad_clip: float = 1.0) -> dict[str, float]:
    """`epochs` minibatch passes over one iteration's data; returns means."""
    stats: dict[str, list[float]] = {"kl": [], "value_mse": [], "top1_agree": []}
    n = len(obs)
    for _ in range(epochs):
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, minibatch_size):
            mb = perm[start:start + minibatch_size]
            loss, st = distill_loss(student, obs[mb], mask[mb], teacher_log_probs[mb], teacher_values[mb], value_weight)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            for k, v in st.items():
                stats[k].append(v)
    return {k: sum(v) / len(v) for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="where to save the student")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--teacher-frac", type=float, default=0.3,
                        help="fraction of iterations (the first ones) whose games the teacher plays; the "
                             "student plays the rest")
    parser.add_argument("--epochs", type=int, default=2, help="minibatch passes over each iteration's data")
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4, help="cosine-decayed to 10%% over the run")
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=200,
                        help="games vs the teacher and vs BigMoney+terminal at each eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    teacher = DomibotNet.load(args.teacher, map_location=device).to(device)
    teacher.eval()
    student = DomibotNet(obs_dim=teacher.obs_dim, num_actions=teacher.num_actions, hidden_dim=args.hidden_dim,
                         num_blocks=args.num_blocks, extra_dim=teacher.extra_dim).to(device)
    n_params = lambda net: sum(p.numel() for p in net.parameters())
    print(f"device: {device}  |  teacher: {args.teacher} ({teacher.hidden_dim}x{teacher.num_blocks}, "
          f"{n_params(teacher):,} params)  |  student: {args.hidden_dim}x{args.num_blocks}, "
          f"{n_params(student):,} params", flush=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr * 0.1)
    full_obs = bool(teacher.extra_dim or student.extra_dim)
    teacher_iters = round(args.iterations * args.teacher_frac)
    rng = random.Random(args.seed)

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        student.eval()
        actor = teacher if iteration <= teacher_iters else student
        games = collect_rollouts(actor, args.games_per_iter, max_moves=DEFAULT_MAX_MOVES, device=device,
                                 seed=rng.randrange(2**31), full_obs=full_obs)
        transitions = [t for g in games for t in g]
        obs = torch.from_numpy(np.stack([t.obs for t in transitions])).to(device)
        mask = torch.from_numpy(np.stack([t.mask for t in transitions])).to(device)
        teacher_log_probs, teacher_values = teacher_targets(teacher, obs, mask)
        rollout_time = time.time() - t0

        # Measured on this batch before training on it: held-out fit.
        with torch.no_grad():
            _, held_out = distill_loss(student, obs, mask, teacher_log_probs, teacher_values, args.value_weight)
        student.train()
        st = distill_update(student, optimizer, obs, mask, teacher_log_probs, teacher_values, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, value_weight=args.value_weight)
        scheduler.step()
        print(f"iter {iteration}/{args.iterations}  actor={'teacher' if actor is teacher else 'student'}  "
              f"transitions={len(transitions)}  rollout={rollout_time:.1f}s  update={time.time() - t0 - rollout_time:.1f}s  "
              f"held_out_kl={held_out['kl']:.4f}  held_out_top1={held_out['top1_agree']:.3f}  "
              f"held_out_value_mse={held_out['value_mse']:.4f}  train_kl={st['kl']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            student.eval()
            student.save(args.out)
            agent = PPOAgent(student, device=device)
            vs_teacher = play_match(agent, PPOAgent(teacher, device=device), n_games=args.eval_games, seed=iteration)
            vs_bmt = play_match(agent, BigMoneyTerminalAgent(), n_games=args.eval_games, seed=iteration)
            print(f"  eval vs teacher: {vs_teacher['agent_a_wins']}/{vs_teacher['games']} wins, "
                  f"{vs_teacher['ties']} ties", flush=True)
            print(f"  eval vs BigMoney+terminal: {vs_bmt['agent_a_wins']}/{vs_bmt['games']} wins, "
                  f"{vs_bmt['ties']} ties", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
