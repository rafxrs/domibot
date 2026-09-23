"""PPO self-play training loop for domibot 2 -- see training/ppo/__init__.py
and training/README.md's "domibot 2" section for why this exists alongside
the MCTS lineage in training/train.py, not instead of it.

    python -m training.ppo.train
    python -m training.ppo.train --iterations 200 --games-per-iter 64

Reuses training.network.DomibotNet unchanged as the actor-critic network
(nothing about it is MCTS-specific -- it's a plain (obs) -> (policy_logits,
value) residual MLP) and training.evaluate.play_match/training.agents'
baselines unchanged for eval, so results are directly comparable to every
number already measured for the MCTS lineage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from domibot import Action, Game

from .. import encoding
from ..agents import BigMoneyAgent, BigMoneyTerminalAgent, DomibotAgent
from ..evaluate import play_match
from ..network import DomibotNet, get_device
from ..self_play import DEFAULT_MAX_MOVES
from .gae import Transition
from .rollout import collect_rollouts

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


class PPOAgent:
    """Wraps a PPO-trained DomibotNet behind the same `act(game) -> Action`
    interface every other agent in `training/agents.py` uses, so it drops
    straight into `evaluate.play_match` -- no search, just the policy
    head's own (masked, greedy) choice, since Stage 1 has no inference-
    time search layer yet (see the plan's Stage 4)."""

    def __init__(self, network: torch.nn.Module, device: torch.device | None = None):
        self.network = network
        self.device = device or next(network.parameters()).device

    def act(self, game: Game) -> Action:
        decider = game.current_decider()
        obs = encoding.encode_observation(game, decider)
        mask = encoding.legal_action_mask(game)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _value = self.network(obs_t)
        masked = logits.masked_fill(~mask_t, -1e9)
        action_idx = int(masked.argmax(dim=-1).item())
        return encoding.index_to_action(action_idx)


def ppo_update(
    network: DomibotNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    clip_eps: float = 0.2,
    epochs: int = 4,
    minibatch_size: int = 256,
    value_loss_weight: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip: float = 0.5,
) -> tuple[float, float, float]:
    """One PPO update: `epochs` passes of minibatch clipped-surrogate
    updates over `transitions` (already carrying `.advantage`/`.return_`
    from `gae.compute_gae`). Advantage normalization is the standard PPO
    stabilization trick. Returns mean (policy_loss, value_loss, entropy)
    across all minibatches/epochs, for logging."""
    obs = torch.from_numpy(np.stack([t.obs for t in transitions])).to(device)
    mask = torch.from_numpy(np.stack([t.mask for t in transitions])).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=device)
    advantages = torch.tensor([t.advantage for t in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([t.return_ for t in transitions], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(transitions)
    policy_losses, value_losses, entropies = [], [], []
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            mb = torch.from_numpy(perm[start:start + minibatch_size]).to(device)
            logits, values = network(obs[mb])
            masked_logits = logits.masked_fill(~mask[mb], -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[mb])
            loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return (sum(policy_losses) / len(policy_losses),
            sum(value_losses) / len(value_losses),
            sum(entropies) / len(entropies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=64,
                         help="self-play games collected per iteration, stepped side by side sharing one batched "
                              "network forward pass per round (see ppo.rollout.collect_rollouts) -- no tree search, "
                              "so no --simulations knob, iteration speed should be far faster than the MCTS lineage")
    parser.add_argument("--max-moves", type=int, default=None,
                         help="safety cap on decisions per self-play game (default: self_play.DEFAULT_MAX_MOVES)")
    parser.add_argument("--min-sub-decision-cards", type=int, default=0,
                         help="see self_play._sample_kingdom -- same curriculum knob, 0 disables it")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE discount (episodic/undiscounted by default)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipped-surrogate epsilon")
    parser.add_argument("--epochs-per-update", type=int, default=4, help="minibatch passes over each rollout batch")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                         help="entropy bonus weight -- PPO's exploration driver, replacing the MCTS lineage's "
                              "Dirichlet-noise-at-root/action_bias; probably matters a lot for whether it ever "
                              "tries chaining action cards, so worth tuning deliberately, not left at the default")
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-reference-checkpoint", type=str, default=None,
                         help="also eval against this checkpoint (e.g. checkpoints/domibot_v4.4.pt) via "
                              "agents.DomibotAgent, for a same-footing comparison against the MCTS lineage")
    parser.add_argument("--eval-reference-simulations", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from this checkpoint file")
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    max_moves = args.max_moves if args.max_moves is not None else DEFAULT_MAX_MOVES
    print(f"device: {device}  |  max_moves: {max_moves}  |  min_sub_decision_cards: {args.min_sub_decision_cards}")
    print(f"games_per_iter: {args.games_per_iter}  |  epochs_per_update: {args.epochs_per_update}  |  "
          f"minibatch_size: {args.minibatch_size}  |  lr: {args.lr}  |  gamma: {args.gamma}  |  "
          f"gae_lambda: {args.gae_lambda}  |  clip_eps: {args.clip_eps}  |  entropy_coef: {args.entropy_coef}")

    network = DomibotNet.load(args.checkpoint, map_location=device).to(device) if args.checkpoint else DomibotNet().to(device)
    if args.checkpoint:
        print(f"resumed from {args.checkpoint}")
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    reference_agent = None
    if args.eval_reference_checkpoint:
        ref_net = DomibotNet.load(args.eval_reference_checkpoint, map_location=device).to(device)
        ref_net.eval()
        reference_agent = DomibotAgent(ref_net, num_simulations=args.eval_reference_simulations, device=device)
        print(f"reference opponent: {args.eval_reference_checkpoint}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    end_iteration = args.start_iteration + args.iterations - 1
    for iteration in range(args.start_iteration, end_iteration + 1):
        network.eval()
        t0 = time.time()
        games = collect_rollouts(
            network, args.games_per_iter, max_moves=max_moves, device=device,
            seed=args.seed * 1_000_003 + iteration, min_sub_decision_cards=args.min_sub_decision_cards,
            gamma=args.gamma, lam=args.gae_lambda,
        )
        rollout_time = time.time() - t0
        transitions = [t for game in games for t in game]

        network.train()
        pl, vl, ent = ppo_update(
            network, optimizer, transitions, device,
            clip_eps=args.clip_eps, epochs=args.epochs_per_update, minibatch_size=args.minibatch_size,
            value_loss_weight=args.value_loss_weight, entropy_coef=args.entropy_coef, grad_clip=args.grad_clip,
        )
        update_time = time.time() - t0 - rollout_time

        network.save(CHECKPOINT_DIR / "domibot2_latest.pt")
        print(f"iter {iteration}/{end_iteration}  transitions={len(transitions)}  "
              f"rollout={rollout_time:.1f}s  update={update_time:.1f}s  "
              f"policy_loss={pl:.4f}  value_loss={vl:.4f}  entropy={ent:.4f}", flush=True)

        if iteration % args.eval_every == 0:
            network.eval()
            agent = PPOAgent(network, device=device)
            result = play_match(agent, BigMoneyAgent(), n_games=args.eval_games, seed=iteration)
            print(f"  eval vs BigMoney: {result['agent_a_wins']}/{result['games']} wins, {result['ties']} ties", flush=True)
            bmt_result = play_match(agent, BigMoneyTerminalAgent(), n_games=args.eval_games, seed=iteration)
            print(f"  eval vs BigMoney+terminal: {bmt_result['agent_a_wins']}/{bmt_result['games']} wins, "
                  f"{bmt_result['ties']} ties", flush=True)
            if reference_agent is not None:
                ref_result = play_match(agent, reference_agent, n_games=args.eval_games, seed=iteration)
                print(f"  eval vs {Path(args.eval_reference_checkpoint).stem}: "
                      f"{ref_result['agent_a_wins']}/{ref_result['games']} wins, {ref_result['ties']} ties", flush=True)
            network.save(CHECKPOINT_DIR / f"domibot2_iter_{iteration}.pt")


if __name__ == "__main__":
    main()
