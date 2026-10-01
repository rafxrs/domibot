"""PPO training loop: rollouts, GAE, clipped-surrogate updates, periodic evals.

    python -m training.ppo.train --iterations 2400 --games-per-iter 50
"""
from __future__ import annotations

import argparse
import functools
import random
import time
from pathlib import Path

import numpy as np
import torch

from .. import encoding
from ..agents import BigMoneyAgent, BigMoneyTerminalAgent, DomibotAgent, PPOAgent
from ..env import MAX_MOVES, win_weighted_value
from ..evaluate import play_match
from ..network import HIDDEN_DIM, NUM_RESIDUAL_BLOCKS, DomibotNet, get_device
from ..plan_search import load_plans
from ..strategy_bots import SCRIPTED
from .explore import ExploreConfig
from .gae import Transition
from .league import collect_league_rollouts, load_league
from .rollout import collect_rollouts

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "domibot2"


def ppo_update(network: DomibotNet, optimizer: torch.optim.Optimizer, transitions: list[Transition],
               device: torch.device, clip_eps: float = 0.2, epochs: int = 4, minibatch_size: int = 256,
               value_loss_weight: float = 0.5, entropy_coef: float = 0.01, grad_clip: float = 0.5,
               target_kl: float | None = None, imitate: float = 0.0) -> dict[str, float]:
    """`epochs` passes of minibatch clipped-surrogate updates, with normalized advantages.

    Forced moves (one legal action) and steering plans' buys train only the
    value head. `target_kl` stops after an epoch whose mean approximate KL
    exceeds 1.5x it. `imitate` > 0 adds self-imitation (Oh et al. 2018) of
    steered buys: `imitate` x mean of -log pi(buy) x max(0, return - value).
    Returns each statistic's mean over the minibatches run.
    """
    obs = torch.from_numpy(np.stack([t.obs for t in transitions])).to(device)
    mask = torch.from_numpy(np.stack([t.mask for t in transitions])).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=device)
    advantages = torch.tensor([t.advantage for t in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([t.return_ for t in transitions], dtype=torch.float32, device=device)
    on_policy = ~torch.tensor([t.explore for t in transitions], dtype=torch.bool, device=device)
    free = (mask.sum(dim=-1) > 1) & on_policy
    if free.sum() > 1:
        advantages = (advantages - advantages[free].mean()) / (advantages[free].std() + 1e-8)

    stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                                                      "imitation")}
    for _ in range(epochs):
        epoch_kls = []
        perm = np.random.permutation(len(transitions))
        for start in range(0, len(transitions), minibatch_size):
            mb = torch.from_numpy(perm[start:start + minibatch_size]).to(device)
            logits, values = network(obs[mb])
            dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask[mb], -1e9))
            log_ratio = dist.log_prob(actions[mb]) - old_log_probs[mb]
            ratio = torch.exp(log_ratio)
            w = free[mb].float()
            n_free = w.sum().clamp(min=1.0)
            with torch.no_grad():
                approx_kl = float((((ratio - 1) - log_ratio) * w).sum() / n_free)
                clipfrac = float((((ratio - 1).abs() > clip_eps).float() * w).sum() / n_free)
            epoch_kls.append(approx_kl)

            surrogate = torch.min(ratio * advantages[mb], torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb])
            policy_loss = -(surrogate * w).sum() / n_free
            entropy = (dist.entropy() * w).sum() / n_free
            value_loss = ((values - returns[mb]) ** 2).mean()
            loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy
            imitation = None
            steered = (~on_policy[mb]).float()
            if imitate > 0 and steered.sum() > 0:
                gain = (returns[mb] - values.detach()).clamp(min=0) * steered
                imitation = -(dist.log_prob(actions[mb]) * gain).sum() / steered.sum()
                loss = loss + imitate * imitation

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()
            for k, v in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy),
                         ("imitation", imitation)):
                if v is not None:
                    stats[k].append(float(v.item()))
            stats["approx_kl"].append(approx_kl)
            stats["clipfrac"].append(clipfrac)
        if target_kl is not None and sum(epoch_kls) / len(epoch_kls) > 1.5 * target_kl:
            break
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in stats.items()}
    out["updates"] = len(stats["policy_loss"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--max-moves", type=int, default=MAX_MOVES, help="decision cap per game")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs-per-update", type=int, default=4, help="passes over each batch")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final-frac", type=float, default=1.0,
                   help="cosine-decay the learning rate to this fraction of --lr over the run (1.0: constant)")
    p.add_argument("--value-loss-weight", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None, help="stop an update early past 1.5x this KL")
    p.add_argument("--reward-win-weight", type=float, default=0.8,
                   help="reward = this x win/loss + the rest x tanh VP margin (env.win_weighted_value)")
    p.add_argument("--explore-frac", type=float, default=0.0,
                   help="share of self-play games whose buys follow whole plans (ppo/explore.py)")
    p.add_argument("--explore-turn-limit", type=int, default=16, help="steer buys for 1 to this many turns")
    p.add_argument("--explore-plans", type=str, nargs="*", default=[],
                   help="steer one player with the plans in these plan_search.py --out files, on their boards")
    p.add_argument("--explore-plans-min-score", type=float, default=0.5,
                   help="only boards whose best plan scored at least this")
    p.add_argument("--imitate-steered", type=float, default=0.0,
                   help="weight of self-imitation of steered buys that paid off (see ppo_update)")
    p.add_argument("--public-features", action=argparse.BooleanOptionalAction, default=True,
                   help="read encoding.encode_public_extras (added zero-initialized on resume)")
    p.add_argument("--zone-features", action=argparse.BooleanOptionalAction, default=True,
                   help="read encoding.encode_own_zones; needs --public-features")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="a fresh network's width")
    p.add_argument("--num-blocks", type=int, default=NUM_RESIDUAL_BLOCKS, help="a fresh network's depth")
    p.add_argument("--league-frac", type=float, default=0.0, help="share of games against the league (ppo/league.py)")
    p.add_argument("--league-checkpoints", type=str, nargs="*", default=[], help="checkpoint paths or globs")
    p.add_argument("--league-scripted", type=str, nargs="*", default=[], choices=sorted(SCRIPTED))
    p.add_argument("--league-opponents-per-iter", type=int, default=4)
    p.add_argument("--league-snapshot-every", type=int, default=0,
                   help="add a frozen copy of the learner every this many iterations (0: never)")
    p.add_argument("--league-max-snapshots", type=int, default=4)
    p.add_argument("--league-hard-power", type=float, default=2.0, help="PFSP: weight (1 - learner score) ** this")
    p.add_argument("--league-uniform-mix", type=float, default=0.2, help="share of sampling weight spread evenly")
    p.add_argument("--league-decay", type=float, default=0.95, help="per-iteration decay of recorded results")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-reference-checkpoint", type=str, default=None,
                   help="also eval against this checkpoint with MCTS (agents.DomibotAgent), e.g. domibot_v4.4.pt")
    p.add_argument("--eval-reference-simulations", type=int, default=100)
    p.add_argument("--eval-reference-games", type=int, default=None, help="default: --eval-games")
    p.add_argument("--eval-reference-every", type=int, default=None, help="default: --eval-every")
    p.add_argument("--eval-rival-checkpoint", type=str, default=None,
                   help="also eval against this checkpoint's raw policy, e.g. the current release")
    p.add_argument("--eval-rival-games", type=int, default=None, help="default: --eval-games")
    p.add_argument("--checkpoint", type=str, default=None, help="resume from this checkpoint")
    p.add_argument("--start-iteration", type=int, default=1)
    p.add_argument("--run-name", type=str, default="domibot2",
                   help="checkpoints are <run-name>_latest.pt and <run-name>_iter_N.pt in checkpoints/domibot2/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.zone_features and not args.public_features:
        p.error("--zone-features needs --public-features (use --no-zone-features without them)")
    return args


def load_network(args: argparse.Namespace, device: torch.device) -> DomibotNet:
    if not args.checkpoint:
        return DomibotNet(hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                          extra_dim=encoding.EXTRA_DIM if args.public_features else 0,
                          zones_dim=encoding.ZONES_DIM if args.zone_features else 0).to(device)
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    print(f"resumed from {args.checkpoint} ({network.hidden_dim}x{network.num_blocks})")
    if args.public_features and not network.extra_dim:
        network = network.with_extra_inputs()
        print(f"added {network.extra_dim} public-feature inputs (zero-initialized)")
    if args.zone_features and not network.zones_dim:
        network = network.with_zone_inputs()
        print(f"added {network.zones_dim} own-zone inputs (zero-initialized)")
    return network


def frozen(path: str, device: torch.device) -> DomibotNet:
    net = DomibotNet.load(path, map_location=device).to(device)
    net.eval()
    return net


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    print(" | ".join(f"{k}: {v}" for k, v in vars(args).items()))
    reward_fn = functools.partial(win_weighted_value, win_weight=args.reward_win_weight)
    network = load_network(args, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.iterations, 1),
                                                            eta_min=args.lr * args.lr_final_frac)
                 if args.lr_final_frac < 1.0 else None)

    explore = None
    if args.explore_frac > 0:
        plans = tuple(p for path in args.explore_plans for p in load_plans(path, args.explore_plans_min_score))
        if args.explore_plans and not plans:
            raise SystemExit(f"no board in {args.explore_plans} has a plan scoring {args.explore_plans_min_score}+")
        explore = ExploreConfig(frac=args.explore_frac, turn_limit=args.explore_turn_limit, plans=plans)
    league = None
    if args.league_frac > 0:
        league = load_league(args.league_checkpoints, args.league_scripted, device,
                             hard_power=args.league_hard_power, uniform_mix=args.league_uniform_mix,
                             decay=args.league_decay, max_snapshots=args.league_max_snapshots)
        print("league: " + ", ".join(o.name for o in league.opponents))
    evals = [("BigMoney", BigMoneyAgent(), args.eval_games, args.eval_every),
             ("BigMoney+terminal", BigMoneyTerminalAgent(), args.eval_games, args.eval_every)]
    if args.eval_reference_checkpoint:
        evals.append((Path(args.eval_reference_checkpoint).stem,
                      DomibotAgent(frozen(args.eval_reference_checkpoint, device),
                                   num_simulations=args.eval_reference_simulations, device=device),
                      args.eval_reference_games or args.eval_games, args.eval_reference_every or args.eval_every))
    if args.eval_rival_checkpoint:
        evals.append((Path(args.eval_rival_checkpoint).stem, PPOAgent(frozen(args.eval_rival_checkpoint, device)),
                      args.eval_rival_games or args.eval_games, args.eval_every))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    end = args.start_iteration + args.iterations - 1
    for iteration in range(args.start_iteration, end + 1):
        network.eval()
        t0 = time.time()
        league_games = round(args.games_per_iter * args.league_frac) if league else 0
        games, steer_stats, note = [], {}, ""
        if args.games_per_iter > league_games:
            games = collect_rollouts(network, args.games_per_iter - league_games, max_moves=args.max_moves,
                                     device=device, seed=rng.randrange(2**31), reward_fn=reward_fn, gamma=args.gamma,
                                     lam=args.gae_lambda, explore=explore, stats=steer_stats)
        if league_games:
            drawn = league.sample(args.league_opponents_per_iter, rng)
            per_game = [drawn[j % len(drawn)] for j in range(league_games)]
            league_transitions, results = collect_league_rollouts(
                network, per_game, max_moves=args.max_moves, device=device, seed=rng.randrange(2**31),
                reward_fn=reward_fn, gamma=args.gamma, lam=args.gae_lambda)
            games += league_transitions
            league.end_iteration()
            by_name: dict[str, list[float]] = {}
            for opponent, result in zip(per_game, results):
                by_name.setdefault(opponent.name, []).append(result)
            for opponent in set(per_game):
                league.record(opponent, by_name[opponent.name])
            note = "  league=" + ",".join(f"{name}:{sum(r):g}/{len(r)}" for name, r in by_name.items())
        rollout_time = time.time() - t0
        transitions = [t for game in games for t in game]

        network.train()
        st = ppo_update(network, optimizer, transitions, device, clip_eps=args.clip_eps,
                        epochs=args.epochs_per_update, minibatch_size=args.minibatch_size,
                        value_loss_weight=args.value_loss_weight, entropy_coef=args.entropy_coef,
                        grad_clip=args.grad_clip, target_kl=args.target_kl, imitate=args.imitate_steered)
        if scheduler is not None:
            scheduler.step()
        network.save(CHECKPOINT_DIR / f"{args.run_name}_latest.pt")
        msg = (f"iter {iteration}/{end}  transitions={len(transitions)}  rollout={rollout_time:.1f}s  "
               f"update={time.time() - t0 - rollout_time:.1f}s  policy_loss={st['policy_loss']:.4f}  "
               f"value_loss={st['value_loss']:.4f}  entropy={st['entropy']:.4f}  kl={st['approx_kl']:.4f}  "
               f"clipfrac={st['clipfrac']:.3f}  updates={st['updates']}")
        if explore is not None:
            msg += f"  steered_won={steer_stats.get('won', 0)}/{steer_stats.get('games', 0)}"
        if args.imitate_steered > 0:
            msg += f"  imitation={st['imitation']:.4f}"
        msg += note
        if scheduler is not None:
            msg += f"  lr={optimizer.param_groups[0]['lr']:.2e}"
        print(msg, flush=True)

        if iteration % args.eval_every == 0:
            network.eval()
            agent = PPOAgent(network, device=device)
            for label, opponent, n_games, every in evals:
                if iteration % every == 0:
                    r = play_match(agent, opponent, n_games=n_games, seed=iteration)
                    print(f"  eval vs {label}: {r['agent_a_wins']}/{r['games']} wins, {r['ties']} ties", flush=True)
            if league is not None:
                print(f"  league (learner score/weight): {league.summary()}", flush=True)
            network.save(CHECKPOINT_DIR / f"{args.run_name}_iter_{iteration}.pt")
        if league is not None and args.league_snapshot_every and iteration % args.league_snapshot_every == 0:
            league.add_snapshot(network, f"self@{iteration}")


if __name__ == "__main__":
    main()
