"""Play one interactive game against a trained Domibot checkpoint.

    python examples/play_vs_domibot.py
    python examples/play_vs_domibot.py --checkpoint checkpoints/domibot_v4.4.pt --simulations 400
    python examples/play_vs_domibot.py --gpu   # only if you're not also training right now
    python examples/play_vs_domibot.py --gui   # a pygame window instead of the text prompt

Defaults to checkpoints/domibot2/domibot2.1.pt, the strongest checkpoint
in the project (see training/README.md's checkpoint lineage table).

Runs on CPU by default so it doesn't compete with a training run that may
still be using the GPU. Domibot "thinks" (runs MCTS) for a moment before
each of its play/buy decisions -- more --simulations means stronger but
slower play.
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path

import torch

from domibot import Action, Game, KINGDOM_CARDS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # training/ is a sibling of examples/, not on sys.path by default
from training.agents import DomibotAgent  # noqa: E402
from training.network import DomibotNet, get_device  # noqa: E402

GAME_LOGS_DIR = ROOT / "game_logs"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "domibot2" / "domibot2.1.pt"

HUMAN = 0
BOT = 1


def describe_decision(game: Game) -> str:
    d = game.pending_decision
    if d is not None:
        return d.prompt
    if game.phase.name == "ACTION":
        return "Choose an Action card to play, or end your Action phase."
    return "Buy a card, or end your Buy phase. (Treasures are played for you automatically.)"


def print_state(game: Game) -> None:
    p = game.players[HUMAN]
    opp = game.players[BOT]
    whose_turn = "your" if game.current_player == HUMAN else "Domibot's"
    reacting_note = "" if game.current_player == HUMAN else "  (you're reacting during Domibot's turn)"
    print(f"\n--- Turn {game.current_turn_number}, {whose_turn} turn | {game.phase.name} phase "
          f"| actions={p.actions} buys={p.buys} coins={p.coins} ---{reacting_note}")
    print(f"Your hand:      {sorted(p.hand)}")
    if p.play_area:
        print(f"Your play area: {p.play_area}")
    print(f"Domibot: {len(opp.hand)} cards in hand, {opp.total_cards() - len(opp.hand)} elsewhere")
    supply = ", ".join(f"{n}:{c}" for n, c in sorted(game.supply.items()) if c > 0)
    print(f"Supply: {supply}")
    if game.trash:
        print(f"Trash: {sorted(game.trash)}")


def choose_action(game: Game, actions: list[Action]) -> Action:
    print(describe_decision(game))
    for i, a in enumerate(actions):
        print(f"  [{i}] {a}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(actions):
            return actions[int(raw)]
        print("invalid choice, try again")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per Domibot decision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available (avoid while also training)")
    parser.add_argument("--gui", action="store_true", help="play in a pygame window instead of the text prompt")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} -- download domibot2.1.pt from https://github.com/rafxrs/domibot/releases into checkpoints/domibot2/ (see README Setup), or train your own")

    device = get_device() if args.gpu else torch.device("cpu")
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    network.eval()
    domibot = DomibotAgent(network, num_simulations=args.simulations, device=device)
    print(f"Loaded {args.checkpoint} onto {device}, {args.simulations} sims/decision.")

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    rng = random.Random(seed)
    kingdom = rng.sample(list(KINGDOM_CARDS), 10)
    print(f"Kingdom (seed {seed}): {sorted(kingdom)}")
    game = Game(kingdom, num_players=2, seed=seed)

    if args.gui:
        from gui.app import DominionGUI  # deferred: only needs pygame installed if --gui is actually used

        DominionGUI(game, domibot, human_seat=HUMAN).run()
    else:
        while not game.is_game_over():
            decider = game.current_decider()
            if decider == HUMAN:
                print_state(game)
                action = choose_action(game, game.legal_actions())
            else:
                action = domibot.act(game)
                print(f"\n[Domibot] {action}")
            game.step(action)

    print("\n=== Game over ===")
    scores = game.get_scores()
    print(f"Scores: you={scores[HUMAN]}  Domibot={scores[BOT]}")
    winners = game.winners()
    if winners == [HUMAN]:
        print("You win!")
    elif winners == [BOT]:
        print("Domibot wins.")
    else:
        print("Tie.")

    if input("Save game log? [y/N] ").strip().lower() == "y":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        matchup_dir = GAME_LOGS_DIR / f"human_vs_{Path(args.checkpoint).stem}"
        path = matchup_dir / f"{timestamp}_seed{seed}.log"
        game.save_log(path)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
