"""Play one interactive game against a bot that picks uniformly random
legal actions. Run with an optional seed for a reproducible kingdom/shuffle:

    python examples/play_vs_random.py [seed]
"""
from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

from domibot import Action, Game, KINGDOM_CARDS

GAME_LOGS_DIR = Path(__file__).resolve().parent.parent / "game_logs"

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
    whose_turn = "your" if game.current_player == HUMAN else "the bot's"
    reacting_note = "" if game.current_player == HUMAN else "  (you're reacting during the bot's turn)"
    print(f"\n--- Turn {game.current_turn_number}, {whose_turn} turn | {game.phase.name} phase "
          f"| actions={p.actions} buys={p.buys} coins={p.coins} ---{reacting_note}")
    print(f"Your hand:      {sorted(p.hand)}")
    if p.play_area:
        print(f"Your play area: {p.play_area}")
    print(f"Opponent: {len(opp.hand)} cards in hand, {opp.deck_size() - len(opp.hand)} elsewhere")
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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(1_000_000)
    rng = random.Random(seed)
    kingdom = rng.sample(list(KINGDOM_CARDS), 10)
    print(f"Kingdom (seed {seed}): {sorted(kingdom)}")
    game = Game(kingdom, num_players=2, seed=seed)

    while not game.is_game_over():
        decider = game.current_decider()
        actions = game.legal_actions()
        if decider == HUMAN:
            print_state(game)
            action = choose_action(game, actions)
        else:
            action = rng.choice(actions)
            print(f"\n[bot] {action}")
        game.step(action)

    print("\n=== Game over ===")
    scores = game.get_scores()
    print(f"Scores: you={scores[HUMAN]}  bot={scores[BOT]}")
    winners = game.winners()
    if winners == [HUMAN]:
        print("You win!")
    elif winners == [BOT]:
        print("The bot wins.")
    else:
        print("Tie.")

    if input("Save game log? [y/N] ").strip().lower() == "y":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = GAME_LOGS_DIR / f"{timestamp}_seed{seed}.log"
        game.save_log(path)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
