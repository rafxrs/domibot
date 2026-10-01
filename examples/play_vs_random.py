"""Play one game in the terminal against a bot that picks random legal actions.

    python examples/play_vs_random.py [seed]

`play_in_terminal` is also the text interface of `play_vs_domibot.py`.
"""
from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

from domibot import Action, Game, KINGDOM_CARDS

GAME_LOGS_DIR = Path(__file__).resolve().parent.parent / "game_logs"
HUMAN, BOT = 0, 1


def print_state(game: Game, bot_name: str) -> None:
    p, opp = game.players[HUMAN], game.players[BOT]
    whose = "your" if game.current_player == HUMAN else f"{bot_name}'s"
    reacting = "" if game.current_player == HUMAN else f"  (you're reacting during {bot_name}'s turn)"
    print(f"\n--- Turn {game.current_turn_number}, {whose} turn | {game.phase.name} phase "
          f"| actions={p.actions} buys={p.buys} coins={p.coins} ---{reacting}")
    print(f"Your hand:      {sorted(p.hand)}")
    if p.play_area:
        print(f"Your play area: {p.play_area}")
    print(f"{bot_name}: {len(opp.hand)} cards in hand, {opp.total_cards() - len(opp.hand)} elsewhere")
    print("Supply: " + ", ".join(f"{n}:{c}" for n, c in sorted(game.supply.items()) if c > 0))
    if game.trash:
        print(f"Trash: {sorted(game.trash)}")


def choose_action(game: Game, actions: list[Action]) -> Action:
    if game.pending_decision is not None:
        print(game.pending_decision.prompt)
    elif game.phase.name == "ACTION":
        print("Choose an Action card to play, or end your Action phase.")
    else:
        print("Buy a card, or end your Buy phase. (Treasures are played for you automatically.)")
    for i, a in enumerate(actions):
        print(f"  [{i}] {a}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(actions):
            return actions[int(raw)]
        print("invalid choice, try again")


def play_in_terminal(game: Game, bot, bot_name: str) -> None:
    """You play seat 0 at the prompt; `bot` plays seat 1."""
    while not game.is_game_over():
        if game.current_decider() == HUMAN:
            print_state(game, bot_name)
            action = choose_action(game, game.legal_actions())
        else:
            action = bot.act(game)
            print(f"\n[{bot_name}] {action}")
        game.step(action)


def finish(game: Game, bot_name: str, log_dir: Path, seed: int) -> None:
    """Print the result and offer to save the log."""
    scores, winners = game.get_scores(), game.winners()
    print(f"\n=== Game over ===\nScores: you={scores[HUMAN]}  {bot_name}={scores[BOT]}")
    print("You win!" if winners == [HUMAN] else f"{bot_name} wins." if winners == [BOT] else "Tie.")
    if input("Save game log? [y/N] ").strip().lower() == "y":
        path = log_dir / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}.log"
        game.save_log(path)
        print(f"Saved to {path}")


class RandomBot:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def act(self, game: Game) -> Action:
        return self.rng.choice(game.legal_actions())


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(1_000_000)
    rng = random.Random(seed)
    kingdom = rng.sample(list(KINGDOM_CARDS), 10)
    print(f"Kingdom (seed {seed}): {sorted(kingdom)}")
    game = Game(kingdom, num_players=2, seed=seed)
    play_in_terminal(game, RandomBot(rng), "bot")
    finish(game, "bot", GAME_LOGS_DIR / "human_vs_random", seed)


if __name__ == "__main__":
    main()
