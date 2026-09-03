"""Save a played (or in-progress) game's action history to a file.

Not named `logging.py` to avoid shadowing the stdlib module.
"""
from __future__ import annotations

import json
from pathlib import Path


def to_text(game) -> str:
    lines = [
        f"kingdom: {', '.join(sorted(game.kingdom))}",
        f"players: {game.num_players}",
        f"seed: {game.seed}",
        "",
    ]
    for entry in game.action_log:
        lines.append(f"turn {entry.turn:>3}  P{entry.player}  {entry.action}")
    if game.is_game_over():
        lines.append("")
        lines.append(f"scores: {game.get_scores()}")
        lines.append(f"winners: {game.winners()}")
    return "\n".join(lines) + "\n"


def to_dict(game) -> dict:
    data = {
        "kingdom": sorted(game.kingdom),
        "num_players": game.num_players,
        "seed": game.seed,
        "actions": [
            {"turn": e.turn, "player": e.player, "verb": e.action.verb, "card": e.action.card}
            for e in game.action_log
        ],
    }
    if game.is_game_over():
        data["scores"] = game.get_scores()
        data["winners"] = game.winners()
    return data


def save(game, path: str | Path, fmt: str = "text") -> None:
    """Write the game's action log to `path`. `fmt` is 'text' (human-readable,
    the default) or 'json' (structured, e.g. for a future training pipeline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "text":
        path.write_text(to_text(game), encoding="utf-8")
    elif fmt == "json":
        path.write_text(json.dumps(to_dict(game), indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unknown fmt: {fmt!r} (expected 'text' or 'json')")
