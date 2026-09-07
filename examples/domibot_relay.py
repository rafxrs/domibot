"""A move-advisor for playing against real people: you play a real game
yourself (e.g. on dominion.games), and at each of your phase-action
decisions (what to play, what to buy) this tells you what Domibot would do,
using only what's actually visible to a player at the table.

    python examples/domibot_relay.py
    python examples/domibot_relay.py --checkpoint checkpoints/domibot_v1.4.pt --simulations 400

This never touches the real game for you -- you type in what's on screen,
it prints a recommendation, you click the move yourself. See training/relay.py
for exactly what information it needs and why (short version: your own hand
and total card ownership exactly; for the opponent, only what Dominion
actually makes public -- their discard pile, and their hand/draw-pile
*sizes*, never contents).

State is re-entered in full at every decision rather than tracked
incrementally turn to turn -- more typing per query, but far less risk of
this tool's internal state silently drifting from the real game's if a
public event gets missed or mis-entered. Only covers your own phase-action
decisions (what to play/buy); for a forced sub-decision (a trash/discard/
topdeck choice), just apply the same "keep the good stuff, give up junk"
rule Domibot itself uses for those (see training/heuristics.py) -- neither
Domibot nor this tool actually searches those.

At each query you paste in that game's dominion.games text log (the whole
thing, fresh, every time -- it keeps growing) to auto-fill the supply, the
trash, and your own total card ownership (the three genuinely tedious/
error-prone-to-tally fields) -- see training/log_parser.py for exactly what
it does and doesn't derive from it, and why. Everything it fills in still
shows up as an editable default, so you can sanity check or override it
before confirming. Paste nothing (hit END right away) to skip it for one
query and fall back to manual entry instead.

Any card list (the kingdom included -- dominion.games' log never states it,
so it's typed by hand every game) accepts a short code instead of the full
name, e.g. 'POA' for Poacher or 'CR' for Council Room -- run with
--list-abbreviations to see the full table (also printed at startup). The
kingdom itself also doesn't need commas between entries.

Your account name defaults to 'domibot_v1.4' (override with
--account-name) -- it's just whatever your dominion.games username is,
unrelated to which checkpoint --checkpoint points at.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # training/ is a sibling of examples/, not on sys.path by default
from training.log_parser import parse_dominion_log  # noqa: E402
from training.mcts import run_mcts, select_action, visit_distribution  # noqa: E402
from training.network import DomibotNet, get_device  # noqa: E402
from training.relay import CARD_ABBREVIATIONS, TableState, reconstruct_game, resolve_card_name  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"


def parse_cards(raw: str) -> list[str]:
    """Comma-separated card names, each optionally suffixed 'xN' (matching
    how hands are shown throughout this project, e.g. 'Copperx3, Estate').
    Each name may also be a short code from CARD_ABBREVIATIONS (e.g. 'POA'
    for Poacher, 'CR' for Council Room) -- see --list-abbreviations. Blank
    input means an empty zone."""
    raw = raw.strip()
    if not raw:
        return []
    cards: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" in token and token.rsplit("x", 1)[-1].isdigit():
            name, count = token.rsplit("x", 1)
            name = name.strip()
        else:
            name, count = token, "1"
        cards.extend([resolve_card_name(name)] * int(count))
    return cards


def format_cards(cards: list[str]) -> str:
    """Inverse of parse_cards, for echoing a default back in 'Copperx3, Estate' form."""
    if not cards:
        return ""
    counts: dict[str, int] = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    return ", ".join(f"{name}x{n}" if n > 1 else name for name, n in counts.items())


def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{msg}{suffix}: ").strip()
    return raw if raw else default


def prompt_cards(msg: str, default: list[str] | None = None) -> list[str]:
    default_str = format_cards(default) if default else ""
    while True:
        try:
            return parse_cards(prompt(msg, default_str))
        except ValueError as e:
            print(f"  {e} -- try again")


def prompt_multiline(msg: str) -> str:
    print(f"{msg} (end with a line containing just END):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def prompt_int(msg: str, default: int) -> int:
    while True:
        raw = prompt(msg, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  not a number -- try again")


def parse_kingdom(raw: str) -> list[str]:
    """Whitespace- *or* comma-separated (no comma required) -- e.g.
    'POA CR CHA MIL MIN MOA MLR VAS VIL TR'. No 'xN' support (a kingdom is
    always exactly one of each), so a multi-word full name typed out
    (rather than its code, e.g. 'Council Room' instead of 'CR') would be
    misread as two separate tokens -- use the short code for those two."""
    tokens = raw.replace(",", " ").split()
    return [resolve_card_name(t) for t in tokens]


def prompt_kingdom() -> list[str]:
    while True:
        try:
            kingdom = parse_kingdom(prompt("Kingdom (10 cards, space-separated codes are fine)"))
        except ValueError as e:
            print(f"  {e} -- try again")
            continue
        if len(kingdom) == 10 and len(set(kingdom)) == 10:
            return kingdom
        print(f"  need exactly 10 distinct kingdom cards, got {len(kingdom)} -- try again")


_FRESH_SUPPLY_DEFAULTS = {"Copper": 46, "Silver": 40, "Gold": 30, "Estate": 8, "Duchy": 8, "Province": 8, "Curse": 10}


def prompt_supply(kingdom: list[str], previous: dict[str, int] | None = None) -> dict[str, int]:
    print("Current supply counts (one number each; defaults are the last value you entered, "
          "or a fresh 2p game's starting counts the first time):")
    defaults = previous or {**_FRESH_SUPPLY_DEFAULTS, **{name: 10 for name in kingdom}}
    supply = {}
    for name in list(_FRESH_SUPPLY_DEFAULTS) + list(kingdom):
        supply[name] = prompt_int(f"  {name}", defaults.get(name, 10))
    return supply


def prompt_table_state(
    kingdom: list[str], supply: dict[str, int],
    default_trash: list[str] | None = None, default_my_total: list[str] | None = None,
    default_my_turns_taken: int = 0,
) -> TableState:
    print("\n--- your side ---")
    my_hand = prompt_cards("Your hand")
    my_discard = prompt_cards("Your discard pile")
    my_play_area = prompt_cards("Your play area (cards played so far this turn, if any)")
    my_total = prompt_cards("EVERY card you currently own, any zone (hand+deck+discard+play area)", default_my_total)
    my_phase = prompt("Phase (ACTION/BUY)", "ACTION").upper()
    my_actions = prompt_int("Your actions remaining", 1 if my_phase == "ACTION" else 0)
    my_buys = prompt_int("Your buys remaining", 1)
    my_coins = prompt_int("Your coins available (treasures already counted)", 0)
    my_turns_taken = prompt_int("Your completed turns before this one (0 on your first turn)", default_my_turns_taken)

    print("\n--- opponent's side (only what's publicly visible) ---")
    opp_discard = prompt_cards("Opponent's discard pile")
    opp_play_area = prompt_cards("Opponent's play area (usually empty between turns)")
    opp_hand_size = prompt_int("Opponent's hand size", 5)
    opp_draw_pile_size = prompt_int("Opponent's draw pile size", 5)

    print("\n--- shared ---")
    trash = prompt_cards("Trash pile", default_trash)

    return TableState(
        kingdom=kingdom, supply=supply, trash=trash,
        my_hand=my_hand, my_discard=my_discard, my_play_area=my_play_area, my_total=my_total,
        my_actions=my_actions, my_buys=my_buys, my_coins=my_coins, my_phase=my_phase, my_turns_taken=my_turns_taken,
        opp_discard=opp_discard, opp_play_area=opp_play_area,
        opp_hand_size=opp_hand_size, opp_draw_pile_size=opp_draw_pile_size,
    )


def recommend(state: TableState, network: torch.nn.Module, simulations: int, device: torch.device) -> None:
    game = reconstruct_game(state)
    root = run_mcts(game, network, simulations, device=device)
    dist = visit_distribution(root)
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])

    print("\nDomibot's read on this decision (share of search spent on each option):")
    for action, share in ranked[:6]:
        print(f"  {share * 100:5.1f}%  {action}")
    best = select_action(root, temperature=0.0)
    print(f"\n==> recommended: {best}\n")


def try_parse_log(kingdom: list[str], my_name: str):
    """Always prompts for a log paste; returns a `log_parser.ParsedLog`, or
    None if it couldn't be parsed (an empty paste included -- just hit
    END immediately to skip and fall back to manual entry for this query)."""
    text = prompt_multiline("Paste a dominion.games log to auto-fill supply/trash/your total")
    if not text.strip():
        return None
    try:
        parsed = parse_dominion_log(text, my_name=my_name, kingdom=kingdom)
    except ValueError as e:
        print(f"  couldn't parse that log: {e} -- falling back to manual entry\n")
        return None
    print(f"\n  derived from the log: trash={format_cards(parsed.trash) or '(empty)'}")
    print(f"  your total ownership: {format_cards(parsed.my_total)}")
    print("  (still shown as editable defaults below -- double check them)\n")
    return parsed


def print_abbreviations() -> None:
    print("Kingdom card abbreviations (case-insensitive, use anywhere a card list is asked for):")
    width = max(len(name) for name in CARD_ABBREVIATIONS.values())
    for code, name in sorted(CARD_ABBREVIATIONS.items(), key=lambda kv: kv[1]):
        print(f"  {name:<{width}}  {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations per query (no self-play "
                                                                       "speed pressure here, so it's fine to go higher than training's default)")
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available")
    parser.add_argument("--list-abbreviations", action="store_true", help="print the kingdom card short codes and exit")
    parser.add_argument("--account-name", type=str, default="domibot_v1.4",
                         help="your account name as it appears in a pasted log -- unrelated to which "
                              "checkpoint is giving advice, just whatever your dominion.games username is")
    args = parser.parse_args()

    if args.list_abbreviations:
        print_abbreviations()
        return

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    device = get_device() if args.gpu else torch.device("cpu")
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    network.eval()
    print(f"Loaded {args.checkpoint} onto {device}, {args.simulations} sims/query.\n")
    print_abbreviations()
    print()

    kingdom = prompt_kingdom()
    my_name = args.account_name
    print(f"Using account name {my_name!r} for log parsing (override with --account-name).\n")
    supply = None
    try:
        while True:
            parsed = try_parse_log(kingdom, my_name)
            if parsed is not None:
                supply = parsed.supply
            elif supply is None or prompt("Update supply counts this query? (y/N)", "n").lower().startswith("y"):
                supply = prompt_supply(kingdom, previous=supply)

            default_turns = parsed.turns_taken.get(my_name, 0) if parsed else 0
            state = prompt_table_state(
                kingdom, supply,
                default_trash=parsed.trash if parsed else None,
                default_my_total=parsed.my_total if parsed else None,
                default_my_turns_taken=default_turns,
            )
            try:
                recommend(state, network, args.simulations, device)
            except ValueError as e:
                print(f"\nInput doesn't add up: {e}\n")
                continue
            if not prompt("Another decision? (Y/n)", "y").lower().startswith("y"):
                break
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
