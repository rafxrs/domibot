"""A move advisor for real games you play yourself (e.g. on dominion.games); it clicks nothing.

    python examples/domibot_relay.py
    python examples/domibot_relay.py --checkpoint checkpoints/domibot1/domibot_v4.4.pt --simulations 400

At each query, paste the game's whole text log and press Enter on a blank
line. The log is replayed (`training/log_parser.py`) to rebuild what you can
see (`training/relay.py`), and the advisor recommends your move: what to play
or buy, or the choice waiting inside a card (yours, or the opponent's attack).
A choice made in several steps with nothing revealed between them (Chapel's
trashes) is shown as one sequence. If the log can't be followed, whatever it
did derive is offered as editable defaults for manual entry; paste nothing to
enter everything by hand. With nothing playable it ends your Action phase for
you, as dominion.games does. Card lists accept short codes ('POA', 'CR'; see
--list-abbreviations). Ctrl+C quits.

The recommendation comes from MCTS on the network, which in practice picks
the network's own move (see training/README.md).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

from domibot import Action, Phase
from domibot.models import END_ACTIONS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.log_parser import parse_dominion_log  # noqa: E402
from training.mcts import materialize, run_mcts, select_action, visit_distribution  # noqa: E402
from training.network import DomibotNet, get_device  # noqa: E402
from training.relay import (CARD_ABBREVIATIONS, TableState, reconstruct_game,  # noqa: E402
                            reconstruct_opponent_turn_boundary, replay_open_play, resolve_card_name, table_state)

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "domibot2" / "domibot2.4.pt"
_LEADING_COUNT = re.compile(r"^(\d+)x?$", re.IGNORECASE)
_TRAILING_COUNT = re.compile(r"^(.+?)x(\d+)$", re.IGNORECASE)
_FRESH_SUPPLY = {"Copper": 46, "Silver": 40, "Gold": 30, "Estate": 8, "Duchy": 8, "Province": 8, "Curse": 10}


def _try_resolve(token: str) -> str | None:
    try:
        return resolve_card_name(token)
    except ValueError:
        return None


def parse_cards(raw: str) -> list[str]:
    """Names or codes separated by commas or spaces, each with an optional count
    ('3 Copper', '3x Copper', 'Copperx3'). Two-word names are joined back up."""
    tokens = raw.replace(",", " ").split()
    cards: list[str] = []
    i = 0
    while i < len(tokens):
        tok, count = tokens[i], 1
        if m := _LEADING_COUNT.match(tok):
            count = int(m.group(1))
            i += 1
            if i >= len(tokens):
                raise ValueError(f"expected a card name after {tok!r}")
            tok = tokens[i]
        elif (m := _TRAILING_COUNT.match(tok)) and _try_resolve(m.group(1)) is not None:
            cards.extend([resolve_card_name(m.group(1))] * int(m.group(2)))
            i += 1
            continue
        if (resolved := _try_resolve(tok)) is not None:
            i += 1
        elif i + 1 < len(tokens) and (resolved := _try_resolve(f"{tok} {tokens[i + 1]}")) is not None:
            i += 2
        else:
            raise ValueError(f"not a recognized card name or abbreviation: {tok!r}")
        cards.extend([resolved] * count)
    return cards


def format_cards(cards: list[str]) -> str:
    """'Copperx3, Estate': the inverse of `parse_cards`."""
    counts: dict[str, int] = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    return ", ".join(f"{name}x{n}" if n > 1 else name for name, n in counts.items())


def parse_kingdom(raw: str) -> list[str]:
    """Names or codes separated by spaces or commas; type a two-word name as its code."""
    return [resolve_card_name(t) for t in raw.replace(",", " ").split()]


def prompt(msg: str, default: str = "") -> str:
    raw = input(f"{msg}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def prompt_cards(msg: str, default: list[str] | None = None) -> list[str]:
    while True:
        try:
            return parse_cards(prompt(msg, format_cards(default or [])))
        except ValueError as e:
            print(f"  {e} -- try again")


def prompt_int(msg: str, default: int) -> int:
    while True:
        try:
            return int(prompt(msg, str(default)))
        except ValueError:
            print("  not a number -- try again")


def prompt_multiline(msg: str) -> str:
    """Lines until a blank one, skipping blank lines before any content (a Windows
    console paste can start with a spurious empty line)."""
    print(f"{msg} (paste it, then press Enter on its own once more when done):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


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


def prompt_supply(kingdom: list[str], previous: dict[str, int] | None = None) -> dict[str, int]:
    print("Current supply counts (defaults: what you entered last, or a fresh 2-player game):")
    defaults = previous or {**_FRESH_SUPPLY, **{name: 10 for name in kingdom}}
    return {name: prompt_int(f"  {name}", defaults.get(name, 10)) for name in list(_FRESH_SUPPLY) + list(kingdom)}


def prompt_table_state(kingdom: list[str], supply: dict[str, int], d: TableState) -> TableState:
    """Every field, each defaulting to `d`'s value."""
    print("\n--- your side ---")
    my_hand = prompt_cards("Your hand", d.my_hand)
    my_discard = prompt_cards("Your discard pile", d.my_discard)
    my_play_area = prompt_cards("Your play area (cards played so far this turn, if any)", d.my_play_area)
    my_total = prompt_cards("EVERY card you currently own, any zone (hand+deck+discard+play area)", d.my_total)
    my_phase = prompt("Phase (ACTION/BUY)", d.my_phase).upper()
    my_actions = prompt_int("Your actions remaining", d.my_actions)
    my_buys = prompt_int("Your buys remaining", d.my_buys)
    my_coins = prompt_int("Your coins available (treasures already counted)", d.my_coins)
    my_turns_taken = prompt_int("Your completed turns before this one (0 on your first turn)", d.my_turns_taken)
    print("\n--- opponent's side (only what's publicly visible) ---")
    opp_discard = prompt_cards("Opponent's discard pile", d.opp_discard)
    opp_play_area = prompt_cards("Opponent's play area (usually empty between turns)", d.opp_play_area)
    opp_hand_size = prompt_int("Opponent's hand size", d.opp_hand_size)
    opp_draw_pile_size = prompt_int("Opponent's draw pile size", d.opp_draw_pile_size)
    print("\n--- shared ---")
    trash = prompt_cards("Trash pile", d.trash)
    return TableState(
        kingdom=kingdom, supply=supply, trash=trash, my_hand=my_hand, my_discard=my_discard,
        my_play_area=my_play_area, my_total=my_total, my_actions=my_actions, my_buys=my_buys, my_coins=my_coins,
        my_phase=my_phase, my_turns_taken=my_turns_taken, opp_discard=opp_discard, opp_play_area=opp_play_area,
        opp_hand_size=opp_hand_size, opp_draw_pile_size=opp_draw_pile_size, my_deck_top=d.my_deck_top,
        opp_deck_top=d.opp_deck_top, opp_known_hand=d.opp_known_hand, my_merchant_bonus=d.my_merchant_bonus,
        my_silver_played=d.my_silver_played)


def print_recommendation(root) -> None:
    print("\nDomibot's read on this decision (share of search spent on each option):")
    for action, share in sorted(visit_distribution(root).items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {share * 100:5.1f}%  {action}")
    print(f"\n==> recommended: {select_action(root, temperature=0.0)}\n")


def recommend(state: TableState, network: torch.nn.Module, simulations: int, device: torch.device) -> None:
    game = reconstruct_game(state)
    if game.phase == Phase.ACTION and game.legal_actions() == [END_ACTIONS]:  # as dominion.games skips it
        game.step(END_ACTIONS)
        print("(no action cards playable -- auto-ending your action phase)")
    print_recommendation(run_mcts(game, network, simulations, device=device))


def recommend_choice(boundary, path: list[Action], network: torch.nn.Module, simulations: int,
                     device: torch.device) -> None:
    """A choice inside a card's effect, reached by replaying `path` from `boundary`,
    followed by its later steps while nothing new is revealed in between."""
    game = materialize(boundary, path)
    print(f"\n{game.pending_decision.prompt}")
    root = run_mcts(boundary, network, simulations, device=device, path=path)
    print_recommendation(root)
    chain = [select_action(root, temperature=0.0)]
    before = game
    while len(chain) < 12:
        after = materialize(boundary, path + chain)
        revealed = after.rng.getstate() != before.rng.getstate() or \
            len(after.players[0].deck) < len(before.players[0].deck)
        if after.pending_decision is None or after.pending_decision.player != 0 or revealed:
            break
        chain.append(select_action(run_mcts(boundary, network, simulations, device=device, path=path + chain),
                                   temperature=0.0))
        before = after
    if len(chain) > 1:
        print("==> the whole choice: " + ", ".join(str(a) for a in chain) + "\n")


def recommend_pending_reaction(state: TableState, parsed, network: torch.nn.Module, simulations: int,
                               device: torch.device) -> None:
    """Your answer to the opponent's attack, searched from the start of their turn."""
    path = parsed.pending_reaction_path
    boundary = reconstruct_opponent_turn_boundary(state, parsed.pending_reaction_opp_discard, path=path,
                                                  my_deck_top=parsed.pending_reaction_my_deck_top,
                                                  opp_gains=parsed.pending_reaction_opp_gains)
    game = materialize(boundary, path)
    if game.pending_decision is None:
        print("(no reaction needed here -- paste more of the log once the opponent's turn continues)\n")
    elif game.pending_decision.player != 0:
        print("(a decision is pending, but it's not yours -- paste more of the log)\n")
    else:
        recommend_choice(boundary, path, network, simulations, device)


def try_parse_log(kingdom: list[str], my_name: str):
    """A pasted log's `ParsedLog`, or None for an empty paste or one that can't be parsed."""
    text = prompt_multiline("Paste a dominion.games log to auto-fill this decision")
    if not text.strip():
        return None
    try:
        return parse_dominion_log(text, my_name=my_name, kingdom=kingdom)
    except ValueError as e:
        print(f"  couldn't parse that log: {e} -- falling back to manual entry\n")
        return None


def _fully_derived(parsed) -> bool:
    return all(getattr(parsed, f) is not None for f in (
        "my_hand", "my_discard", "my_play_area", "my_phase", "my_actions", "my_buys", "my_coins", "opp_discard",
        "opp_play_area", "opp_hand_size", "opp_draw_pile_size"))


def print_abbreviations() -> None:
    print("Kingdom card abbreviations (case-insensitive, use anywhere a card list is asked for):")
    width = max(len(name) for name in CARD_ABBREVIATIONS.values())
    for code, name in sorted(CARD_ABBREVIATIONS.items(), key=lambda kv: kv[1]):
        print(f"  {name:<{width}}  {code}")


def advise_from_log(parsed, kingdom: list[str], network, simulations: int, device) -> bool:
    """Advise from a fully derived log; False if the state doesn't add up."""
    if parsed.open_play is not None:
        card = parsed.open_play.card
        result = replay_open_play(parsed.open_play, kingdom, parsed.my_hand)
        if result.status == "pending":
            print(f"Your {card} is waiting on a choice:")
            recommend_choice(result.boundary, result.path, network, simulations, device)
            return True
        if result.status == "opponent":
            print(f"(your {card} is waiting on the opponent -- paste the log again once it's your move)\n")
            return True
        if result.status == "unsupported":
            print(f"(couldn't tell whether your {card} is still waiting on a choice: {result.reason} "
                  f"-- the recommendation below assumes it has finished)")
    print("Everything needed was fully derived from the log -- here's the recommendation:")
    try:
        recommend(table_state(parsed, kingdom), network, simulations, device)
        return True
    except ValueError as e:
        print(f"\nThe log-derived state doesn't add up: {e}\n"
              f"  (falling back to manual entry, pre-filled from the log -- fix whatever's off)\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations per query")
    parser.add_argument("--gpu", action="store_true", help="use CUDA if available")
    parser.add_argument("--list-abbreviations", action="store_true", help="print the card short codes and exit")
    parser.add_argument("--account-name", type=str, default="domibot_v1.4",
                        help="your dominion.games name, as it appears in a pasted log")
    args = parser.parse_args()
    if args.list_abbreviations:
        print_abbreviations()
        return
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} -- download domibot2.4.pt from "
                         f"https://github.com/rafxrs/domibot/releases into checkpoints/domibot2/ "
                         f"(see README Setup), or train your own")

    device = get_device() if args.gpu else torch.device("cpu")
    network = DomibotNet.load(args.checkpoint, map_location=device).to(device)
    network.eval()
    print(f"Loaded {args.checkpoint} onto {device}, {args.simulations} sims/query.\n")
    print_abbreviations()
    print()
    kingdom = prompt_kingdom()
    print(f"Using account name {args.account_name!r} for log parsing (override with --account-name).\n")
    supply = None
    try:
        while True:
            parsed = try_parse_log(kingdom, args.account_name)
            if parsed is not None:
                supply = parsed.supply
            if parsed is not None and parsed.pending_reaction_path is not None:
                print("The opponent's turn is still open and a reaction may be pending on you:")
                try:
                    recommend_pending_reaction(table_state(parsed, kingdom), parsed, network, args.simulations,
                                               device)
                except ValueError as e:
                    print(f"\nInput doesn't add up: {e}\n")
                continue
            if parsed is not None and _fully_derived(parsed):
                if advise_from_log(parsed, kingdom, network, args.simulations, device):
                    continue
            elif parsed is not None:
                print(f"  derived from the log: trash={format_cards(parsed.trash) or '(empty)'}, "
                      f"your total={format_cards(parsed.my_total)}")
                if parsed.my_hand is not None:
                    print(f"  your hand: {format_cards(parsed.my_hand)}")
                print("  (the rest still needs manual entry -- shown as editable defaults below)\n")
            elif supply is None or prompt("Update supply counts this query? (y/N)", "n").lower().startswith("y"):
                supply = prompt_supply(kingdom, previous=supply)
            state = prompt_table_state(kingdom, supply, table_state(parsed, kingdom, supply))
            try:
                recommend(state, network, args.simulations, device)
            except ValueError as e:
                print(f"\nInput doesn't add up: {e}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
