"""A move-advisor for playing against real people: you play a real game
yourself (e.g. on dominion.games), and at each of your phase-action
decisions (what to play, what to buy) this tells you what Domibot would do,
using only what's actually visible to a player at the table.

    python examples/domibot_relay.py
    python examples/domibot_relay.py --checkpoint checkpoints/domibot_v4.4.pt --simulations 400

This never touches the real game for you -- you type in what's on screen,
it prints a recommendation, you click the move yourself. See training/relay.py
for exactly what information it needs and why (short version: your own hand
and total card ownership exactly; for the opponent, only what Dominion
actually makes public -- their discard pile, and their hand/draw-pile
*sizes*, never contents).

After each recommendation it loops straight back to asking for the next
one -- there's no "another decision?" prompt to answer. Ctrl+C is how you
stop (not Escape: input() is line-buffered, so a bare Escape keypress
doesn't submit anything and just sits there until you press Enter).

State is re-entered in full at every decision rather than tracked
incrementally turn to turn -- more typing per query, but far less risk of
this tool's internal state silently drifting from the real game's if a
public event gets missed or mis-entered. Mainly covers your own
phase-action decisions (what to play/buy) -- state is normally
reconstructed at a phase-action boundary, so a forced sub-decision (a
trash/discard/topdeck choice) can't usually be represented here. The one
exception: if you paste a log ending with the opponent having just played
Militia and your forced discard not yet shown, this recognizes that a
reaction is pending on you and recommends the discard directly, instead of
a nonsense phase-action suggestion. Every other sub-decision (Bureaucrat/
Bandit's forced reactions, Moat's reveal-or-not against them, or any of
your own mid-turn choices like an unresolved Chapel trash) still falls
back to the same "keep the good stuff, give up junk" rule (see
training/heuristics.py); Domibot itself searches every sub-decision by
default when playing directly (see training/mcts.py), but this relay tool
only reaches that path for Militia's discard so far.

At each query you paste in that game's dominion.games text log (the whole
thing, fresh, every time -- it keeps growing) and press Enter once more on
the blank line when done. That alone derives the supply, the trash, your
own total card ownership, and (via a full turn-by-turn replay) your hand,
discard, play area, phase, actions, buys, coins, and the opponent's hand
size, draw-pile size, and discard pile -- covering plays, buys/gains,
trashes/discards/topdecks (including Sentry/Bandit-style reveals and
Harbinger's discard-sourced topdeck), Throne-Room-style replays, and
explicit "+N Action/Buy/$" lines. When all of that lines up, it skips
straight to the recommendation -- no confirmation step. It's still
best-effort: dominion.games occasionally renders a card as a bare, unnamed
"a card" (e.g. some Cellar-style discards), which makes exact replay
impossible from that point on -- when that happens it falls back to
showing whatever it *did* derive as editable defaults for manual entry,
same as before. See training/log_parser.py for the full picture. Paste
nothing (blank line right away) to skip it for one query entirely.

Any card list (the kingdom included -- dominion.games' log never states it,
so it's typed by hand every game) accepts a short code instead of the full
name, e.g. 'POA' for Poacher or 'CR' for Council Room -- run with
--list-abbreviations to see the full table (also printed at startup). The
kingdom itself also doesn't need commas between entries.

Your account name defaults to 'domibot_v1.4' (override with
--account-name) -- it's just whatever your dominion.games username is,
unrelated to which checkpoint --checkpoint points at. If the pasted log
has no "name: rating" header at all (e.g. a trimmed practice-game log),
it defaults to you vs. "Lord Rattington", dominion.games' own built-in
bot, rather than refusing to parse it.

If your hand has nothing playable, dominion.games itself auto-skips
straight to the Buy phase (treasures auto-played) rather than making you
click "end actions" -- this does the same, silently, so the recommendation
you get is always the useful one (what to buy), not "end your action
phase" (something you'd never actually see asked on the real site).

Defaults to the strongest checkpoint in the project, `domibot2.1.pt`
(domibot 2's PPO lineage -- see training/README.md). This still works
via MCTS search (`training/mcts.py`'s `run_mcts`) exactly as it did for
the MCTS lineage's checkpoints, even though domibot 2 itself never
searches during training: the network's `(obs) -> (policy_logits,
value)` interface never changed, so search-at-inference-time on top of a
PPO-trained network is just an unrelated, orthogonal choice -- and search
should still help here the same way it always has, refining the policy's
own recommendation with lookahead instead of taking it greedily.
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
sys.path.insert(0, str(ROOT))  # training/ is a sibling of examples/, not on sys.path by default
from training.log_parser import parse_dominion_log  # noqa: E402
from training.mcts import materialize, run_mcts, select_action, visit_distribution  # noqa: E402
from training.network import DomibotNet, get_device  # noqa: E402
from training.relay import (  # noqa: E402
    CARD_ABBREVIATIONS,
    TableState,
    reconstruct_game,
    reconstruct_opponent_turn_boundary,
    resolve_card_name,
)

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "domibot2" / "domibot2.1.pt"


_LEADING_COUNT = re.compile(r"^(\d+)x?$", re.IGNORECASE)
_TRAILING_COUNT = re.compile(r"^(.+?)x(\d+)$", re.IGNORECASE)


def _try_resolve(token: str) -> str | None:
    try:
        return resolve_card_name(token)
    except ValueError:
        return None


def parse_cards(raw: str) -> list[str]:
    """Card names/codes, comma- *or* space-separated (or both), each
    optionally paired with a count either before ('3 Copper', '3x Copper')
    or after ('Copperx3', matching how hands are shown throughout this
    project) -- no count means one. Card names may be a short code from
    CARD_ABBREVIATIONS (e.g. 'POA' for Poacher, 'CR' for Council Room) --
    see --list-abbreviations. A two-word full name (only 'Council Room'
    and 'Throne Room') is recovered by combining adjacent tokens when the
    first alone doesn't resolve. Blank input means an empty zone."""
    tokens = raw.replace(",", " ").split()
    cards: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        count = 1
        m = _LEADING_COUNT.match(tok)
        if m:
            count = int(m.group(1))
            i += 1
            if i >= len(tokens):
                raise ValueError(f"expected a card name after {tok!r}")
            tok = tokens[i]
        else:
            m2 = _TRAILING_COUNT.match(tok)
            if m2 and _try_resolve(m2.group(1)) is not None:
                cards.extend([resolve_card_name(m2.group(1))] * int(m2.group(2)))
                i += 1
                continue

        resolved = _try_resolve(tok)
        if resolved is not None:
            i += 1
        elif i + 1 < len(tokens) and _try_resolve(f"{tok} {tokens[i + 1]}") is not None:
            resolved = _try_resolve(f"{tok} {tokens[i + 1]}")
            i += 2
        else:
            raise ValueError(f"not a recognized card name or abbreviation: {tok!r}")
        cards.extend([resolved] * count)
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
    """A blank line ends the paste -- but a large paste into a Windows
    console commonly delivers a spurious empty first line before the real
    content arrives (no bracketed-paste support), so a blank line only
    counts as the terminator once real content has actually started;
    leading blank lines are just skipped."""
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
    default_my_turns_taken: int = 0, default_my_hand: list[str] | None = None,
    default_my_phase: str | None = None, default_my_actions: int | None = None,
    default_my_buys: int | None = None, default_my_coins: int | None = None,
    default_my_play_area: list[str] | None = None,
) -> TableState:
    print("\n--- your side ---")
    my_hand = prompt_cards("Your hand", default_my_hand)
    my_discard = prompt_cards("Your discard pile")
    my_play_area = prompt_cards("Your play area (cards played so far this turn, if any)", default_my_play_area)
    my_total = prompt_cards("EVERY card you currently own, any zone (hand+deck+discard+play area)", default_my_total)
    my_phase = prompt("Phase (ACTION/BUY)", default_my_phase or "ACTION").upper()
    my_actions = prompt_int("Your actions remaining", default_my_actions if default_my_actions is not None else (1 if my_phase == "ACTION" else 0))
    my_buys = prompt_int("Your buys remaining", default_my_buys if default_my_buys is not None else 1)
    my_coins = prompt_int("Your coins available (treasures already counted)", default_my_coins if default_my_coins is not None else 0)
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


def print_recommendation(root) -> None:
    dist = visit_distribution(root)
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])

    print("\nDomibot's read on this decision (share of search spent on each option):")
    for action, share in ranked[:6]:
        print(f"  {share * 100:5.1f}%  {action}")
    best = select_action(root, temperature=0.0)
    print(f"\n==> recommended: {best}\n")


def recommend(state: TableState, network: torch.nn.Module, simulations: int, device: torch.device) -> None:
    game = reconstruct_game(state)
    # dominion.games itself auto-skips straight to the Buy phase (treasures
    # auto-played) whenever nothing in the Action phase is actually
    # playable -- matching that here means the recommendation is always
    # "what to buy" in that situation, not the trivial "end your action
    # phase" you'd never see asked on the real site. Uses the engine's own
    # step(), not a reimplementation, so treasure coin totals etc. are
    # exactly what the real game would produce.
    if game.phase == Phase.ACTION and game.legal_actions() == [END_ACTIONS]:
        game.step(END_ACTIONS)
        print("(no action cards playable -- auto-ending your action phase)")
    root = run_mcts(game, network, simulations, device=device)
    print_recommendation(root)


def recommend_pending_reaction(
    state: TableState, opp_turn_start_discard: list[str], path: list[Action], network: torch.nn.Module,
    simulations: int, device: torch.device,
) -> None:
    """Militia's forced discard only, for now (see log_parser's
    _SUPPORTED_TERMINAL_ATTACKS) -- reuses the same replay-based search
    training.mcts already does for self-play/DomibotAgent sub-decisions,
    just from a boundary at the *opponent's* turn start instead of yours."""
    boundary = reconstruct_opponent_turn_boundary(state, opp_turn_start_discard, path=path)
    game = materialize(boundary, path)
    if game.pending_decision is None:
        print("(no reaction needed here -- paste more of the log once the opponent's turn continues)\n")
        return
    if game.pending_decision.player != 0:
        print("(a decision is pending, but it's not yours -- paste more of the log)\n")
        return
    print(f"\n{game.pending_decision.prompt}")
    root = run_mcts(boundary, network, simulations, device=device, path=path)
    print_recommendation(root)


def try_parse_log(kingdom: list[str], my_name: str):
    """Always prompts for a log paste; returns a `log_parser.ParsedLog`, or
    None if it couldn't be parsed (an empty paste included -- just press
    Enter on the blank line right away to skip and fall back to manual
    entry for this query)."""
    text = prompt_multiline("Paste a dominion.games log to auto-fill this decision")
    if not text.strip():
        return None
    try:
        parsed = parse_dominion_log(text, my_name=my_name, kingdom=kingdom)
    except ValueError as e:
        print(f"  couldn't parse that log: {e} -- falling back to manual entry\n")
        return None
    return parsed


def _fully_derived(parsed) -> bool:
    return all(x is not None for x in (
        parsed.my_hand, parsed.my_discard, parsed.my_play_area, parsed.my_phase,
        parsed.my_actions, parsed.my_buys, parsed.my_coins,
        parsed.opp_discard, parsed.opp_play_area, parsed.opp_hand_size, parsed.opp_draw_pile_size,
    ))


def state_from_parsed(kingdom: list[str], parsed, my_name: str) -> TableState:
    return TableState(
        kingdom=kingdom, supply=parsed.supply, trash=parsed.trash,
        my_hand=parsed.my_hand, my_discard=parsed.my_discard, my_play_area=parsed.my_play_area,
        my_total=parsed.my_total, my_actions=parsed.my_actions, my_buys=parsed.my_buys,
        my_coins=parsed.my_coins, my_phase=parsed.my_phase,
        my_turns_taken=parsed.turns_taken.get(my_name, 0),
        opp_discard=parsed.opp_discard, opp_play_area=parsed.opp_play_area,
        opp_hand_size=parsed.opp_hand_size, opp_draw_pile_size=parsed.opp_draw_pile_size,
    )


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
        raise SystemExit(f"no checkpoint at {args.checkpoint} -- download domibot2.1.pt from https://github.com/rafxrs/domibot/releases into checkpoints/domibot2/ (see README Setup), or train your own")

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

            if parsed is not None and parsed.pending_reaction_path is not None:
                print("The opponent's turn is still open and a reaction is pending on you:")
                state = state_from_parsed(kingdom, parsed, my_name)
                try:
                    recommend_pending_reaction(
                        state, parsed.pending_reaction_opp_discard, parsed.pending_reaction_path,
                        network, args.simulations, device,
                    )
                except ValueError as e:
                    print(f"\nInput doesn't add up: {e}\n")
                continue

            if parsed is not None and _fully_derived(parsed):
                print("Everything needed was fully derived from the log -- here's the recommendation:")
                state = state_from_parsed(kingdom, parsed, my_name)
            else:
                if parsed is not None:
                    print(f"  derived from the log: trash={format_cards(parsed.trash) or '(empty)'}, "
                          f"your total={format_cards(parsed.my_total)}")
                    if parsed.my_hand is not None:
                        print(f"  your hand: {format_cards(parsed.my_hand)}")
                    print("  (the rest still needs manual entry -- shown as editable defaults below)\n")
                elif supply is None or prompt("Update supply counts this query? (y/N)", "n").lower().startswith("y"):
                    supply = prompt_supply(kingdom, previous=supply)

                default_turns = parsed.turns_taken.get(my_name, 0) if parsed else 0
                state = prompt_table_state(
                    kingdom, supply,
                    default_trash=parsed.trash if parsed else None,
                    default_my_total=parsed.my_total if parsed else None,
                    default_my_turns_taken=default_turns,
                    default_my_hand=parsed.my_hand if parsed else None,
                    default_my_phase=parsed.my_phase if parsed else None,
                    default_my_actions=parsed.my_actions if parsed else None,
                    default_my_buys=parsed.my_buys if parsed else None,
                    default_my_coins=parsed.my_coins if parsed else None,
                    default_my_play_area=parsed.my_play_area if parsed else None,
                )
            try:
                recommend(state, network, args.simulations, device)
            except ValueError as e:
                print(f"\nInput doesn't add up: {e}\n")
            # Always loops straight back to pasting the next decision --
            # Ctrl+C is how you stop (a bare Escape does nothing here;
            # input() is line-buffered and won't submit until Enter).
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
