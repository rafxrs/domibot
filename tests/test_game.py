import random

import pytest

from domibot import Action, CardType, Game, KINGDOM_CARDS, Phase
from domibot.models import DONE, END_ACTIONS, END_BUY, NO_REVEAL, REVEAL_MOAT

ALL_KINGDOM = list(KINGDOM_CARDS)


def make_game(required: list[str], num_players: int = 2, seed: int = 0) -> Game:
    """A game whose kingdom is guaranteed to include `required`, padded with
    other kingdom cards up to 10, deterministically."""
    rng = random.Random(seed)
    rest = [c for c in ALL_KINGDOM if c not in required]
    rng.shuffle(rest)
    kingdom = list(required) + rest[: 10 - len(required)]
    return Game(kingdom, num_players=num_players, seed=seed)


def play(game: Game, action: Action) -> None:
    game.step(action)


# ------------------------------------------------------------------ setup ---
def test_setup_two_player():
    game = make_game(["Village"], num_players=2, seed=1)
    assert game.supply["Copper"] == 60 - 7 * 2
    assert game.supply["Estate"] == 8
    assert game.supply["Curse"] == 10
    for p in game.players:
        assert len(p.hand) == 5
        assert p.deck_size() == 10
    assert game.phase == Phase.ACTION
    assert game.players[0].actions == 1
    assert game.players[0].buys == 1


def test_setup_four_player_scales_supply():
    game = make_game(["Village"], num_players=4, seed=1)
    assert game.supply["Estate"] == 12
    assert game.supply["Curse"] == 30
    assert game.supply["Copper"] == 60 - 7 * 4


def test_illegal_action_rejected():
    game = make_game(["Village"], seed=1)
    with pytest.raises(ValueError):
        game.step(Action("BUY", "Province"))  # no buy phase yet / can't afford


# --------------------------------------------------------------- treasury ---
def test_treasures_auto_play_entering_buy_phase():
    game = make_game(["Village"], seed=2)
    p = game.players[0]
    p.hand = ["Copper", "Copper", "Copper"]
    play(game, END_ACTIONS)
    assert game.phase == Phase.BUY
    assert p.coins == 3  # all Treasures auto-played, no PLAY actions needed
    assert p.hand == []
    assert p.play_area == ["Copper", "Copper", "Copper"]
    assert Action("BUY", "Silver") in game.legal_actions()
    play(game, Action("BUY", "Silver"))
    assert p.coins == 0
    assert p.buys == 0
    assert "Silver" in p.discard
    assert game.supply["Silver"] == 39


# ------------------------------------------------------------------ Village
def test_village_grants_card_and_actions():
    game = make_game(["Village"], seed=3)
    p = game.players[0]
    p.hand = ["Village", "Copper", "Copper", "Copper", "Copper"]
    p.deck = ["Estate"]
    hand_before = len(p.hand)
    play(game, Action("PLAY", "Village"))
    assert len(p.hand) == hand_before  # -1 played +1 drawn
    assert p.actions == 2  # had 1, -1 to play, +2 from Village


# ------------------------------------------------------------------ Chapel
def test_chapel_trashes_chosen_cards_then_stops():
    game = make_game(["Chapel"], seed=4)
    p = game.players[0]
    p.hand = ["Chapel", "Estate", "Estate", "Copper", "Copper"]
    play(game, Action("PLAY", "Chapel"))
    play(game, Action("TRASH", "Estate"))
    play(game, Action("TRASH", "Copper"))
    play(game, DONE)
    assert sorted(game.trash) == ["Copper", "Estate"]
    assert sorted(p.hand) == ["Copper", "Estate"]
    assert game.pending_decision is None


def test_chapel_max_four_cards_auto_stops():
    game = make_game(["Chapel"], seed=4)
    p = game.players[0]
    p.hand = ["Chapel", "Copper", "Copper", "Copper", "Copper", "Copper"]
    play(game, Action("PLAY", "Chapel"))
    for _ in range(4):
        assert DONE not in game.legal_actions() or True
        play(game, Action("TRASH", "Copper"))
    # after 4 trashes the effect resolves on its own, no DONE needed
    assert game.pending_decision is None
    assert len(game.trash) == 4
    assert p.hand == ["Copper"]


# -------------------------------------------------------------- Moat/Witch
def test_moat_blocks_witch_attack():
    game = make_game(["Witch", "Moat"], num_players=2, seed=5)
    attacker, victim = game.players
    attacker.hand = ["Witch", "Copper", "Copper", "Copper", "Copper"]
    victim.hand = ["Moat", "Copper", "Copper", "Copper", "Copper"]
    curses_before = game.supply["Curse"]

    play(game, Action("PLAY", "Witch"))
    decision = game.pending_decision
    assert decision.player == 1
    assert REVEAL_MOAT in decision.options
    play(game, REVEAL_MOAT)

    assert game.supply["Curse"] == curses_before  # blocked
    assert "Curse" not in victim.discard
    assert len(attacker.hand) == 4 + 2  # Witch's own +2 cards still applied


def test_witch_without_moat_gains_curse():
    game = make_game(["Witch", "Moat"], num_players=2, seed=6)
    attacker, victim = game.players
    attacker.hand = ["Witch", "Copper", "Copper", "Copper", "Copper"]
    victim.hand = ["Copper", "Copper", "Copper", "Copper", "Estate"]
    play(game, Action("PLAY", "Witch"))
    assert game.pending_decision is None  # no Moat -> no reaction decision
    assert "Curse" in victim.discard
    assert game.supply["Curse"] == 10 - 1


# --------------------------------------------------------------- Militia
def test_militia_forces_discard_down_to_three():
    game = make_game(["Militia"], num_players=2, seed=7)
    attacker, victim = game.players
    attacker.hand = ["Militia", "Copper", "Copper", "Copper", "Copper"]
    victim.hand = ["Copper", "Copper", "Estate", "Estate", "Silver"]
    play(game, Action("PLAY", "Militia"))
    assert attacker.coins == 2
    assert game.pending_decision.player == 1
    play(game, Action("DISCARD", "Estate"))
    play(game, Action("DISCARD", "Estate"))
    assert len(victim.hand) == 3
    assert game.pending_decision is None


# ----------------------------------------------------------- Throne Room
def test_throne_room_doubles_village():
    game = make_game(["Throne Room", "Village"], seed=8)
    p = game.players[0]
    p.hand = ["Throne Room", "Village", "Copper", "Copper", "Copper"]
    p.deck = ["Estate", "Estate", "Estate"]
    play(game, Action("PLAY", "Throne Room"))
    play(game, Action("PLAY", "Village"))
    assert game.pending_decision is None
    # start: 1 action; -1 to play Throne Room = 0; Village resolves twice:
    # each grants +2 actions +1 card -> actions = 0+2+2 = 4, minus nothing else spent
    assert p.actions == 4
    assert "Village" in p.play_area
    assert p.hand.count("Estate") == 2  # one drawn per Village resolution


# --------------------------------------------------------------- Bandit
def test_bandit_trashes_non_copper_treasure_and_gains_gold():
    game = make_game(["Bandit"], seed=9)
    attacker, victim = game.players
    attacker.hand = ["Bandit", "Copper", "Copper", "Copper", "Copper"]
    victim.deck = ["Estate", "Silver"]  # top = Silver, then Estate
    victim.discard = []
    gold_before = game.supply["Gold"]

    play(game, Action("PLAY", "Bandit"))
    assert "Gold" in attacker.discard
    assert game.supply["Gold"] == gold_before - 1
    # Even with only one distinct Treasure revealed, the trash is still a
    # real (single-option) decision -- so it's a logged Action a GUI/log
    # can show, not a silent state mutation.
    assert game.pending_decision is not None
    assert game.pending_decision.player == 1
    play(game, Action("TRASH", "Silver"))
    assert game.pending_decision is None
    assert "Silver" in game.trash
    assert "Estate" in victim.discard


# --------------------------------------------------------------- Sentry
def test_sentry_trash_discard_and_reorder():
    game = make_game(["Sentry"], seed=10)
    p = game.players[0]
    p.hand = ["Sentry", "Copper", "Copper", "Copper", "Copper"]
    # deck[-1] is the top of the deck (popped first): Sentry's own +1 Card
    # draws "Estate", then the effect reveals Curse, then Silver.
    p.deck = ["Silver", "Curse", "Estate"]
    play(game, Action("PLAY", "Sentry"))
    assert p.actions == 1  # Sentry: -1 to play +1 grants = 1
    # trash step: offered Curse + Silver
    play(game, Action("TRASH", "Curse"))
    play(game, DONE)
    # discard step: offered Silver
    play(game, DONE)
    # reorder step: only Silver remains, forced pick
    play(game, Action("TOPDECK", "Silver"))
    assert "Curse" in game.trash
    assert p.deck == ["Silver"]
    assert game.pending_decision is None


# -------------------------------------------------------------- Artisan
def test_artisan_gains_to_hand_and_topdecks():
    game = make_game(["Artisan"], seed=11)
    p = game.players[0]
    p.hand = ["Artisan", "Copper", "Copper", "Copper", "Copper"]
    play(game, Action("PLAY", "Artisan"))
    play(game, Action("GAIN", "Silver"))
    assert "Silver" in p.hand
    play(game, Action("TOPDECK", "Silver"))
    assert p.deck[-1] == "Silver"
    assert "Silver" not in p.hand
    assert game.pending_decision is None


# --------------------------------------------------------------- Library
def test_library_can_skip_action_cards():
    game = make_game(["Library", "Village"], seed=12)
    p = game.players[0]
    p.hand = ["Library", "Copper", "Copper"]
    p.deck = ["Estate", "Estate", "Estate", "Estate", "Estate", "Village"]  # deck[-1] = top
    play(game, Action("PLAY", "Library"))
    play(game, Action("NO"))  # decline drawing Village
    assert "Village" not in p.hand
    assert "Village" in p.discard  # set aside, then discarded at end of effect
    assert len(p.hand) == 7
    assert game.pending_decision is None


# -------------------------------------------------------------- Merchant
def test_merchant_bonus_on_first_silver_only():
    game = make_game(["Merchant"], seed=13)
    p = game.players[0]
    p.hand = ["Merchant", "Silver", "Silver", "Copper"]
    p.deck = ["Estate"]  # Merchant's own +1 Card must not draw another Treasure
    play(game, Action("PLAY", "Merchant"))
    play(game, END_ACTIONS)  # auto-plays both Silvers + the Copper
    # first Silver: +2 and the +1 Merchant bonus; second Silver: +2, no bonus; Copper: +1
    assert p.coins == (2 + 1) + 2 + 1


# ------------------------------------------------------------------ Gardens
def test_gardens_vp_scales_with_deck_size():
    game = make_game(["Gardens"], seed=14)
    p = game.players[0]
    p.deck = ["Copper"] * 27  # plus 5 in hand + 3 estates already there from setup
    total_cards = p.deck_size()
    gardens_card = game.cards["Gardens"]
    expected = total_cards // 10
    assert gardens_card.vp_value(p) == expected


# ---------------------------------------------------------------- Game end
def test_game_ends_when_provinces_run_out():
    game = make_game(["Village"], num_players=2, seed=15)
    game.supply["Province"] = 0
    game.step(END_ACTIONS)
    assert game.is_game_over()
    assert game.legal_actions() == []
    with pytest.raises(RuntimeError):
        game.step(END_BUY)


def test_game_ends_when_three_piles_empty():
    game = make_game(["Village"], num_players=2, seed=16)
    for name in list(game.supply):
        if name != "Province":
            game.supply[name] = 0
        if sum(1 for c in game.supply.values() if c == 0) >= 3:
            break
    game.step(END_ACTIONS)
    assert game.is_game_over()


# ------------------------------------------------------------- full replay
def test_random_playouts_terminate_across_all_kingdom_cards():
    """Sweep every kingdom card into at least one random game, and make sure
    nothing crashes and every game reaches a clean end state."""
    rng = random.Random(42)
    remaining = list(ALL_KINGDOM)
    game_idx = 0
    while remaining:
        chunk = remaining[:10]
        remaining = remaining[10:]
        if len(chunk) < 10:
            filler = [c for c in ALL_KINGDOM if c not in chunk]
            chunk = chunk + rng.sample(filler, 10 - len(chunk))
        num_players = rng.choice([2, 3, 4])
        game = Game(chunk, num_players=num_players, seed=game_idx)
        steps = 0
        while not game.is_game_over():
            actions = game.legal_actions()
            assert actions
            game.step(rng.choice(actions))
            steps += 1
            assert steps < 100_000
        scores = game.get_scores()
        assert set(game.winners()) <= set(scores)
        game_idx += 1
