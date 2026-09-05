"""The pygame front-end. Drives the real domibot.Game / training.DomibotAgent
directly -- this module never touches game rules, it only turns
`game.legal_actions()` into clickable regions and `game.step()` calls.

Interaction model (deliberately uniform, see widgets.Clickable):
- Action-phase PLAY options are clickable on the matching card in your hand.
- Buy-phase BUY options are clickable on the matching supply pile.
- END_ACTIONS / END_BUY are a button in the status bar.
- Everything else (a card-effect sub-decision: trashing, discarding,
  gaining, topdecking, a Moat reveal, a yes/no) renders as a row of
  buttons in the decision panel, one per legal option, labeled by the
  card (or verb, for card-less options like DONE/YES/NO) -- "a button
  that shows up," independent of which zone the card conceptually lives
  in, which is what makes this generalize to every card without
  per-card UI logic.
"""
from __future__ import annotations

import pygame

from domibot import Action, Game, Phase
from domibot.models import END_ACTIONS, END_BUY
from training.agents import DomibotAgent

from . import colors
from .widgets import BUTTON_H, CARD_W, CARD_H, Clickable, draw_button, draw_card_like, row_positions

WIDTH, HEIGHT = 1280, 980
FPS = 30

# The supply grid is 3 rows deep (7 basics + 2 rows of 5 kingdom cards), so
# it alone needs 3*CARD_H + 2*16 (row gaps) =~ 368px -- every band below is
# sized off that, rather than a guessed constant, so nothing overlaps.
_SUPPLY_ROWS_HEIGHT = 3 * CARD_H + 2 * 16

TOP_BAR = pygame.Rect(0, 0, WIDTH, 50)
SUPPLY_AREA = pygame.Rect(20, 58, 860, _SUPPLY_ROWS_HEIGHT)
TRASH_AREA = pygame.Rect(900, 58, 140, 110)
DECISION_PANEL = pygame.Rect(20, SUPPLY_AREA.bottom + 10, WIDTH - 40, 46 + CARD_H + 14)
PLAY_AREA = pygame.Rect(20, DECISION_PANEL.bottom + 10, WIDTH - 40, CARD_H + 20)
HAND_AREA = pygame.Rect(20, PLAY_AREA.bottom + 10, WIDTH - 40, CARD_H + 20)
STATUS_BAR = pygame.Rect(0, HAND_AREA.bottom + 10, WIDTH, 60)
END_PHASE_BUTTON = pygame.Rect(WIDTH - 200, STATUS_BAR.top + 10, 170, BUTTON_H)


def _label_for(action: Action) -> str:
    if action.card is not None:
        return action.card
    return {
        "END_ACTIONS": "End Actions",
        "END_BUY": "End Buy",
        "DONE": "Done",
        "YES": "Yes",
        "NO": "No",
        "REVEAL_MOAT": "Reveal Moat",
        "NO_REVEAL": "Don't Reveal",
        "NONE": "None",
    }.get(action.verb, action.verb)


class DominionGUI:
    def __init__(self, game: Game, domibot: DomibotAgent, human_seat: int = 0):
        pygame.init()
        pygame.display.set_caption("Domibot")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 15)
        self.small_font = pygame.font.SysFont("arial", 12)
        self.big_font = pygame.font.SysFont("arial", 22, bold=True)

        self.game = game
        self.domibot = domibot
        self.human_seat = human_seat
        self.bot_seat = 1 - human_seat
        self.clickables: list[Clickable] = []
        self.running = True
        self.game_over_reported = False

    # ------------------------------------------------------------- loop ---
    def run(self) -> None:
        while self.running:
            self._maybe_take_bot_turn()
            self._rebuild_clickables()
            self._handle_events()
            self._draw()
            self.clock.tick(FPS)
        pygame.quit()

    def _maybe_take_bot_turn(self) -> None:
        if self.game.is_game_over() or self.game.current_decider() != self.bot_seat:
            return
        self._draw(thinking=True)
        pygame.event.pump()  # keep the OS from flagging the window as unresponsive
        action = self.domibot.act(self.game)
        self.game.step(action)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.game.is_game_over() or self.game.current_decider() != self.human_seat:
            return
        for c in self.clickables:
            if c.contains(pos):
                self.game.step(c.action)
                return

    # ------------------------------------------------------ clickables ---
    def _rebuild_clickables(self) -> None:
        self.clickables = []
        if self.game.is_game_over() or self.game.current_decider() != self.human_seat:
            return

        legal = set(self.game.legal_actions())
        decision = self.game.pending_decision
        player = self.game.players[self.human_seat]

        if decision is not None:
            self._place_decision_panel(decision.options)
            return

        if self.game.phase == Phase.ACTION:
            self._place_hand(player, legal)
            self.clickables.append(Clickable(END_PHASE_BUTTON, END_ACTIONS, "End Actions", colors.BUTTON))
        else:
            self._place_supply(legal)
            self.clickables.append(Clickable(END_PHASE_BUTTON, END_BUY, "End Buy", colors.BUTTON))

    def _place_hand(self, player, legal: set[Action]) -> None:
        xs = row_positions(len(player.hand), HAND_AREA.left, HAND_AREA.width, CARD_W)
        for x, name in zip(xs, player.hand):
            action = Action("PLAY", name)
            if action in legal:
                rect = pygame.Rect(x, HAND_AREA.top, CARD_W, CARD_H)
                self.clickables.append(Clickable(rect, action, name, colors.card_color(name)))

    def _place_supply(self, legal: set[Action]) -> None:
        for name, rect in self._supply_rects().items():
            action = Action("BUY", name)
            if action in legal:
                self.clickables.append(Clickable(rect, action, name, colors.card_color(name)))

    def _place_decision_panel(self, options: list[Action]) -> None:
        xs = row_positions(len(options), DECISION_PANEL.left + 10, DECISION_PANEL.width - 20, CARD_W)
        y = DECISION_PANEL.top + 46
        for x, action in zip(xs, options):
            rect = pygame.Rect(x, y, CARD_W, CARD_H)
            label = _label_for(action)
            color = colors.card_color(action.card) if action.card else colors.BUTTON
            self.clickables.append(Clickable(rect, action, label, color))

    # ----------------------------------------------------------- layout ---
    def _supply_rects(self) -> dict[str, pygame.Rect]:
        """Basic cards get a top row, the 10 kingdom cards two rows below --
        a compact version of the usual dominion.games supply grid."""
        basics = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
        kingdom = sorted(self.game.kingdom)
        rects: dict[str, pygame.Rect] = {}

        xs = row_positions(len(basics), SUPPLY_AREA.left, SUPPLY_AREA.width, CARD_W)
        for x, name in zip(xs, basics):
            rects[name] = pygame.Rect(x, SUPPLY_AREA.top, CARD_W, CARD_H)

        row1, row2 = kingdom[:5], kingdom[5:]
        y2 = SUPPLY_AREA.top + CARD_H + 16
        xs1 = row_positions(len(row1), SUPPLY_AREA.left, SUPPLY_AREA.width, CARD_W)
        for x, name in zip(xs1, row1):
            rects[name] = pygame.Rect(x, y2, CARD_W, CARD_H)
        xs2 = row_positions(len(row2), SUPPLY_AREA.left, SUPPLY_AREA.width, CARD_W)
        y3 = y2 + CARD_H + 16
        for x, name in zip(xs2, row2):
            rects[name] = pygame.Rect(x, y3, CARD_W, CARD_H)
        return rects

    # ----------------------------------------------------------- drawing ---
    def _draw(self, thinking: bool = False) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self.screen.fill(colors.BACKGROUND)

        self._draw_top_bar()
        self._draw_supply(mouse_pos)
        self._draw_trash()
        self._draw_decision_panel(mouse_pos)
        self._draw_play_area()
        self._draw_hand(mouse_pos)
        self._draw_status_bar(mouse_pos)

        if thinking:
            self._draw_banner("Domibot is thinking...")
        elif self.game.is_game_over():
            self._draw_game_over()

        pygame.display.flip()

    def _draw_top_bar(self) -> None:
        pygame.draw.rect(self.screen, colors.PANEL_BG, TOP_BAR)
        opp = self.game.players[self.bot_seat]
        whose_turn = "your" if self.game.current_player == self.human_seat else "Domibot's"
        text = (
            f"Domibot -- hand: {len(opp.hand)} cards, "
            f"{opp.deck_size() - len(opp.hand)} elsewhere -- "
            f"turn {self.game.current_turn_number}, "
            f"{whose_turn} turn "
            f"({self.game.phase.name})"
        )
        surf = self.font.render(text, True, colors.TEXT_LIGHT)
        self.screen.blit(surf, (16, 18))

    def _draw_supply(self, mouse_pos) -> None:
        clickable_names = {c.action.card for c in self.clickables if c.action.verb == "BUY"}
        for name, rect in self._supply_rects().items():
            count = self.game.supply.get(name, 0)
            hovered = rect.collidepoint(mouse_pos) and name in clickable_names
            draw_card_like(
                self.screen, self.font, self.small_font, rect, name, colors.card_color(name),
                subtitle=f"${self.game.cards[name].cost}  x{count}",
                hovered=hovered, dim=(count == 0 or name not in clickable_names),
            )

    def _draw_trash(self) -> None:
        pygame.draw.rect(self.screen, colors.PANEL_BG, TRASH_AREA, border_radius=8)
        pygame.draw.rect(self.screen, colors.BORDER, TRASH_AREA, width=1, border_radius=8)
        title = self.small_font.render(f"Trash ({len(self.game.trash)})", True, colors.TEXT_LIGHT)
        self.screen.blit(title, (TRASH_AREA.left + 8, TRASH_AREA.top + 6))
        if self.game.trash:
            names = ", ".join(sorted(set(self.game.trash)))
            lines_surf = self.small_font.render(names[:30], True, colors.TEXT_LIGHT)
            self.screen.blit(lines_surf, (TRASH_AREA.left + 8, TRASH_AREA.top + 26))

    def _draw_decision_panel(self, mouse_pos) -> None:
        decision = self.game.pending_decision
        is_human_decision = self.game.current_decider() == self.human_seat and decision is not None
        pygame.draw.rect(self.screen, colors.DECISION_PANEL_BG, DECISION_PANEL, border_radius=8)
        pygame.draw.rect(self.screen, colors.BORDER, DECISION_PANEL, width=1, border_radius=8)

        prompt = decision.prompt if decision is not None else self._default_prompt()
        prompt_surf = self.font.render(prompt, True, colors.TEXT_LIGHT)
        self.screen.blit(prompt_surf, (DECISION_PANEL.left + 10, DECISION_PANEL.top + 8))

        if is_human_decision:
            for c in self.clickables:
                hovered = c.rect.collidepoint(mouse_pos)
                draw_card_like(self.screen, self.font, self.small_font, c.rect, c.label, c.color, hovered=hovered)

    def _default_prompt(self) -> str:
        if self.game.current_decider() != self.human_seat:
            return "Waiting for Domibot..."
        if self.game.phase == Phase.ACTION:
            return "Choose an Action card to play, or end your Action phase."
        return "Buy a card, or end your Buy phase. (Treasures are played for you automatically.)"

    def _draw_play_area(self) -> None:
        pygame.draw.rect(self.screen, colors.PANEL_BG, PLAY_AREA, border_radius=8)
        player = self.game.players[self.human_seat]
        xs = row_positions(len(player.play_area), PLAY_AREA.left + 10, PLAY_AREA.width - 20, CARD_W)
        for x, name in zip(xs, player.play_area):
            rect = pygame.Rect(x, PLAY_AREA.top + 10, CARD_W, CARD_H)
            draw_card_like(self.screen, self.font, self.small_font, rect, name, colors.card_color(name))

    def _draw_hand(self, mouse_pos) -> None:
        pygame.draw.rect(self.screen, colors.PANEL_BG, HAND_AREA, border_radius=8)
        player = self.game.players[self.human_seat]
        clickable_by_rect = {id(c.rect): c for c in self.clickables}
        xs = row_positions(len(player.hand), HAND_AREA.left, HAND_AREA.width, CARD_W)
        for x, name in zip(xs, player.hand):
            rect = pygame.Rect(x, HAND_AREA.top, CARD_W, CARD_H)
            match = next((c for c in self.clickables if c.rect == rect), None)
            hovered = bool(match) and rect.collidepoint(mouse_pos)
            draw_card_like(
                self.screen, self.font, self.small_font, rect, name, colors.card_color(name),
                hovered=hovered, dim=(match is None and self.game.current_decider() == self.human_seat
                                      and self.game.pending_decision is None),
            )

    def _draw_status_bar(self, mouse_pos) -> None:
        pygame.draw.rect(self.screen, colors.PANEL_BG, STATUS_BAR)
        player = self.game.players[self.human_seat]
        text = f"Actions: {player.actions}   Buys: {player.buys}   Coins: {player.coins}"
        surf = self.font.render(text, True, colors.TEXT_LIGHT)
        self.screen.blit(surf, (16, HEIGHT - 42))

        end_click = next((c for c in self.clickables if c.rect == END_PHASE_BUTTON), None)
        if end_click:
            hovered = END_PHASE_BUTTON.collidepoint(mouse_pos)
            draw_button(self.screen, self.font, END_PHASE_BUTTON, end_click.label, hovered=hovered)

    def _draw_banner(self, text: str) -> None:
        surf = self.big_font.render(text, True, colors.TEXT_LIGHT)
        box = surf.get_rect(center=(WIDTH // 2, DECISION_PANEL.top + DECISION_PANEL.height // 2))
        pygame.draw.rect(self.screen, colors.DECISION_PANEL_BG, box.inflate(30, 20), border_radius=8)
        self.screen.blit(surf, box)

    def _draw_game_over(self) -> None:
        scores = self.game.get_scores()
        winners = self.game.winners()
        if winners == [self.human_seat]:
            result = "You win!"
        elif winners == [self.bot_seat]:
            result = "Domibot wins."
        else:
            result = "Tie."
        text = f"Game over -- you: {scores[self.human_seat]}  Domibot: {scores[self.bot_seat]}  {result}"
        self._draw_banner(text)
