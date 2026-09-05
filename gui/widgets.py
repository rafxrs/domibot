"""Card/button rendering and hit-testing. Everything clickable in the GUI
-- a hand card, a supply pile, a plain "End Actions" button, a decision
option -- is the same `Clickable`: a rect, the domibot.Action it triggers
if clicked, and how to draw it. That uniformity is what lets sub-decisions
(trashing, discarding, reacting, ...) "just be a button that shows up"
without any per-card-type special-casing in the app loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from domibot import Action

from . import colors

CARD_W, CARD_H = 84, 112
BUTTON_H = 44


@dataclass
class Clickable:
    rect: pygame.Rect
    action: Action
    label: str
    color: tuple[int, int, int]
    subtitle: str = ""
    enabled: bool = True

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_card_like(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    color: tuple[int, int, int],
    subtitle: str = "",
    hovered: bool = False,
    dim: bool = False,
) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=10)
    border_width = 3 if hovered else 1
    pygame.draw.rect(surface, colors.BORDER, rect, width=border_width, border_radius=10)

    lines = wrap_text(font, label, rect.width - 10)
    total_h = len(lines) * font.get_linesize()
    y = rect.centery - total_h // 2 - (8 if subtitle else 0)
    for line in lines:
        text_surf = font.render(line, True, colors.TEXT)
        surface.blit(text_surf, text_surf.get_rect(centerx=rect.centerx, top=y))
        y += font.get_linesize()

    if subtitle:
        sub_surf = small_font.render(subtitle, True, colors.TEXT)
        surface.blit(sub_surf, sub_surf.get_rect(centerx=rect.centerx, bottom=rect.bottom - 6))

    if dim:
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill(colors.DIM_OVERLAY)
        surface.blit(overlay, rect.topleft)


def draw_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    hovered: bool = False,
    enabled: bool = True,
) -> None:
    color = colors.BUTTON_HOVER if (hovered and enabled) else colors.BUTTON
    if not enabled:
        color = (150, 150, 150)
    pygame.draw.rect(surface, color, rect, border_radius=8)
    pygame.draw.rect(surface, colors.BORDER, rect, width=2, border_radius=8)
    text_surf = font.render(label, True, colors.TEXT)
    surface.blit(text_surf, text_surf.get_rect(center=rect.center))


def row_positions(n: int, area_left: int, area_width: int, item_w: int, gap: int = 10) -> list[int]:
    """x (left-edge) positions for `n` equal-width items, centered in the
    area, overlapping (fanned) instead of overflowing if they wouldn't fit
    with full gaps."""
    if n <= 0:
        return []
    total = n * item_w + (n - 1) * gap
    if total <= area_width:
        start = area_left + (area_width - total) // 2
        return [start + i * (item_w + gap) for i in range(n)]
    step = max(12, (area_width - item_w) // max(1, n - 1))
    start = area_left
    return [start + i * step for i in range(n)]
