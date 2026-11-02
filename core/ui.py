"""Drawing toolkit: theme, anti-aliased TTF text, translucent panels, hover buttons, icons, skeletons.

OpenCV's built-in fonts look dated, so text is rendered once with Pillow
(TrueType) and cached as an alpha sprite; drawing cached text costs almost
nothing per frame.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CFG
from .gestures import HAND_CONNECTIONS
from .pose_metrics import POSE_CONNECTIONS


class C:
    """Colour theme (BGR)."""
    BG = (44, 26, 22)
    PANEL = (66, 42, 34)
    PANEL_DARK = (40, 24, 20)
    TEXT = (246, 242, 238)
    MUTED = (200, 188, 178)
    ACCENT = (255, 196, 64)        # sky blue
    TEAL = (200, 220, 60)
    GOOD = (110, 214, 92)          # green
    WARN = (52, 186, 255)          # orange
    BAD = (96, 96, 240)            # red
    GOLD = (48, 206, 255)
    PINK = (190, 120, 255)
    PURPLE = (220, 110, 160)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    EASY = (110, 214, 92)
    MEDIUM = (52, 186, 255)
    HARD = (96, 96, 240)
    LOCK = (120, 110, 104)


DIFF_COLORS = {"EASY": C.EASY, "MEDIUM": C.MEDIUM, "HARD": C.HARD}


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
def _font_candidates(bold: bool):
    win = Path("C:/Windows/Fonts")
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    cands = [win / n for n in names]
    cands += [Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
              Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    try:
        import matplotlib
        mpl = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        cands.append(mpl / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    except Exception:
        pass
    return cands


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False):
    for p in _font_candidates(bold):
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


@lru_cache(maxsize=1500)
def _render(s: str, size: int, color: tuple, bold: bool, stroke: int, stroke_color: tuple):
    f = font(size, bold)
    x0, y0, x1, y1 = f.getbbox(s or " ", stroke_width=stroke)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((-x0, -y0), s, font=f, fill=(color[2], color[1], color[0], 255),
           stroke_width=stroke, stroke_fill=(stroke_color[2], stroke_color[1], stroke_color[0], 255))
    arr = np.asarray(im)
    alpha = (arr[..., 3:4].astype(np.float32) / 255.0)
    premul = arr[..., 2::-1].astype(np.float32) * alpha
    return premul, alpha, -y0


@lru_cache(maxsize=4096)
def text_size(s: str, size: int = 32, bold: bool = False) -> tuple:
    f = font(size, bold)
    x0, y0, x1, y1 = f.getbbox(s or " ")
    return x1 - x0, y1 - y0


def text(img, s: str, pos, size: int = 32, color=C.TEXT, bold: bool = False, anchor: str = "lt",
         stroke: int = 0, stroke_color=(20, 12, 10), alpha: float = 1.0) -> tuple:
    """Draw text. anchor: two chars, horizontal l/m/r + vertical t/m/b."""
    if not s:
        return 0, 0
    bgr, a, _ = _render(s, int(size), tuple(int(c) for c in color), bold, int(stroke), tuple(stroke_color))
    h, w = a.shape[:2]
    x, y = int(pos[0]), int(pos[1])
    if anchor[0] == "m":
        x -= w // 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h // 2
    elif anchor[1] == "b":
        y -= h
    if alpha < 1:
        bgr, a = bgr * alpha, a * alpha
    _blit(img, bgr, a, x, y)
    return w, h


def _blit(img, premul, a, x, y):
    H, W = img.shape[:2]
    h, w = a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - x, y0 - y
    roi = img[y0:y1, x0:x1].astype(np.float32)
    aa = a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    src = premul[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    img[y0:y1, x0:x1] = (roi * (1 - aa) + src).astype(np.uint8)


def wrap(s: str, size: int, max_w: int, bold: bool = False) -> list:
    return list(_wrap(s, size, max_w, bold))


@lru_cache(maxsize=1024)
def _wrap(s: str, size: int, max_w: int, bold: bool = False) -> list:
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_size(trial, size, bold)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return tuple(lines)


def paragraph(img, s: str, pos, max_w: int, size: int = 28, color=C.TEXT, bold=False, line_gap: float = 1.3) -> int:
    y = pos[1]
    for line in wrap(s, size, max_w, bold):
        text(img, line, (pos[0], y), size, color, bold)
        y += int(size * line_gap)
    return y


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def rounded_rect(img, rect, radius: int, color, thickness: int = -1):
    x, y, w, h = [int(v) for v in rect]
    r = int(max(0, min(radius, w // 2, h // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        t = thickness
        cv2.line(img, (x + r, y), (x + w - r, y), color, t, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, t, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, t, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, t, cv2.LINE_AA)
        for (cx, cy), a0 in (((x + r, y + r), 180), ((x + w - r, y + r), 270), ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, t, cv2.LINE_AA)


def panel(img, rect, color=C.PANEL, alpha: float = 0.78, radius: int = 22, border=None, border_t: int = 2):
    x, y, w, h = [int(v) for v in rect]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    rounded_rect(over, (x - x0, y - y0, w, h), radius, color, -1)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, roi)
    if border is not None:
        rounded_rect(img, rect, radius, border, border_t)


def darken(img, amount: float = 0.5, tint=None):
    """Dim the whole frame (and optionally tint it) in place."""
    if tint is None:
        cv2.convertScaleAbs(img, img, alpha=1 - amount, beta=0)
    else:
        cv2.addWeighted(img, 1 - amount, solid(img.shape, tint), amount, 0, img)


@lru_cache(maxsize=16)
def _solid(shape, color):
    layer = np.empty(shape, np.uint8)
    layer[:, :] = color
    return layer


def solid(shape, color) -> np.ndarray:
    """Cached full-frame colour layer (read-only)."""
    return _solid(tuple(shape), tuple(int(c) for c in color))


def vertical_gradient(w: int, h: int, top, bottom) -> np.ndarray:
    g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.array(top, np.float32) * (1 - g) + np.array(bottom, np.float32) * g
    return np.repeat(img, w, axis=1).astype(np.uint8)


def progress_bar(img, rect, frac: float, color=C.ACCENT, bg=(30, 20, 18), radius: int = 12, label: str = "",
                 marker: Optional[float] = None):
    x, y, w, h = [int(v) for v in rect]
    frac = float(np.clip(frac, 0, 1))
    rounded_rect(img, (x, y, w, h), radius, bg, -1)
    if frac > 0.01:
        rounded_rect(img, (x, y, max(h, int(w * frac)), h), radius, color, -1)
    if marker is not None:
        mx = x + int(w * np.clip(marker, 0, 1))
        cv2.line(img, (mx, y - 5), (mx, y + h + 5), C.WHITE, 3, cv2.LINE_AA)
    if label:
        text(img, label, (x + w // 2, y + h // 2), max(14, int(h * 0.7)), C.WHITE, True, "mm", stroke=2)


def ring(img, center, radius: int, frac: float, color, thickness: int = 8, bg=(60, 50, 46)):
    c = (int(center[0]), int(center[1]))
    if bg is not None:
        cv2.circle(img, c, int(radius), bg, thickness, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(img, c, (int(radius), int(radius)), -90, 0, 360 * min(1.0, frac), color, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Glow (additive, blurred only inside a bounding box -> cheap)
# --------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _glow_sprite(radius: int):
    size = radius * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - radius
    d = np.sqrt(xx ** 2 + yy ** 2) / radius
    return np.clip(1 - d, 0, 1) ** 2


@lru_cache(maxsize=256)
def _glow_rgb(radius: int, color: tuple, level: int):
    s = _glow_sprite(radius)[..., None] * (level / 20.0) * np.array(color, np.float32)
    return np.clip(s, 0, 255).astype(np.uint8)


def radial_glow(img, center, radius: int, color, intensity: float = 1.0):
    """Additive soft glow. Sprites are cached by (radius/8, colour, intensity/0.05)."""
    radius = max(8, (int(radius) // 8) * 8)
    level = int(round(min(2.0, max(0.0, intensity)) * 20))
    if level == 0:
        return
    sprite = _glow_rgb(radius, tuple(int(c) for c in color), level)
    cx, cy = int(center[0]), int(center[1])
    H, W = img.shape[:2]
    x0, y0 = cx - radius, cy - radius
    x1, y1 = x0 + sprite.shape[1], y0 + sprite.shape[0]
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    roi = img[iy0:iy1, ix0:ix1]
    cv2.add(roi, sprite[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0], roi)


def glow_segments(img, segments, color, thickness: int = 6, glow: int = 14, core_color=None):
    """Draw line segments with a soft additive glow."""
    if not segments:
        return
    pts = np.array([p for s in segments for p in s], dtype=np.int32)
    H, W = img.shape[:2]
    pad = glow * 2 + thickness
    x0, y0 = max(0, pts[:, 0].min() - pad), max(0, pts[:, 1].min() - pad)
    x1, y1 = min(W, pts[:, 0].max() + pad), min(H, pts[:, 1].max() + pad)
    if x0 < x1 and y0 < y1 and glow > 0:
        layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        for a, b in segments:
            cv2.line(layer, (a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0), color, thickness + glow, cv2.LINE_AA)
        k = glow * 2 + 1
        layer = cv2.GaussianBlur(layer, (k, k), 0)
        cv2.add(img[y0:y1, x0:x1], layer, img[y0:y1, x0:x1])
    cc = core_color or tuple(min(255, int(c * 0.5 + 128)) for c in color)
    for a, b in segments:
        cv2.line(img, tuple(a), tuple(b), cc, thickness, cv2.LINE_AA)


def draw_hand(img, hand, color=C.ACCENT, glow: bool = True, thickness: int = 4):
    H, W = img.shape[:2]
    p = hand.px(W, H)
    segs = [(p[a], p[b]) for a, b in HAND_CONNECTIONS]
    if glow:
        glow_segments(img, segs, color, thickness, 10)
    else:
        for a, b in segs:
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
    for i, q in enumerate(p):
        cv2.circle(img, tuple(q), 6 if i in (4, 8, 12, 16, 20) else 4, C.WHITE, -1, cv2.LINE_AA)


def draw_pose(img, pose_norm, color=C.ACCENT, thickness: int = 8, glow: int = 14, vis=None, head: bool = True,
              core_color=None):
    H, W = img.shape[:2]
    p = np.column_stack([pose_norm[:, 0] * W, pose_norm[:, 1] * H]).astype(np.int32)
    segs = [(p[a], p[b]) for a, b in POSE_CONNECTIONS if vis is None or min(vis[a], vis[b]) > 0.4]
    glow_segments(img, segs, color, thickness, glow, core_color)
    if head:
        r = int(max(18, np.linalg.norm(p[7] - p[8]) * 0.75))
        cv2.circle(img, tuple(p[0] - np.array([0, r // 4])), r, core_color or color, max(3, thickness // 2), cv2.LINE_AA)
    for i in (11, 12, 13, 14, 15, 16):
        if vis is None or vis[i] > 0.4:
            cv2.circle(img, tuple(p[i]), thickness, C.WHITE, -1, cv2.LINE_AA)


def hand_icon(img, gesture: str, center, size: int, color=C.TEXT, side: str = "Right", thickness: int = 3):
    """Small hand diagram for a gesture (drawn from the synthetic hand model)."""
    from .synth import icon_points
    pts = icon_points(gesture, side)
    scale = size / 0.19
    p = (pts * scale + np.array(center)).astype(np.int32)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)
    for i in (4, 8, 12, 16, 20):
        cv2.circle(img, tuple(p[i]), thickness + 1, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Icons (always paired with text: never colour-only feedback)
# --------------------------------------------------------------------------
def star(img, center, r: int, color=C.GOLD, filled: bool = True, thickness: int = 2):
    cx, cy = center
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    pts = np.array(pts, np.int32)
    if filled:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (40, 90, 140) if filled else color, thickness, cv2.LINE_AA)


def lock(img, center, s: int, color=C.LOCK):
    cx, cy = int(center[0]), int(center[1])
    cv2.ellipse(img, (cx, cy - s // 4), (s // 3, s // 3), 0, 180, 360, color, max(3, s // 9), cv2.LINE_AA)
    rounded_rect(img, (cx - s // 2, cy - s // 4, s, int(s * 0.75)), s // 8, color, -1)
    cv2.circle(img, (cx, cy + s // 8), max(2, s // 10), (40, 30, 28), -1, cv2.LINE_AA)


def check(img, center, s: int, color=C.GOOD, thickness: int = 5):
    cx, cy = int(center[0]), int(center[1])
    pts = np.array([(cx - s // 2, cy), (cx - s // 8, cy + s // 3), (cx + s // 2, cy - s // 3)], np.int32)
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def cross(img, center, s: int, color=C.BAD, thickness: int = 5):
    cx, cy, h = int(center[0]), int(center[1]), s // 2
    cv2.line(img, (cx - h, cy - h), (cx + h, cy + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - h, cy + h), (cx + h, cy - h), color, thickness, cv2.LINE_AA)


def warn_icon(img, center, s: int, color=C.WARN):
    cx, cy = int(center[0]), int(center[1])
    pts = np.array([(cx, cy - s // 2), (cx + s // 2, cy + s // 2), (cx - s // 2, cy + s // 2)], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.line(img, (cx, cy - s // 6), (cx, cy + s // 6), (20, 20, 20), max(2, s // 8), cv2.LINE_AA)
    cv2.circle(img, (cx, cy + s // 3), max(2, s // 14), (20, 20, 20), -1, cv2.LINE_AA)


def clock_icon(img, center, r: int, frac: float, color=C.TEXT):
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
    a = -math.pi / 2 + 2 * math.pi * (1 - frac)
    cv2.line(img, (cx, cy), (int(cx + r * 0.7 * math.cos(a)), int(cy + r * 0.7 * math.sin(a))), color, 2, cv2.LINE_AA)


def chip(img, label: str, pos, color, size: int = 20, anchor: str = "lt") -> int:
    tw, th = text_size(label, size, True)
    w, h = tw + 24, int(size * 1.6)
    x, y = int(pos[0]), int(pos[1])
    if anchor[0] == "m":
        x -= w // 2
    rounded_rect(img, (x, y, w, h), h // 2, color, -1)
    text(img, label, (x + w // 2, y + h // 2), size, (25, 20, 18), True, "mm")
    return w


# --------------------------------------------------------------------------
# Hover-to-click button (hold your hand over it)
# --------------------------------------------------------------------------
class HoverButton:
    def __init__(self, rect, label: str, color=C.ACCENT, sub: str = "", circle: bool = False,
                 enabled: bool = True, hold: Optional[float] = None, size: int = 30):
        self.rect = rect            # (x,y,w,h) or (cx,cy,r) when circle
        self.label = label
        self.sub = sub
        self.color = color
        self.circle = circle
        self.enabled = enabled
        self.hold = hold or CFG.HOVER_CLICK_SECONDS
        self.size = size
        self.progress = 0.0
        self.hovered = False
        self.pulse = 0.0

    def contains(self, pt) -> bool:
        if pt is None:
            return False
        x, y = pt
        if self.circle:
            cx, cy, r = self.rect
            return (x - cx) ** 2 + (y - cy) ** 2 <= (r * 1.15) ** 2
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def update(self, cursors, dt: float, click=None) -> bool:
        """cursors: list of (x,y) hand pointers. click: mouse click position or None."""
        self.pulse += dt
        if click is not None and self.contains(click):
            self.progress = 0.0
            return True
        self.hovered = any(self.contains(c) for c in cursors)
        if self.hovered:
            self.progress += dt / self.hold
            if self.progress >= 1.0:
                self.progress = 0.0
                return True
        else:
            self.progress = max(0.0, self.progress - dt * 2.5)
        return False

    @property
    def center(self):
        if self.circle:
            return self.rect[0], self.rect[1]
        x, y, w, h = self.rect
        return x + w // 2, y + h // 2

    def draw(self, img):
        col = self.color if self.enabled else C.LOCK
        if self.circle:
            return  # circular nodes are drawn by their owner; only the ring is generic
        x, y, w, h = self.rect
        grow = 4 if self.hovered else 0
        panel(img, (x - grow, y - grow, w + 2 * grow, h + 2 * grow), C.PANEL_DARK, 0.85, h // 2,
              border=col, border_t=3)
        if self.progress > 0:
            fill_w = int((w + 2 * grow) * self.progress)
            over = img.copy()
            rounded_rect(over, (x - grow, y - grow, max(h, fill_w), h + 2 * grow), h // 2, col, -1)
            cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
        cy = y + h // 2 - (8 if self.sub else 0)
        text(img, self.label, (x + w // 2, cy), self.size, C.TEXT, True, "mm")
        if self.sub:
            text(img, self.sub, (x + w // 2, y + h - 16), 18, C.MUTED, False, "mm")

    def draw_ring(self, img, radius_pad: int = 14):
        if self.progress > 0 and self.circle:
            cx, cy, r = self.rect
            ring(img, (cx, cy), r + radius_pad, self.progress, C.WHITE, 8, None)


def draw_cursor(img, pt, progress: float = 0.0):
    if pt is None:
        return
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), 16, C.WHITE, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 5, C.ACCENT, -1, cv2.LINE_AA)
    if progress > 0:
        ring(img, (x, y), 28, progress, C.GOLD, 6, None)
