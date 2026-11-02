"""BaseGame: everything the five games share.

Life-cycle:  intro()  ->  countdown 3-2-1-GO  ->  play (update/draw each frame)
             ->  ending banner  ->  result()

Shared features: timer that pauses while tracking is lost, score with combo
multiplier and floating text, a consistent HUD, star rating, adaptive easing,
and stats collection for the session log.
"""
from __future__ import annotations

import math

import cv2

from core import ui
from core.effects import clamp, ease_out_back
from core.progress import GameResult
from core.ui import C, DIFF_COLORS


class BaseGame:
    id = 0
    name = "Game"
    condition = ""
    helps = ""                      # one-line "Helps with ..."
    body_part = ""
    difficulty = "EASY"
    needs = frozenset({"hands"})
    color = C.ACCENT
    instructions = ["", "", ""]
    pass_text = ""
    gate_text = "Show your hand to start"
    lost_text = "I can't see your hand - please move into view"
    fail_tip = "Take your time - slow and steady movements work best."
    time_limit = 90.0
    countdown_seconds = 3.0
    has_hand_choice = False

    def __init__(self, app, ease: int = 0, settings: dict | None = None):
        self.app = app
        self.cfg = app.cfg
        self.audio = app.audio
        self.fx = app.fx
        self.W, self.H = app.W, app.H
        self.ease = ease
        self.settings = settings or {}
        self.calib = app.progress.calibration or {}
        self.affected = self.calib.get("affected_side", "both")
        self.phase = "countdown"
        self.count = self.countdown_seconds
        self._last_tick = None
        self.elapsed = 0.0
        self.t = 0.0
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.lost_time = 0.0
        self.tracking_lost = False
        self.success = False
        self.end_t = 0.0
        self.reaction_times: list = []
        self.banner = ""
        self.banner_t = 0.0
        self.message = ""
        self.message_t = 0.0
        self.on_start()

    # ------------------------------------------------------------------ helpers
    @property
    def ease_factor(self) -> float:
        """1.0 normally; 0.9 / 0.8 after repeated fails (multiply thresholds by this)."""
        return 1.0 - self.cfg.EASE_STEP * self.ease

    def intro(self) -> dict:
        return {"title": self.name, "condition": self.condition, "helps": self.helps,
                "body_part": self.body_part, "difficulty": self.difficulty,
                "instructions": self.instructions, "pass_text": self.pass_text,
                "gate_text": self.gate_text,
                "hint": (self.fail_tip if self.ease else "")}

    @property
    def time_left(self) -> float:
        return max(0.0, self.time_limit - self.elapsed)

    @property
    def done(self) -> bool:
        return self.phase == "ending" and self.end_t > 2.4

    def add_points(self, base: int, x: float, y: float, color=C.GOLD) -> int:
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        mult = 1.0 + 0.25 * min(self.combo - 1, 4)
        pts = int(round(base * mult))
        self.score += pts
        self.fx.float_text(f"+{pts}", x, y - 30, color, 42)
        if self.combo >= 3:
            self.fx.float_text(f"COMBO x{mult:.2f}", x, y + 20, C.PINK, 28)
        return pts

    def break_combo(self):
        self.combo = 0

    def say(self, msg: str, seconds: float = 2.5):
        self.message, self.message_t = msg, seconds

    def big_banner(self, msg: str, seconds: float = 1.2):
        self.banner, self.banner_t = msg, seconds

    def tracking_ok(self, tracking) -> bool:
        return bool(tracking.hands)

    def finish(self, success: bool):
        if self.phase == "ending":
            return
        self.phase = "ending"
        self.success = success
        self.end_t = 0.0
        if success:
            self.audio.play("fanfare")
            self.fx.confetti(self.W, 120)
            self.fx.shake(8, 0.35)
        else:
            self.audio.play("fail")

    # ------------------------------------------------------------------ loop
    def update(self, frame, tracking, dt: float):
        self.t += dt
        self.message_t = max(0.0, self.message_t - dt)
        self.banner_t = max(0.0, self.banner_t - dt)
        if self.phase == "countdown":
            self.count -= dt
            tick = math.ceil(self.count)
            if tick != self._last_tick and tick > 0:
                self.audio.play("tick")
                self._last_tick = tick
            self.on_countdown(tracking, dt)
            if self.count <= 0:
                self.phase = "play"
                self.audio.play("go")
        elif self.phase == "play":
            ok = self.tracking_ok(tracking)
            self.lost_time = 0.0 if ok else self.lost_time + dt
            self.tracking_lost = self.lost_time > self.cfg.TRACKING_LOST_GRACE
            if not self.tracking_lost:
                self.elapsed += dt
            if ok:                          # game logic only ever sees frames with the body part visible
                self.on_update(tracking, dt)
            if self.phase == "play":
                if self.is_passed():
                    self.finish(True)
                elif self.elapsed >= self.time_limit or self.is_failed():
                    self.finish(False)
        else:
            self.end_t += dt
            self.on_ending(tracking, dt)

    def draw(self, img):
        self.draw_scene(img)
        self.draw_hud(img)
        if self.phase == "countdown":
            self._draw_countdown(img)
        if self.phase == "play" and self.tracking_lost:
            self._draw_lost(img)
        if self.message_t > 0 and self.phase == "play":
            a = min(1.0, self.message_t * 2)
            tw, _ = ui.text_size(self.message, 30, True)
            ui.panel(img, (self.W // 2 - tw // 2 - 30, 112, tw + 60, 56), C.PANEL_DARK, 0.85 * a, 28, border=C.WARN)
            ui.text(img, self.message, (self.W // 2, 140), 30, C.TEXT, True, "mm", alpha=a)
        if self.banner_t > 0:
            k = ease_out_back(clamp((1.2 - self.banner_t) * 4))
            ui.text(img, self.banner, (self.W // 2, self.H // 2 - 40), int(80 * k) + 1, C.GOLD, True, "mm", stroke=5)
        if self.phase == "ending":
            self._draw_ending(img)

    def _draw_countdown(self, img):
        n = math.ceil(self.count)
        frac = n - self.count
        k = ease_out_back(clamp(frac * 2.5))
        ui.panel(img, (self.W // 2 - 330, self.H // 2 - 170, 660, 330), C.PANEL_DARK, 0.6, 40)
        ui.text(img, "Get ready!", (self.W // 2, self.H // 2 - 120), 40, C.MUTED, True, "mm")
        ui.text(img, str(max(1, n)), (self.W // 2, self.H // 2 + 10), int(150 * k) + 1, C.GOLD, True, "mm", stroke=6)
        ui.text(img, self.countdown_tip(), (self.W // 2, self.H // 2 + 125), 28, C.TEXT, False, "mm")

    def countdown_tip(self) -> str:
        return self.instructions[0]

    def _draw_lost(self, img):
        ui.darken(img, 0.35)
        w = 900
        ui.panel(img, (self.W // 2 - w // 2, self.H // 2 - 90, w, 180), C.PANEL_DARK, 0.92, 30, border=C.WARN, border_t=4)
        ui.warn_icon(img, (self.W // 2 - w // 2 + 70, self.H // 2), 70)
        ui.text(img, self.lost_text, (self.W // 2 + 40, self.H // 2 - 22), 30, C.TEXT, True, "mm")
        ui.text(img, "Timer paused  -  take your time", (self.W // 2 + 40, self.H // 2 + 30), 26, C.MUTED, False, "mm")

    def _draw_ending(self, img):
        k = ease_out_back(clamp(self.end_t * 2.5))
        msg = "WELL DONE!" if self.success else ("TIME'S UP" if self.elapsed >= self.time_limit else "GOOD TRY!")
        col = C.GOOD if self.success else C.WARN
        ui.text(img, msg, (self.W // 2, self.H // 2), int(110 * k) + 1, col, True, "mm", stroke=6)

    def draw_hud(self, img):
        W = self.W
        ui.panel(img, (16, 12, W - 32, 88), C.PANEL_DARK, 0.72, 24)
        ui.text(img, self.name, (40, 24), 30, C.TEXT, True)
        ui.chip(img, self.difficulty, (40, 62), DIFF_COLORS[self.difficulty], 16)
        label, cur, target = self.progress_info()
        bx, bw = 400, 420
        ui.text(img, f"{label}  {cur} / {target}", (bx + bw // 2, 34), 26, C.TEXT, True, "mm")
        ui.progress_bar(img, (bx, 58, bw, 22), cur / max(1, target), self.color)
        # timer
        frac = self.time_left / self.time_limit
        tc = C.TEXT if frac > 0.25 else C.WARN
        ui.clock_icon(img, (W - 300, 56), 20, frac, tc)
        ui.text(img, f"{int(math.ceil(self.time_left))}s", (W - 270, 56), 34, tc, True, "lm")
        ui.text(img, f"{self.score}", (W - 44, 40), 34, C.GOLD, True, "rm")
        ui.text(img, f"combo x{self.combo}" if self.combo > 1 else "score", (W - 44, 76), 20, C.MUTED, False, "rm")
        instr = self.current_instruction()
        if instr and self.phase == "play" and not self.tracking_lost:
            tw, _ = ui.text_size(instr, 32, True)
            ui.panel(img, (W // 2 - tw // 2 - 36, self.H - 86, tw + 72, 64), C.PANEL_DARK, 0.8, 32, border=self.color)
            ui.text(img, instr, (W // 2, self.H - 54), 32, C.TEXT, True, "mm")

    # ------------------------------------------------------------------ result
    def stars(self) -> int:
        if not self.success:
            return 0
        used = self.elapsed / self.time_limit
        acc = self.accuracy()
        return 1 + int(used < 0.7) + int(used < 0.45 and acc >= 0.8)

    def result(self) -> GameResult:
        rt = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0.0
        rom, rom_label = self.rom()
        return GameResult(
            game_id=self.id, name=self.name, difficulty=self.difficulty, score=int(self.score),
            accuracy=float(self.accuracy()), passed=self.success, stars=self.stars(),
            duration=round(self.elapsed, 1), reaction_time=rt, rom_value=float(rom), rom_label=rom_label,
            extra=self.extra(), summary=self.summary(), hint="" if self.success else self.fail_tip,
            ease=self.ease)

    # ------------------------------------------------------------------ overridables
    def on_start(self): ...
    def on_countdown(self, tracking, dt): ...
    def on_update(self, tracking, dt): ...
    def on_ending(self, tracking, dt): ...
    def draw_scene(self, img): ...
    def draw_demo(self, img, rect, t: float): ...
    def demo_intent(self) -> dict: return {}
    def current_instruction(self) -> str: return ""
    def progress_info(self) -> tuple: return ("Progress", 0, 1)
    def is_passed(self) -> bool: return False
    def is_failed(self) -> bool: return False
    def accuracy(self) -> float: return 0.0
    def rom(self) -> tuple: return 0.0, ""
    def extra(self) -> dict: return {}
    def summary(self) -> list: return []


def demo_frame(img, rect, title: str = "Watch how to do it"):
    """Common frame for tutorial animations."""
    x, y, w, h = rect
    ui.panel(img, rect, (30, 20, 18), 0.9, 24, border=C.ACCENT)
    ui.text(img, title, (x + w // 2, y + 26), 22, C.MUTED, True, "mm")
    sub = img[y:y + h, x:x + w]
    return sub


def put_hand_in_rect(img, hand_norm_xy, rect, color=C.ACCENT, thickness=4):
    """Draw normalised hand points (0..1 in the rect) as a glowing skeleton."""
    from core.gestures import HAND_CONNECTIONS
    x, y, w, h = rect
    p = [(int(x + px * w), int(y + py * h)) for px, py in hand_norm_xy]
    segs = [(p[a], p[b]) for a, b in HAND_CONNECTIONS]
    ui.glow_segments(img, segs, color, thickness, 8)
    for i, q in enumerate(p):
        cv2.circle(img, q, 6 if i in (4, 8, 12, 16, 20) else 4, C.WHITE, -1, cv2.LINE_AA)
