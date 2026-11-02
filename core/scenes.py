"""Screens of the app: menu journey map, tutorial, game wrapper, results, rest, pain stop, report, messages."""
from __future__ import annotations

import math
import threading
import webbrowser

import cv2
import numpy as np

from . import ui
from .effects import clamp, ease_out_back, ease_out_cubic, ease_in_out_sine
from .ui import C, DIFF_COLORS, HoverButton

DISCLAIMER = "Practice aid only - it does not replace your physiotherapist or doctor."


class Scene:
    needs = frozenset()
    allow_pain_gesture = True
    counts_as_play = False

    def __init__(self, app):
        self.app = app
        self.W, self.H = app.W, app.H
        self.t = 0.0

    def update(self, frame, tracking, dt):
        self.t += dt

    def draw(self, img):
        pass

    def on_key(self, key: int) -> bool:
        return False

    def demo_intent(self):
        return None          # None -> virtual hand follows the mouse in demo mode


def footer(img, W, H):
    ui.text(img, DISCLAIMER, (W // 2, H - 14), 18, C.MUTED, False, "mb")


def title_bar(img, W, title, sub=""):
    ui.text(img, title, (48, 26), 50, C.TEXT, True, stroke=2)
    if sub:
        ui.text(img, sub, (52, 88), 24, C.MUTED)


# ==========================================================================
# Menu: the journey map
# ==========================================================================
class MenuScene(Scene):
    needs = frozenset({"hands"})
    NODE_POS = [(170, 370), (400, 245), (630, 370), (860, 245), (1090, 370)]

    def __init__(self, app):
        super().__init__(app)
        from games.registry import GAMES
        self.games = GAMES
        self.nodes = [HoverButton((x, y, 62), str(g), circle=True) for g, (x, y) in zip(GAMES, self.NODE_POS)]
        by = 590
        self.btn_report = HoverButton((200, by, 270, 70), "Progress Report", C.ACCENT, size=26)
        self.btn_calib = HoverButton((505, by, 270, 70), "Recalibrate", C.TEAL, size=26)
        self.btn_quit = HoverButton((810, by, 270, 70), "Quit", C.BAD, size=26)
        self.cursors = []
        self.locked_msg_t = 0.0
        self.auto_t = 0.0

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        app = self.app
        self.cursors = app.cursors(tracking)
        click = app.click
        self.locked_msg_t = max(0.0, self.locked_msg_t - dt)
        for gid, node in zip(self.games, self.nodes):
            node.enabled = app.progress.is_unlocked(gid)
            if node.update(self.cursors, dt, click):
                self.choose(gid)
                return
        if self.btn_report.update(self.cursors, dt, click):
            app.audio.play("select")
            app.switch(ReportScene(app))
        elif self.btn_calib.update(self.cursors, dt, click):
            app.audio.play("select")
            app.start_calibration()
        elif self.btn_quit.update(self.cursors, dt, click):
            app.running = False
        if app.autoplay:
            self.auto_t += dt
            if self.auto_t > 1.5:
                if app.progress.all_passed():
                    app.autoplay_finished()
                else:
                    self.choose(app.progress.next_game())

    def choose(self, gid):
        app = self.app
        if not app.progress.is_unlocked(gid):
            app.audio.play("miss")
            self.locked_msg_t = 2.5
            self.locked_gid = gid
            return
        app.audio.play("select")
        app.start_tutorial(gid)

    def on_key(self, key):
        if ord("1") <= key <= ord("5"):
            self.choose(key - ord("0"))
            return True
        if key in (ord("r"), ord("R")):
            self.app.switch(ReportScene(self.app))
            return True
        if key in (ord("c"), ord("C")):
            self.app.start_calibration()
            return True
        if key in (ord("n"), ord("N")):
            gid = self.app.progress.next_game()
            self.app.progress.skip(gid)
            self.app.toast(f"Teacher demo: Game {gid} marked as passed")
            return True
        return False

    def draw(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.55, (50, 28, 24))
        title_bar(img, W, "RehabQuest", "Your recovery journey  -  hold your hand over a game for 1.5 s to start")
        prog = self.app.progress
        # path between nodes
        pts = []
        for (x0, y0), (x1, y1) in zip(self.NODE_POS, self.NODE_POS[1:]):
            for k in range(0, 20):
                t = k / 20
                cx = x0 + (x1 - x0) * t
                cy = y0 + (y1 - y0) * t - math.sin(t * math.pi) * 40
                pts.append((cx, cy))
        for i, (x, y) in enumerate(pts):
            if i % 2 == 0:
                seg_game = i // 20 + 2
                col = C.GOLD if prog.is_unlocked(seg_game) else (90, 80, 76)
                cv2.circle(img, (int(x), int(y)), 4, col, -1, cv2.LINE_AA)
        nxt = prog.next_game()
        for (gid, cls), node in zip(self.games.items(), self.nodes):
            self._draw_node(img, gid, cls, node, gid == nxt and not prog.all_passed())
        # difficulty legend
        ui.chip(img, "EASY x3", (W - 420, 40), C.EASY, 18)
        ui.chip(img, "MEDIUM", (W - 300, 40), C.MEDIUM, 18)
        ui.chip(img, "HARD", (W - 180, 40), C.HARD, 18)
        for b in (self.btn_report, self.btn_calib, self.btn_quit):
            b.draw(img)
        if self.locked_msg_t > 0:
            g = getattr(self, "locked_gid", 2)
            msg = f"Game {g} is locked - pass Game {g - 1} first to unlock it"
            tw, _ = ui.text_size(msg, 28, True)
            ui.panel(img, (W // 2 - tw // 2 - 70, 120, tw + 110, 60), C.PANEL_DARK, 0.92, 30, border=C.WARN)
            ui.lock(img, (W // 2 - tw // 2 - 34, 150), 30, C.WARN)
            ui.text(img, msg, (W // 2 + 20, 150), 28, C.TEXT, True, "mm")
        ui.text(img, "Keys: 1-5 play  |  R report  |  C calibrate  |  M mute  |  F fullscreen  |  P pain stop  |  ESC quit",
                (W // 2, H - 48), 18, C.MUTED, False, "mm")
        footer(img, W, H)
        for c in self.cursors:
            ui.draw_cursor(img, c, max([n.progress for n in self.nodes + [self.btn_report, self.btn_calib, self.btn_quit]]))

    def _draw_node(self, img, gid, cls, node, current):
        cx, cy, r = node.rect
        prog = self.app.progress
        g = prog.game(gid)
        unlocked = prog.is_unlocked(gid)
        col = DIFF_COLORS[cls.difficulty] if unlocked else C.LOCK
        bob = int(math.sin(self.t * 2 + gid) * 4) if unlocked else 0
        cy += bob
        if current:
            pulse = 0.5 + 0.5 * math.sin(self.t * 4)
            ui.radial_glow(img, (cx, cy), int(r * 2.1), col, 0.35 + 0.25 * pulse)
        scale = 1.08 if node.hovered else 1.0
        rr = int(r * scale)
        cv2.circle(img, (cx, cy), rr + 6, (30, 20, 18), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rr, col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rr, C.WHITE, 3, cv2.LINE_AA)
        if unlocked:
            ui.text(img, str(gid), (cx, cy - 4), 56, (30, 22, 20), True, "mm")
        else:
            ui.lock(img, (cx, cy), 46, (60, 54, 50))
        if g["passed"]:
            cv2.circle(img, (cx + rr - 10, cy - rr + 10), 18, C.GOOD, -1, cv2.LINE_AA)
            ui.check(img, (cx + rr - 10, cy - rr + 12), 18, C.WHITE, 4)
        node.draw_ring(img)
        ty = cy + rr + 22
        ui.text(img, cls.name, (cx, ty), 24, C.TEXT if unlocked else C.MUTED, True, "mt", stroke=2)
        ui.chip(img, cls.difficulty, (cx, ty + 32), col, 15, "mt")
        for k in range(3):
            ui.star(img, (cx - 30 + 30 * k, ty + 80), 12, C.GOLD, k < g["stars"])
        if g["best_score"]:
            ui.text(img, f"best {g['best_score']}", (cx, ty + 98), 18, C.MUTED, False, "mt")
        if current:
            ui.text(img, "PLAY NEXT", (cx, cy - rr - 30), 20, C.GOLD, True, "mm", stroke=2)

    def demo_intent(self):
        return None


# ==========================================================================
# Tutorial
# ==========================================================================
class TutorialScene(Scene):
    def __init__(self, app, game_cls):
        super().__init__(app)
        self.cls = game_cls
        st = app.progress.game(game_cls.id)
        self.ease = st["ease"]
        self.choice = app.progress.setting("hand_choice") or {
            "left": "left", "right": "right"}.get(app.progress.calibration.get("affected_side", ""), "both")
        self.preview = game_cls(app, self.ease, {"hand": self.choice})
        self.info = self.preview.intro()
        self.needs = frozenset(game_cls.needs | {"hands"}) if game_cls.has_hand_choice else game_cls.needs
        self.gate = 0.0
        self.visible = False
        self.choice_btns = []
        if game_cls.has_hand_choice:
            for i, (key, label) in enumerate((("left", "Left hand"), ("right", "Right hand"), ("both", "Both"))):
                self.choice_btns.append((key, HoverButton((744 + i * 168, 492, 156, 56), label, C.ACCENT, size=22,
                                                          hold=1.0)))
        self.fails = st["fails_in_row"]
        self.cursors = []

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        app = self.app
        self.visible = self.preview.tracking_ok(tracking)
        self.cursors = app.cursors(tracking) if self.choice_btns else []
        for key, b in self.choice_btns:
            if b.update(self.cursors, dt, app.click):
                self.set_choice(key)
        hovering = any(b.hovered for _, b in self.choice_btns)
        if self.visible and not hovering and self.t > 1.0:
            self.gate += dt
        else:
            self.gate = max(0.0, self.gate - dt * 1.5)
        if self.gate >= app.cfg.START_GATE_SECONDS:
            self.start()

    def set_choice(self, key):
        self.choice = key
        self.app.progress.set_setting("hand_choice", key)
        self.preview = self.cls(self.app, self.ease, {"hand": self.choice})
        self.info = self.preview.intro()
        self.app.audio.play("select")

    def start(self):
        game = self.cls(self.app, self.ease, {"hand": self.choice})
        self.app.switch(GameScene(self.app, game))

    def on_key(self, key):
        if key in (13, 32):
            self.start()
            return True
        if self.choice_btns and key in (ord("l"), ord("r"), ord("b")):
            self.set_choice({"l": "left", "r": "right", "b": "both"}[chr(key)])
            return True
        return False

    def draw(self, img):
        W, H = self.W, self.H
        info = self.info
        ui.darken(img, 0.62, (45, 26, 22))
        col = DIFF_COLORS[info["difficulty"]]
        # left column
        ui.panel(img, (36, 24, 680, H - 70), C.PANEL_DARK, 0.86, 28)
        ui.text(img, f"GAME {self.cls.id}", (66, 44), 22, C.MUTED, True)
        ui.chip(img, info["difficulty"], (170, 42), col, 16)
        ui.text(img, info["title"], (64, 74), 58, C.TEXT, True, stroke=2)
        y = 150
        ui.text(img, "Helps with:", (66, y), 22, C.GOLD, True)
        y = ui.paragraph(img, info["condition"], (66, y + 30), 620, 24, C.TEXT, True)
        y = ui.paragraph(img, info["helps"], (66, y + 2), 620, 22, C.MUTED)
        y += 12
        for i, line in enumerate(info["instructions"], 1):
            cv2.circle(img, (86, y + 20), 22, col, -1, cv2.LINE_AA)
            ui.text(img, str(i), (86, y + 20), 26, (30, 22, 20), True, "mm")
            y = ui.paragraph(img, line, (124, y + 3), 560, 30, C.TEXT, True, 1.2) + 10
        pass_lines = len(ui.wrap("To pass: " + self.preview.pass_text, 24, 540, True))
        bh = 34 + 30 * pass_lines
        ui.panel(img, (60, y + 4, 632, bh), (40, 70, 50), 0.9, 18, border=C.GOOD)
        ui.check(img, (92, y + 4 + bh // 2), 26, C.GOOD, 5)
        ui.paragraph(img, "To pass: " + self.preview.pass_text, (124, y + 18), 540, 24, C.TEXT, True, 1.25)
        y += bh + 18
        if self.ease:
            ui.panel(img, (60, y, 632, 70), (40, 60, 80), 0.9, 18, border=C.WARN)
            ui.text(img, f"Help mode {self.ease}: targets are a little easier", (80, y + 10), 22, C.WARN, True)
            ui.paragraph(img, "Tip: " + self.cls.fail_tip, (80, y + 38), 600, 20, C.TEXT)
        # demo animation
        rect = (744, 24, 500, 452 if self.choice_btns else 500)
        self.preview.draw_demo(img, rect, self.t)
        for key, b in self.choice_btns:
            b.color = C.GOLD if key == self.choice else C.ACCENT
            b.label = ("> " if key == self.choice else "") + {"left": "Left hand", "right": "Right hand", "both": "Both"}[key]
            b.draw(img)
        # gate
        gy = 640 if self.choice_btns else 610
        gx = 994
        frac = self.gate / self.app.cfg.START_GATE_SECONDS
        ui.panel(img, (744, gy - 54, 500, 108), C.PANEL_DARK, 0.88, 26,
                 border=C.GOOD if self.visible else C.WARN, border_t=3)
        ui.ring(img, (800, gy), 34, frac, C.GOOD, 9)
        if self.visible:
            ui.check(img, (800, gy + 2), 30, C.GOOD, 5)
            ui.text(img, "Great - hold still..." if frac < 1 else "Starting!", (850, gy - 16), 28, C.TEXT, True, "lm")
            ui.text(img, "or press SPACE", (850, gy + 20), 20, C.MUTED, False, "lm")
        else:
            ui.warn_icon(img, (800, gy), 34)
            ui.text(img, info["gate_text"], (850, gy - 14), 26, C.TEXT, True, "lm")
            ui.text(img, "or press SPACE to start   |   ESC = menu", (850, gy + 20), 19, C.MUTED, False, "lm")
        footer(img, W, H)
        for c in self.cursors:
            ui.draw_cursor(img, c, max((b.progress for _, b in self.choice_btns), default=0))

    def demo_intent(self):
        g = self.preview
        if "pose" in g.needs:
            return {"pose": {"spec": {}}, "hands": []}
        if "face" in g.needs:
            return {"face": {"blend": {}}}
        return {"hands": [{"side": "Right", "pos": (0.3, 0.55), "gesture": "open"}]}


# ==========================================================================
# Game wrapper
# ==========================================================================
class GameScene(Scene):
    counts_as_play = True

    def __init__(self, app, game):
        super().__init__(app)
        self.game = game
        self.needs = game.needs
        self.finished = False

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        self.game.update(frame, tracking, dt)
        if self.game.done and not self.finished:
            self.finished = True
            self.app.finish_game(self.game)

    def draw(self, img):
        self.game.draw(img)

    def demo_intent(self):
        return self.game.demo_intent()

    def on_key(self, key):
        if key in (ord("n"), ord("N")):
            self.app.progress.skip(self.game.id)
            self.app.toast(f"Teacher demo: Game {self.game.id} skipped and marked as passed")
            self.app.open_menu()
            return True
        return False


# ==========================================================================
# Results
# ==========================================================================
class ResultScene(Scene):
    needs = frozenset({"hands"})

    def __init__(self, app, game_cls, res, flags, history):
        super().__init__(app)
        self.cls = game_cls
        self.res = res
        self.flags = flags
        self.history = history[-10:]
        from .progress import GAME_COUNT
        has_next = res.passed and res.game_id < GAME_COUNT
        y = 600
        self.btn_retry = HoverButton((250, y, 240, 70), "Retry", C.ACCENT, size=28)
        self.btn_menu = HoverButton((520, y, 240, 70), "Menu", C.TEAL, size=28)
        self.btn_next = HoverButton((790, y, 240, 70), "Next Game", C.GOOD, size=28) if has_next else None
        self.cursors = []
        if res.passed:
            app.fx.confetti(self.W, 160)
        self.star_sounds = 0

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        app = self.app
        self.cursors = app.cursors(tracking)
        n_shown = min(self.res.stars, int(max(0, self.t - 0.4) / 0.35) + (1 if self.t > 0.4 else 0))
        while self.star_sounds < n_shown:
            self.star_sounds += 1
            app.audio.play("chime")
        if self.t < 1.0:
            return
        if self.btn_retry.update(self.cursors, dt, app.click):
            app.start_tutorial(self.res.game_id)
        elif self.btn_menu.update(self.cursors, dt, app.click):
            app.open_menu()
        elif self.btn_next and self.btn_next.update(self.cursors, dt, app.click):
            app.start_tutorial(self.res.game_id + 1)
        elif app.autoplay and self.t > 3.0:
            app.open_menu()

    def on_key(self, key):
        if key in (13, 32):
            if self.btn_next:
                self.app.start_tutorial(self.res.game_id + 1)
            else:
                self.app.start_tutorial(self.res.game_id)
            return True
        return False

    def draw(self, img):
        W, H = self.W, self.H
        res = self.res
        ui.darken(img, 0.7, (40, 24, 20))
        k = ease_out_back(clamp(self.t * 2.2))
        pw, ph = 1060, 560
        px, py = W // 2 - pw // 2, 20 + int((1 - k) * 60)
        ui.panel(img, (px, py, pw, ph), C.PANEL_DARK, 0.92, 32, border=C.GOOD if res.passed else C.WARN, border_t=4)
        title = "PASSED!" if res.passed else "Not yet - let's try again!"
        ui.text(img, title, (W // 2, py + 52), 60 if res.passed else 46, C.GOOD if res.passed else C.WARN, True, "mm",
                stroke=3)
        ui.text(img, f"Game {res.game_id}: {res.name}", (W // 2, py + 102), 26, C.MUTED, True, "mm")
        # stars
        for i in range(3):
            appear = clamp((self.t - 0.4 - i * 0.35) * 4)
            s = int(42 * ease_out_back(appear)) if i < res.stars else 42
            ui.star(img, (W // 2 - 110 + i * 110, py + 170), max(4, s), C.GOLD, i < res.stars and appear > 0, 3)
        # score count-up
        shown = int(res.score * ease_out_cubic(clamp(self.t / 1.2)))
        ui.text(img, f"{shown}", (px + 60, py + 240), 64, C.GOLD, True, stroke=2)
        ui.text(img, "points", (px + 64, py + 312), 22, C.MUTED)
        ui.text(img, f"{res.accuracy * 100:.0f}%", (px + 260, py + 240), 64, C.ACCENT, True, stroke=2)
        ui.text(img, "accuracy", (px + 264, py + 312), 22, C.MUTED)
        # stats
        y = py + 352
        for label, val in res.summary[:5]:
            ui.text(img, label, (px + 60, y), 24, C.MUTED)
            ui.text(img, str(val), (px + 520, y), 24, C.TEXT, True, "rt")
            y += 36
        # mini chart: score history
        self._mini_chart(img, (px + 580, py + 220, 440, 230))
        # messages
        msg, col = "", C.TEXT
        if self.flags.get("promoted"):
            nxt = res.game_id + 1
            msg, col = (f"Game {nxt} unlocked!  Your journey continues." if nxt <= 5 else
                        "You completed the whole journey - amazing!"), C.GOLD
        elif self.flags.get("ready_next"):
            msg, col = "Excellent! Ready for the next challenge?", C.GOLD
        elif self.flags.get("eased"):
            msg, col = "We made the targets a little easier for your next try.", C.WARN
        elif not res.passed:
            msg, col = "Tip: " + res.hint, C.WARN
        elif self.flags.get("new_best"):
            msg, col = "New personal best!", C.GOLD
        if msg:
            ui.text(img, msg, (W // 2, py + ph - 52), 26, col, True, "mm", stroke=2)
        for b in (self.btn_retry, self.btn_menu, self.btn_next):
            if b:
                b.draw(img)
        footer(img, W, H)
        for c in self.cursors:
            ui.draw_cursor(img, c, max(b.progress for b in (self.btn_retry, self.btn_menu, self.btn_next) if b))

    def _mini_chart(self, img, rect):
        x, y, w, h = rect
        ui.panel(img, rect, (30, 20, 18), 0.9, 16)
        ui.text(img, "Your last sessions (score)", (x + 16, y + 10), 20, C.MUTED, True)
        scores = [int(float(r.get("score", 0))) for r in self.history]
        passed = [r.get("passed") in ("1", 1, True) for r in self.history]
        if not scores:
            ui.text(img, "First session!", (x + w // 2, y + h // 2), 26, C.TEXT, True, "mm")
            return
        top = max(scores) or 1
        n = len(scores)
        bw = min(36, (w - 40) // n - 6)
        for i, (s, p) in enumerate(zip(scores, passed)):
            bh = int((h - 80) * s / top * ease_out_cubic(clamp(self.t * 1.5 - i * 0.05)))
            bx = x + 24 + i * (bw + 8)
            by = y + h - 30
            col = C.GOOD if p else C.WARN
            if i == n - 1:
                col = C.GOLD
            ui.rounded_rect(img, (bx, by - bh, bw, max(4, bh)), 6, col, -1)
        ui.text(img, "green = passed   gold = today", (x + 16, y + h - 24), 16, C.MUTED)


# ==========================================================================
# Rest reminder / pain stop / report / messages
# ==========================================================================
class RestScene(Scene):
    needs = frozenset({"hands"})
    allow_pain_gesture = False

    def __init__(self, app, after=None):
        super().__init__(app)
        self.after = after
        self.dur = app.cfg.REST_SECONDS
        self.btn = HoverButton((self.W // 2 - 160, 560, 320, 74), "I'm rested", C.GOOD, size=30)
        self.cursors = []
        app.audio.play("chime")

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        self.cursors = self.app.cursors(tracking)
        if (self.t > 15 and self.btn.update(self.cursors, dt, self.app.click)) or self.t > self.dur \
                or (self.app.autoplay and self.t > 2):
            self.app.open_menu()

    def on_key(self, key):
        if key in (13, 32) and self.t > 3:
            self.app.open_menu()
            return True
        return False

    def draw(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.75, (60, 40, 30))
        ui.text(img, "Time for a rest", (W // 2, 90), 64, C.TEXT, True, "mm", stroke=2)
        ui.text(img, "You have been exercising for a while. Relax your arms and breathe.", (W // 2, 150), 28,
                C.MUTED, False, "mm")
        breath = ease_in_out_sine((math.sin(self.t * 2 * math.pi / 8) + 1) / 2)
        r = int(90 + 70 * breath)
        ui.radial_glow(img, (W // 2, 360), r + 60, C.ACCENT, 0.5)
        cv2.circle(img, (W // 2, 360), r, C.ACCENT, 4, cv2.LINE_AA)
        phase = "Breathe in..." if math.sin(self.t * 2 * math.pi / 8) > 0 else "Breathe out..."
        ui.text(img, phase, (W // 2, 360), 36, C.TEXT, True, "mm")
        left = max(0, int(self.dur - self.t))
        ui.text(img, f"{left}s", (W // 2, 520), 30, C.MUTED, True, "mm")
        if self.t > 15:
            self.btn.draw(img)
            for c in self.cursors:
                ui.draw_cursor(img, c, self.btn.progress)
        footer(img, W, H)


class PainScene(Scene):
    allow_pain_gesture = False
    needs = frozenset({"hands"})

    def __init__(self, app):
        super().__init__(app)
        self.btn = HoverButton((self.W // 2 - 170, 560, 340, 74), "Back to menu", C.ACCENT, size=30, hold=2.0)
        self.cursors = []
        app.audio.play("miss")

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        self.cursors = self.app.cursors(tracking)
        if self.t > 5 and self.btn.update(self.cursors, dt, self.app.click):
            self.app.open_menu()

    def on_key(self, key):
        if key in (13, 32) and self.t > 2:
            self.app.open_menu()
            return True
        return False

    def draw(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.8, (40, 30, 70))
        ui.warn_icon(img, (W // 2, 120), 90)
        ui.text(img, "Exercise stopped", (W // 2, 220), 60, C.TEXT, True, "mm", stroke=2)
        lines = ["You signalled pain or discomfort. That's okay - stopping was the right thing to do.",
                 "Rest your arms, breathe slowly and drink some water.",
                 "If the pain continues or is sharp, please contact your physiotherapist or doctor."]
        for i, s in enumerate(lines):
            ui.text(img, s, (W // 2, 310 + i * 48), 27, C.TEXT if i < 2 else C.WARN, i == 2, "mm")
        if self.t > 5:
            self.btn.draw(img)
            for c in self.cursors:
                ui.draw_cursor(img, c, self.btn.progress)
        else:
            ui.text(img, "Resting...", (W // 2, 590), 26, C.MUTED, False, "mm")
        footer(img, W, H)


class ReportScene(Scene):
    needs = frozenset({"hands"})

    def __init__(self, app):
        super().__init__(app)
        self.status = "Creating your progress report..."
        self.image = None
        self.paths = None
        self.error = ""
        self.btn = HoverButton((self.W - 300, self.H - 110, 260, 66), "Back", C.ACCENT, size=28)
        self.btn_open = HoverButton((self.W - 590, self.H - 110, 270, 66), "Open in browser", C.TEAL, size=24)
        self.cursors = []
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self):
        try:
            from .report import generate_report
            html, png = generate_report(self.app.data_dir, self.app.data_dir.parent / "reports",
                                        demo=self.app.demo)
            self.paths = (html, png)
            img = cv2.imread(str(png))
            self.image = img
            self.status = "Report saved"
            if not self.app.headless and not self.app.autoplay:
                webbrowser.open(html.as_uri())
        except Exception as exc:
            self.app.log_exception(exc)
            self.error = f"Could not build the report: {exc}"

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        self.cursors = self.app.cursors(tracking)
        if self.btn.update(self.cursors, dt, self.app.click):
            self.app.open_menu()
        elif self.paths and self.btn_open.update(self.cursors, dt, self.app.click):
            webbrowser.open(self.paths[0].as_uri())
        if self.app.autoplay and self.t > 4 and (self.image is not None or self.error):
            self.app.autoplay_finished(report_done=True)

    def draw(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.8, (40, 24, 20))
        title_bar(img, W, "Progress Report")
        if self.image is not None:
            ih, iw = self.image.shape[:2]
            maxw, maxh = W - 80, H - 250
            s = min(maxw / iw, maxh / ih)
            im = cv2.resize(self.image, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
            x0 = (W - im.shape[1]) // 2
            img[110:110 + im.shape[0], x0:x0 + im.shape[1]] = im
            ui.text(img, f"Saved: {self.paths[0].name}  (full report with comments opened in your browser)",
                    (40, H - 92), 20, C.MUTED)
            self.btn_open.draw(img)
        elif self.error:
            ui.paragraph(img, self.error, (60, 200), W - 120, 26, C.WARN)
        else:
            a = self.t * 6
            for i in range(8):
                ang = a + i * math.pi / 4
                cv2.circle(img, (int(W / 2 + 40 * math.cos(ang)), int(H / 2 + 40 * math.sin(ang))),
                           4 + i, C.ACCENT, -1, cv2.LINE_AA)
            ui.text(img, self.status, (W // 2, H // 2 + 90), 30, C.TEXT, True, "mm")
        self.btn.draw(img)
        for c in self.cursors:
            ui.draw_cursor(img, c, max(self.btn.progress, self.btn_open.progress))
        footer(img, W, H)


class MessageScene(Scene):
    """Full-screen message with key options (no camera, model download failed, loading)."""

    def __init__(self, app, title, body, options=(), progress=None, icon="warn"):
        super().__init__(app)
        self.title, self.body, self.options = title, body, options
        self.progress = progress
        self.icon = icon

    def draw(self, img):
        W, H = self.W, self.H
        img[:] = ui.vertical_gradient(W, H, (70, 40, 32), (34, 22, 18))
        ui.text(img, "RehabQuest", (W // 2, 90), 64, C.TEXT, True, "mm", stroke=2)
        if self.icon == "warn":
            ui.warn_icon(img, (W // 2, 200), 80)
        else:
            a = self.t * 5
            for i in range(8):
                ang = a + i * math.pi / 4
                cv2.circle(img, (int(W / 2 + 36 * math.cos(ang)), int(200 + 36 * math.sin(ang))), 3 + i,
                           C.ACCENT, -1, cv2.LINE_AA)
        ui.text(img, self.title, (W // 2, 290), 40, C.TEXT, True, "mm")
        y = 350
        for line in self.body.split("\n"):
            ui.text(img, line, (W // 2, y), 24, C.MUTED, False, "mm")
            y += 36
        if self.progress is not None:
            ui.progress_bar(img, (W // 2 - 300, y + 10, 600, 26), self.progress, C.ACCENT)
            y += 60
        for i, opt in enumerate(self.options):
            ui.text(img, opt, (W // 2, y + 20 + i * 44), 28, C.GOLD, True, "mm")
        footer(img, W, H)
