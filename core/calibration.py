"""First-run calibration (~30 s): camera check, affected side, reach and grip baselines.

The measured baseline personalises the games:
  * reach_top            -> how high Bubble Garden bubbles hover
  * open/fist ratio      -> Lantern Lighter's openness scale
  * neck_ratio           -> Mirror Dancer's shrug / posture baseline
  * affected_side        -> default hand, weaker-side reporting
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from . import ui
from .gestures import openness_ratio
from .pose_metrics import features
from .scenes import Scene, footer
from .ui import C, HoverButton

STEPS = ["camera", "side", "reach", "grip", "done"]


class CalibrationScene(Scene):
    def __init__(self, app):
        super().__init__(app)
        cfg = app.cfg
        self.step = "camera"
        self.step_t = 0.0
        self.ok_t = 0.0
        self.data = {"affected_side": "both", "reach_top": 0.22, "reach_top_left": None, "reach_top_right": None,
                     "open_ratio": cfg.OPEN_RATIO_DEFAULT, "fist_ratio": cfg.FIST_RATIO_DEFAULT,
                     "neck_ratio": None, "shoulder_width": None, "brightness": None}
        self.checks = {"light": False, "body": False, "distance": False}
        self.dist_msg = ""
        self.neck_samples: list = []
        self.reach = {"Left": 1.0, "Right": 1.0}
        self.ratios: list = []
        self.side_btns = [(k, HoverButton((140 + i * 260, 330, 230, 130), lab, C.ACCENT, sub=sub, size=30, hold=1.2))
                          for i, (k, lab, sub) in enumerate((("left", "Left", "my left side is weaker"),
                                                             ("right", "Right", "my right side is weaker"),
                                                             ("both", "Both", "both sides"),
                                                             ("none", "Not sure", "no weaker side")))]
        self.cursors = []
        self.cycles = 0
        self._was_open = False

    @property
    def needs(self):
        return frozenset({"pose"}) if self.step == "camera" else frozenset({"hands"})

    def next_step(self):
        i = STEPS.index(self.step)
        self.step = STEPS[i + 1]
        self.step_t = 0.0
        self.ok_t = 0.0
        self.app.audio.play("chime")
        if self.step == "done":
            self._save()

    def _save(self):
        d = self.data
        if d["open_ratio"] - d["fist_ratio"] < 0.4:            # implausible -> defaults
            d["open_ratio"], d["fist_ratio"] = self.app.cfg.OPEN_RATIO_DEFAULT, self.app.cfg.FIST_RATIO_DEFAULT
        tops = [v for v in (d["reach_top_left"], d["reach_top_right"]) if v is not None]
        d["reach_top"] = float(np.clip(min(tops), 0.08, 0.55)) if tops else 0.22
        d["date"] = time.strftime("%Y-%m-%d %H:%M")
        self.app.progress.set_calibration(d)

    def on_key(self, key):
        if key in (ord("s"), ord("S")):
            self.data["affected_side"] = self.data.get("affected_side", "both")
            self.step = "done"
            self._save()
            self.app.open_menu()
            return True
        if self.step == "side" and key in (ord("l"), ord("r"), ord("b"), ord("u")):
            self._choose({"l": "left", "r": "right", "b": "both", "u": "none"}[chr(key)])
            return True
        if key in (13, 32) and self.step != "side":
            if self.step == "done":
                self.app.open_menu()
            else:
                self.next_step()
            return True
        return False

    def _choose(self, side):
        self.data["affected_side"] = side
        self.app.progress.set_setting("hand_choice", side if side in ("left", "right") else "both")
        self.next_step()

    def update(self, frame, tracking, dt):
        super().update(frame, tracking, dt)
        self.step_t += dt
        app = self.app
        if self.step == "camera":
            bright = float(np.mean(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)))
            self.data["brightness"] = round(bright, 1)
            self.checks["light"] = bright >= app.cfg.MIN_BRIGHTNESS or app.demo
            pose = tracking.pose
            self.checks["body"] = bool(pose is not None and min(pose.vis[11], pose.vis[12], pose.vis[0]) > 0.5)
            if self.checks["body"]:
                f = features(pose, tracking.aspect)
                sw_frac = f["sw"] / tracking.aspect          # shoulder width as a fraction of screen width
                self.data["shoulder_width"] = round(sw_frac, 3)
                if sw_frac > 0.42:
                    self.checks["distance"], self.dist_msg = False, "Too close - move back a little"
                elif sw_frac < 0.12:
                    self.checks["distance"], self.dist_msg = False, "Too far - come a little closer"
                else:
                    self.checks["distance"], self.dist_msg = True, "Good distance"
                    self.neck_samples.append(f["neck"])
            else:
                self.checks["distance"], self.dist_msg = False, "Sit so your head and shoulders are visible"
            self.ok_t = self.ok_t + dt if all(self.checks.values()) else 0.0
            if self.ok_t > 2.0 or self.step_t > 15:
                if self.neck_samples:
                    self.data["neck_ratio"] = round(float(np.median(self.neck_samples)), 3)
                self.next_step()
        elif self.step == "side":
            self.cursors = app.cursors(tracking)
            for key, b in self.side_btns:
                if b.update(self.cursors, dt, app.click):
                    self._choose(key)
                    break
        elif self.step == "reach":
            for h in tracking.hands:
                self.reach[h.side] = min(self.reach[h.side], float(h.palm[1]))
            if self.step_t > 9:
                for s in ("Left", "Right"):
                    if self.reach[s] < 0.95:
                        self.data["reach_top_" + s.lower()] = round(self.reach[s], 3)
                self.next_step()
        elif self.step == "grip":
            h = tracking.preferred_hand(None)
            if h is not None:
                r = openness_ratio(h.world)
                self.ratios.append(r)
                is_open = r > (app.cfg.OPEN_RATIO_DEFAULT + app.cfg.FIST_RATIO_DEFAULT) / 2
                if is_open != self._was_open and not is_open:
                    self.cycles += 1
                self._was_open = is_open
            if self.step_t > 9 or self.cycles >= 3 and self.step_t > 4:
                if len(self.ratios) > 10:
                    r = np.array(self.ratios)
                    self.data["open_ratio"] = round(float(np.percentile(r, 95)), 3)
                    self.data["fist_ratio"] = round(float(np.percentile(r, 5)), 3)
                self.next_step()
        elif self.step == "done":
            if self.step_t > 4.0:
                app.open_menu()

    def demo_intent(self):
        if self.step == "camera":
            return {"pose": {"spec": {}}}
        if self.step == "side":
            b = dict(self.side_btns)["both"]
            if self.step_t < 0.8:
                return {"hands": [{"side": "Right", "pos": (0.5, 0.85), "gesture": "point", "anchor": "index"}]}
            cx, cy = b.center
            return {"hands": [{"side": "Right", "pos": (cx / self.W, cy / self.H), "gesture": "point", "anchor": "index"}]}
        if self.step == "reach":
            up = (math.sin(self.step_t * 1.3) + 1) / 2
            return {"hands": [{"side": "Left", "pos": (0.3, 0.75 - 0.55 * up), "gesture": "open"},
                              {"side": "Right", "pos": (0.7, 0.75 - 0.52 * up), "gesture": "open"}]}
        if self.step == "grip":
            g = "open" if int(self.step_t / 1.2) % 2 == 0 else "fist"
            return {"hands": [{"side": "Right", "pos": (0.5, 0.55), "gesture": g}]}
        return {"hands": []}

    # ------------------------------------------------------------------ drawing
    def draw(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.45, (50, 30, 24))
        n = STEPS.index(self.step)
        ui.panel(img, (24, 16, W - 48, 96), C.PANEL_DARK, 0.85, 26)
        ui.text(img, "Setting up  -  about 30 seconds", (52, 28), 34, C.TEXT, True)
        ui.text(img, "This personalises every game to YOUR body.   S = skip,  SPACE = next", (54, 74), 20, C.MUTED)
        for i in range(4):
            cx = W - 300 + i * 70
            col = C.GOOD if i < n else (C.GOLD if i == n else (90, 80, 76))
            cv2.circle(img, (cx, 64), 20, col, -1, cv2.LINE_AA)
            ui.text(img, str(i + 1), (cx, 63), 22, (30, 22, 20), True, "mm")
        getattr(self, "_draw_" + self.step)(img)
        footer(img, W, H)

    def _instr(self, img, title, sub):
        W = self.W
        ui.panel(img, (W // 2 - 520, 130, 1040, 110), C.PANEL_DARK, 0.85, 26, border=C.ACCENT)
        ui.text(img, title, (W // 2, 168), 40, C.TEXT, True, "mm")
        ui.text(img, sub, (W // 2, 214), 24, C.MUTED, False, "mm")

    def _draw_camera(self, img):
        W, H = self.W, self.H
        self._instr(img, "Step 1: Camera check", "Sit comfortably so your head and shoulders are in view")
        rows = [("light", "Lighting", "Good light" if self.checks["light"] else "Too dark - turn on a light"),
                ("body", "Upper body visible", "I can see you" if self.checks["body"] else "I can't see your shoulders"),
                ("distance", "Distance", self.dist_msg or "...")]
        y = 300
        for key, label, msg in rows:
            ok = self.checks[key]
            ui.panel(img, (W // 2 - 360, y, 720, 70), C.PANEL_DARK, 0.85, 20, border=C.GOOD if ok else C.WARN)
            if ok:
                ui.check(img, (W // 2 - 315, y + 35), 30, C.GOOD, 5)
            else:
                ui.warn_icon(img, (W // 2 - 315, y + 35), 34)
            ui.text(img, label, (W // 2 - 270, y + 35), 28, C.TEXT, True, "lm")
            ui.text(img, msg, (W // 2 + 340, y + 35), 22, C.GOOD if ok else C.WARN, True, "rm")
            y += 86
        ui.progress_bar(img, (W // 2 - 300, 580, 600, 18), self.ok_t / 2.0, C.GOOD)

    def _draw_side(self, img):
        self._instr(img, "Step 2: Which side is weaker?", "Hold your hand over an answer (or press L / R / B / U)")
        for key, b in self.side_btns:
            b.draw(img)
        for c in self.cursors:
            ui.draw_cursor(img, c, max(b.progress for _, b in self.side_btns))

    def _draw_reach(self, img):
        W, H = self.W, self.H
        self._instr(img, "Step 3: Reach up", "Raise each hand as HIGH as is comfortable - one at a time is fine")
        for s, col in (("Left", C.ACCENT), ("Right", C.PINK)):
            y = self.reach[s]
            if y < 0.95:
                yy = int(y * H)
                x0, x1 = (40, W // 2 - 20) if s == "Left" else (W // 2 + 20, W - 40)
                cv2.line(img, (x0, yy), (x1, yy), col, 3, cv2.LINE_AA)
                ui.text(img, f"{s} best: {100 * (1 - y):.0f}%", (x0 + 10, yy - 30), 22, col, True)
        ui.progress_bar(img, (W // 2 - 300, H - 70, 600, 20), self.step_t / 9, C.ACCENT, label="")

    def _draw_grip(self, img):
        W, H = self.W, self.H
        self._instr(img, "Step 4: Open and close", "Spread your fingers WIDE, then make a gentle FIST.  Repeat 3 times")
        ui.hand_icon(img, "open", (W // 2 - 140, 380), 110, C.ACCENT)
        ui.text(img, "open", (W // 2 - 140, 470), 26, C.TEXT, True, "mm")
        ui.hand_icon(img, "fist", (W // 2 + 140, 380), 110, C.PINK)
        ui.text(img, "fist", (W // 2 + 140, 470), 26, C.TEXT, True, "mm")
        ui.text(img, f"Repetitions: {self.cycles} / 3", (W // 2, 540), 30, C.GOLD, True, "mm")
        ui.progress_bar(img, (W // 2 - 300, H - 70, 600, 20), self.step_t / 9, C.ACCENT)

    def _draw_done(self, img):
        W = self.W
        d = self.data
        self._instr(img, "All set!", "Your games are now personalised. Opening your journey map...")
        lines = [("Weaker side", d["affected_side"]),
                 ("Comfortable reach", f"{100 * (1 - d['reach_top']):.0f}% of screen height"),
                 ("Finger range (open / fist)", f"{d['open_ratio']:.2f} / {d['fist_ratio']:.2f}"),
                 ("Lighting", f"{d['brightness'] or 0:.0f} / 255")]
        y = 300
        for a, b in lines:
            ui.text(img, a, (W // 2 - 330, y), 28, C.MUTED)
            ui.text(img, str(b), (W // 2 + 330, y), 28, C.TEXT, True, "rt")
            y += 52
