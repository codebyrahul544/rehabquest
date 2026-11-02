"""GAME 2 (EASY) - Lantern Lighter: hand opening / grip for arthritis, stiff fingers, spasticity.

OPEN the hand wide and hold (2 s) to charge a paper lantern, then CLOSE into a
gentle fist (1 s) to light it. Openness = mean fingertip-to-wrist distance /
hand size, rescaled to the patient's own calibrated open/fist range, so it
works at any distance from the camera.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from core import ui
from core.effects import ease_in_out_sine, ease_out_back, clamp
from core.gestures import normalized_openness, openness_ratio
from core.synth import gesture_vector, hand_norm, hand_world
from core.ui import C
from games.base_game import BaseGame, demo_frame, put_hand_in_rect

WARM = (60, 170, 255)
WARM_DEEP = (40, 110, 230)
CHARGE = (255, 210, 120)


class LanternLighter(BaseGame):
    id = 2
    name = "Lantern Lighter"
    condition = "Rheumatoid arthritis, stiff fingers, weak grip, spasticity"
    helps = "Finger extension + flexion range, grip endurance"
    body_part = "Hand & fingers"
    difficulty = "EASY"
    needs = frozenset({"hands"})
    color = WARM
    instructions = ["OPEN your hand wide - spread all fingers. Hold to charge the lantern.",
                    "Then CLOSE into a gentle fist to light it.",
                    "Keep it slow and comfortable. Light the whole festival!"]
    fail_tip = "Hold each position until the ring is full. Gentle is fine - you don't need to squeeze."

    def on_start(self):
        c = self.cfg
        self.time_limit = c.G2_TIME_LIMIT * (1 + 0.15 * self.ease)
        self.target = max(4, int(round(c.G2_PASS_LANTERNS * self.ease_factor)))
        self.pass_text = f"Light {self.target} lanterns within {int(self.time_limit)} seconds"
        self.open_thr = c.G2_OPEN_THRESHOLD * self.ease_factor
        self.fist_thr = min(0.6, c.G2_FIST_THRESHOLD * (1 + c.EASE_STEP * self.ease * 1.5))
        self.open_hold = c.G2_OPEN_HOLD
        self.fist_hold = c.G2_FIST_HOLD
        self.open_ref = float(self.calib.get("open_ratio", c.OPEN_RATIO_DEFAULT))
        self.fist_ref = float(self.calib.get("fist_ratio", c.FIST_RATIO_DEFAULT))
        self.state = "open"
        self.hold = 0.0
        self.state_t = 0.0
        self.lit = 0
        self.openness = 0.0
        self.raw_min, self.raw_max = 9.0, 0.0
        self.open_holds: list = []
        self.best_hold = 0.0
        self.cur_hold = 0.0
        self.flying: list = []           # [x, y, tx, ty, t]
        rng = random.Random(9)
        self.stars_bg = [(rng.randint(0, self.W), rng.randint(0, int(self.H * 0.7)), rng.uniform(0, 6.28),
                          rng.randint(1, 3)) for _ in range(90)]
        self.hand = None

    def prefer_side(self):
        return {"left": "Left", "right": "Right"}.get(self.affected)

    # ------------------------------------------------------------------ logic
    def on_update(self, tracking, dt):
        self.state_t += dt
        hand = tracking.preferred_hand(self.prefer_side())
        self.hand = hand
        if hand is None:
            return
        r = openness_ratio(hand.world)
        self.raw_min, self.raw_max = min(self.raw_min, r), max(self.raw_max, r)
        self.openness = normalized_openness(r, self.fist_ref, self.open_ref)
        if self.openness >= self.open_thr:
            self.cur_hold += dt
            self.best_hold = max(self.best_hold, self.cur_hold)
        else:
            self.cur_hold = 0.0
        cx, cy = self.W // 2, int(self.H * 0.52)
        if self.state == "open":
            if self.openness >= self.open_thr:
                if self.hold == 0:
                    self.audio.play("charge")
                self.hold += dt
            else:
                self.hold = max(0.0, self.hold - dt * 1.5)
            if self.hold >= self.open_hold:
                self.open_holds.append(self.state_t)
                self.reaction_times.append(self.state_t)
                self.state, self.hold, self.state_t = "close", 0.0, 0.0
                self.audio.play("chime")
                self.fx.sparkle(cx, cy, CHARGE, 18)
        elif self.state == "close":
            if self.openness <= self.fist_thr:
                self.hold += dt
            else:
                self.hold = max(0.0, self.hold - dt * 1.5)
            if self.hold >= self.fist_hold:
                self._light(cx, cy)
        elif self.state == "lit":
            if self.state_t > 0.9:
                self.state, self.hold, self.state_t = "open", 0.0, 0.0

    def _light(self, cx, cy):
        self.lit += 1
        self.state, self.hold, self.state_t = "lit", 0.0, 0.0
        self.add_points(120, cx, cy - 120, WARM)
        self.audio.play("success" if self.lit == self.target else "pop")
        self.fx.burst(cx, cy, [WARM, C.GOLD, C.WHITE], 40, 360, 0.9, 7, 120)
        self.fx.shockwave(cx, cy, WARM, 180)
        self.fx.shake(6, 0.2)
        tx, ty = self._slot(self.lit - 1)
        self.flying.append([cx, cy, tx, ty, 0.0])

    def _slot(self, i):
        n = self.target
        x = self.W // 2 + (i - (n - 1) / 2) * min(120, (self.W - 200) / n)
        return int(x), 150

    def on_ending(self, tracking, dt):
        self._advance_flying(dt)

    def _advance_flying(self, dt):
        for f in self.flying:
            f[4] = min(1.0, f[4] + dt * 1.2)

    def is_passed(self):
        return self.lit >= self.target

    # ------------------------------------------------------------------ drawing
    def draw_scene(self, img):
        W, H = self.W, self.H
        self._advance_flying(1 / 30)
        ui.darken(img, 0.62, (60, 20, 10))
        for x, y, ph, s in self.stars_bg:
            a = 0.5 + 0.5 * math.sin(self.t * 2 + ph)
            cv2.circle(img, (x, y), s, tuple(int(150 + 100 * a) for _ in range(3)), -1, cv2.LINE_AA)
        ui.radial_glow(img, (W - 150, 200), 120, (160, 200, 220), 0.25)
        cv2.circle(img, (W - 150, 200), 46, (210, 235, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (W - 135, 190), 40, (90, 50, 35), -1, cv2.LINE_AA)
        # lantern slots
        for i in range(self.target):
            x, y = self._slot(i)
            lit = i < len(self.flying) and self.flying[i][4] >= 1.0
            bob = int(math.sin(self.t * 1.5 + i) * 4) if lit else 0
            draw_lantern(img, x, y + bob, 0.38, 1.0 if lit else 0.0, 0.0, self.t + i)
        if self.hand is not None:
            ui.draw_hand(img, self.hand, CHARGE if self.state == "open" else WARM)
        # main lantern
        cx, cy = W // 2, int(H * 0.52)
        if self.state != "lit" and self.phase != "ending":
            charge = (self.hold / self.open_hold) if self.state == "open" else 1.0
            light = (self.hold / self.fist_hold) if self.state == "close" else 0.0
            s = 0.9 + 0.35 * ease_in_out_sine(charge) + 0.1 * light
            draw_lantern(img, cx, cy, s, light, charge, self.t)
            ring_col = CHARGE if self.state == "open" else WARM
            ui.ring(img, (cx, cy), int(150 * s), self.hold / (self.open_hold if self.state == "open" else self.fist_hold),
                    ring_col, 10)
        for i, f in enumerate(self.flying):
            if f[4] < 1.0:
                k = ease_out_back(f[4])
                x = int(f[0] + (f[2] - f[0]) * k)
                y = int(f[1] + (f[3] - f[1]) * k)
                draw_lantern(img, x, y, 1.1 - 0.72 * f[4], 1.0, 0.0, self.t)
        self._gauge(img)

    def _gauge(self, img):
        x, y, w, h = 50, 200, 46, 380
        ui.panel(img, (x - 24, y - 60, w + 48, h + 120), C.PANEL_DARK, 0.75, 20)
        ui.text(img, "Hand", (x + w // 2, y - 36), 20, C.MUTED, True, "mm")
        ui.rounded_rect(img, (x, y, w, h), 18, (40, 30, 26), -1)
        fh = int(h * self.openness)
        if fh > 4:
            col = CHARGE if self.openness >= self.open_thr else (WARM if self.openness <= self.fist_thr else C.MUTED)
            ui.rounded_rect(img, (x, y + h - fh, w, fh), 18, col, -1)
        for val, label, col in ((self.open_thr, "OPEN", CHARGE), (self.fist_thr, "FIST", WARM)):
            yy = y + h - int(h * val)
            cv2.line(img, (x - 10, yy), (x + w + 10, yy), col, 3, cv2.LINE_AA)
            ui.text(img, label, (x + w + 16, yy), 18, col, True, "lm", stroke=2)
        ui.text(img, f"{int(self.openness * 100)}%", (x + w // 2, y + h + 26), 22, C.TEXT, True, "mm")

    def current_instruction(self):
        if self.state == "open":
            return "OPEN your hand wide and hold"
        if self.state == "close":
            return "Now gently CLOSE into a fist"
        return "Lantern lit!  Get ready for the next one"

    def draw_hud(self, img):
        super().draw_hud(img)
        if self.phase == "play" and not self.tracking_lost and self.state != "lit":
            g = "open" if self.state == "open" else "fist"
            ui.hand_icon(img, g, (self.W // 2 + 330, self.H - 54), 52, CHARGE if g == "open" else WARM)

    def progress_info(self):
        return "Lanterns", self.lit, self.target

    def draw_demo(self, img, rect, t):
        x, y, w, h = rect
        demo_frame(img, rect)
        cyc = t % 4.0
        if cyc < 2.0:
            g, label, charge, light = "open", "1. OPEN wide + hold", min(1, cyc / 1.6), 0.0
        elif cyc < 3.4:
            g, label, charge, light = "fist", "2. Close gently = LIGHT!", 1.0, min(1, (cyc - 2.0) / 0.8)
        else:
            g, label, charge, light = "open", "Next lantern...", 0.0, 0.0
        draw_lantern(img, x + int(w * 0.72), y + int(h * 0.5), 0.8, light, charge if light == 0 else 1, t)
        k = clamp((cyc % 2.0) / 0.4)
        vec = gesture_vector("fist") * k + gesture_vector("open") * (1 - k) if g == "fist" else gesture_vector("open")
        n = hand_norm(hand_world(vec, "Right"), (0.3, 0.55), aspect=w / h, scale=3.0)
        put_hand_in_rect(img, n[:, :2], rect, CHARGE if g == "open" else WARM)
        ui.text(img, label, (x + w // 2, y + h - 34), 28, C.TEXT, True, "mm", stroke=2)

    def demo_intent(self):
        g = "open" if self.state in ("open", "lit") else "fist"
        if self.state == "open" and self.state_t < 0.4:
            g = "relaxed"
        return {"hands": [{"side": "Right", "pos": (0.62, 0.6), "gesture": g, "blend_rate": 3.0}]}

    # ------------------------------------------------------------------ results
    def accuracy(self):
        return self.lit / max(1, self.target)

    def rom(self):
        if self.raw_max <= 0:
            return 0.0, "finger range (open - fist ratio)"
        return self.raw_max - self.raw_min, "finger range (open - fist ratio)"

    def extra(self):
        return {"lit": self.lit, "open_ratio_max": round(self.raw_max, 3),
                "fist_ratio_min": round(self.raw_min if self.raw_min < 9 else 0, 3),
                "best_open_hold_s": round(self.best_hold, 2),
                "mean_time_to_open_s": round(float(np.mean(self.open_holds)), 2) if self.open_holds else 0.0}

    def summary(self):
        e = self.extra()
        return [("Lanterns lit", f"{self.lit} / {self.target}"),
                ("Finger range", f"{self.rom()[0]:.2f} (open {e['open_ratio_max']:.2f} / fist {e['fist_ratio_min']:.2f})"),
                ("Longest open hold", f"{e['best_open_hold_s']:.1f} s"),
                ("Average time to open", f"{e['mean_time_to_open_s']:.1f} s")]


def draw_lantern(img, cx, cy, s, light, charge, t):
    """Paper lantern. light 0..1 = flame, charge 0..1 = blue energy inside."""
    cx, cy = int(cx), int(cy)
    w, h = int(70 * s), int(95 * s)
    if light > 0:
        flick = 0.85 + 0.15 * math.sin(t * 13) * math.sin(t * 7.3)
        ui.radial_glow(img, (cx, cy), int(190 * s), WARM, 0.75 * light * flick)
    body = tuple(int(a * (1 - light) + b * light) for a, b in zip((40, 40, 120), WARM))
    cv2.ellipse(img, (cx, cy), (w, h), 0, 0, 360, body, -1, cv2.LINE_AA)
    if charge > 0 and light < 1:
        ch = int(2 * h * charge * (1 - light))
        roi_top = cy + h - ch
        mask = np.zeros(img.shape[:2], np.uint8)
        y0, y1 = max(0, cy - h), min(img.shape[0], cy + h + 1)
        x0, x1 = max(0, cx - w), min(img.shape[1], cx + w + 1)
        sub = mask[y0:y1, x0:x1]
        cv2.ellipse(sub, (cx - x0, cy - y0), (w - 4, h - 4), 0, 0, 360, 255, -1, cv2.LINE_AA)
        sub[:max(0, roi_top - y0)] = 0
        region = img[y0:y1, x0:x1]
        col = np.array(CHARGE, np.float32)
        a = (sub.astype(np.float32) / 255.0 * 0.55)[..., None]
        region[:] = (region * (1 - a) + col * a).astype(np.uint8)
    rib = tuple(max(0, c - 50) for c in body)
    for k in (0.35, 0.7):
        cv2.ellipse(img, (cx, cy), (int(w * k), h), 0, 0, 360, rib, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (w, h), 0, 0, 360, rib, 2, cv2.LINE_AA)
    cap = (30, 60, 110)
    ui.rounded_rect(img, (cx - int(w * 0.55), cy - h - int(10 * s), int(w * 1.1), int(18 * s)), 4, cap, -1)
    ui.rounded_rect(img, (cx - int(w * 0.55), cy + h - int(8 * s), int(w * 1.1), int(18 * s)), 4, cap, -1)
    cv2.line(img, (cx, cy + h + int(10 * s)), (cx, cy + h + int(34 * s)), (60, 140, 220), max(1, int(3 * s)), cv2.LINE_AA)
    if light > 0.3:
        fh = int(26 * s * light * (0.9 + 0.1 * math.sin(t * 15)))
        pts = np.array([(cx, cy - fh), (cx + int(9 * s), cy + int(6 * s)), (cx - int(9 * s), cy + int(6 * s))], np.int32)
        cv2.fillPoly(img, [pts], (120, 230, 255), cv2.LINE_AA)
