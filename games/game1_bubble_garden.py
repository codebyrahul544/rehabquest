"""GAME 1 (EASY) - Bubble Garden: shoulder flexion / abduction reach after stroke.

Bubbles float up from the garden and hover at a "reach height" that rises as the
patient succeeds, scaled to the reach measured during calibration. Touch a
bubble with your palm to pop it. Escaped bubbles cost nothing.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from core import ui
from core.effects import lerp
from core.synth import hand_norm, hand_world, gesture_vector
from core.ui import C
from games.base_game import BaseGame, demo_frame, put_hand_in_rect

BUBBLE_COLORS = [(255, 170, 90), (200, 120, 255), (120, 230, 140), (90, 200, 255), (255, 120, 200), (120, 220, 255)]


class Bubble:
    def __init__(self, x, y0, target_y, r, color, side):
        self.x, self.y, self.target_y, self.r = x, y0, target_y, r
        self.color, self.side = color, side
        self.state = "rise"
        self.arrived_at = None
        self.hover_t = 0.0
        self.phase = random.uniform(0, 6.28)
        self.born = 0.0


class BubbleGarden(BaseGame):
    id = 1
    name = "Bubble Garden"
    condition = "Stroke - weak arm / reduced shoulder range"
    helps = "Shoulder reach (flexion + abduction) and arm control"
    body_part = "Shoulder & arm"
    difficulty = "EASY"
    needs = frozenset({"hands"})
    color = (255, 170, 90)
    instructions = ["Bubbles float up from the garden.",
                    "Touch a bubble with your open palm to POP it.",
                    "Reach a little higher each time - go slowly, no rush."]
    gate_text = "Show your hand to start"
    has_hand_choice = True

    def on_start(self):
        c = self.cfg
        self.time_limit = c.G1_TIME_LIMIT * (1 + 0.15 * self.ease)
        self.target = max(5, int(round(c.G1_PASS_POPS * self.ease_factor)))
        self.choice = self.settings.get("hand", "both")
        self.r = int(c.G1_BUBBLE_RADIUS * (1 + 0.1 * self.ease))
        self.pass_text = f"Pop {self.target} bubbles within {int(self.time_limit)} seconds"
        self.bubbles: list = []
        self.spawn_t = 0.6
        self.pops = 0
        self.escaped = 0
        self.spawned = 0
        self.alt = 0
        self.reach = {"Left": [], "Right": []}
        self.pops_side = {"Left": 0, "Right": 0}
        self.reach_top = float(self.calib.get("reach_top", 0.22))
        self._garden = None

    # ------------------------------------------------------------------ logic
    def allowed(self, hand) -> bool:
        return self.choice == "both" or hand.side.lower() == self.choice

    def hover_height(self) -> float:
        """Normalised y where new bubbles stop: starts low, climbs towards the patient's best reach."""
        frac = self.pops / max(1, self.target)
        top = max(0.10, self.reach_top - 0.03) + 0.06 * self.ease
        return lerp(0.62, top, min(1.0, frac * 1.15))

    def _spawn(self):
        W, H = self.W, self.H
        self.alt += 1
        if self.choice == "both":
            left = self.alt % 2 == 0
            x = random.uniform(0.12, 0.40) if left else random.uniform(0.60, 0.88)
            side = "Left" if left else "Right"
        else:
            near = self.alt % 3 != 2
            if self.choice == "left":
                x = random.uniform(0.28, 0.45) if near else random.uniform(0.10, 0.22)
                side = "Left"
            else:
                x = random.uniform(0.55, 0.72) if near else random.uniform(0.78, 0.90)
                side = "Right"
        ty = self.hover_height() + random.uniform(-0.04, 0.04)
        b = Bubble(x * W, H + self.r, ty * H, self.r, random.choice(BUBBLE_COLORS), side)
        b.born = self.elapsed
        self.bubbles.append(b)
        self.spawned += 1

    def on_update(self, tracking, dt):
        c = self.cfg
        self.spawn_t -= dt
        active = sum(1 for b in self.bubbles if b.state != "leave")
        if self.spawn_t <= 0 and active < 3:
            self._spawn()
            self.spawn_t = c.G1_SPAWN_INTERVAL * (1 + 0.15 * self.ease)
        for b in self.bubbles:
            b.phase += dt
            if b.state == "rise":
                b.y -= 230 * dt
                if b.y <= b.target_y:
                    b.y = b.target_y
                    b.state = "hover"
                    b.arrived_at = self.elapsed
            elif b.state == "hover":
                b.hover_t += dt
                b.y = b.target_y + math.sin(b.phase * 2.0) * 8
                if b.hover_t > c.G1_HOVER_SECONDS:
                    b.state = "leave"
            else:
                b.y -= 140 * dt
        # pop detection: palm centre and finger tips of allowed hands
        W, H = self.W, self.H
        for hand in tracking.hands:
            if not self.allowed(hand):
                continue
            pts = [hand.palm * (W, H)] + [hand.norm[i, :2] * (W, H) for i in (8, 12)]
            for b in self.bubbles:
                if b.state != "hover":          # only bubbles waiting at the reach line can be popped
                    continue
                if any(math.hypot(p[0] - b.x, p[1] - b.y) < b.r + c.G1_TOUCH_MARGIN for p in pts):
                    self._pop(b, hand)
        kept = []
        for b in self.bubbles:
            if b.state == "popped":
                continue
            if b.y < -b.r * 2:
                self.escaped += 1          # no punishment in easy mode
                continue
            kept.append(b)
        self.bubbles = kept

    def _pop(self, b, hand):
        b.state = "popped"
        self.pops += 1
        side = hand.side
        self.pops_side[side] = self.pops_side.get(side, 0) + 1
        self.reach.setdefault(side, []).append(1.0 - b.y / self.H)
        self.reaction_times.append(max(0.0, self.elapsed - b.arrived_at))
        self.add_points(100, b.x, b.y, b.color)
        self.fx.burst(b.x, b.y, [b.color, C.WHITE], 30, 330, 0.7, 7, 250)
        self.fx.shockwave(b.x, b.y, b.color, b.r * 1.8)
        self.audio.play("pop")
        if self.pops in (5, 10):
            self.big_banner("Great reach!" if self.pops == 5 else "Higher and higher!")

    def is_passed(self):
        return self.pops >= self.target

    # ------------------------------------------------------------------ drawing
    def _garden_layer(self):
        if self._garden is None:
            W, H = self.W, self.H
            h = 130
            layer = np.zeros((h, W, 3), np.uint8)
            mask = np.zeros((h, W), np.float32)
            rng = random.Random(4)
            for x in range(0, W, 9):
                gh = rng.randint(40, 100)
                col = (60 + rng.randint(0, 40), 150 + rng.randint(0, 60), 60 + rng.randint(0, 30))
                tip = (x + rng.randint(-14, 14), h - gh)
                cv2.line(layer, (x, h), tip, col, 5, cv2.LINE_AA)
                cv2.line(mask, (x, h), tip, 1.0, 5, cv2.LINE_AA)
            for _ in range(22):
                fx, fy = rng.randint(20, W - 20), rng.randint(40, h - 20)
                col = rng.choice([(180, 120, 255), (80, 220, 255), (255, 200, 120), (230, 230, 255)])
                for k in range(5):
                    a = k * 2 * math.pi / 5
                    p = (int(fx + 9 * math.cos(a)), int(fy + 9 * math.sin(a)))
                    cv2.circle(layer, p, 6, col, -1, cv2.LINE_AA)
                    cv2.circle(mask, p, 6, 1.0, -1, cv2.LINE_AA)
                cv2.circle(layer, (fx, fy), 5, (60, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(mask, (fx, fy), 5, 1.0, -1, cv2.LINE_AA)
            grad = np.linspace(0, 0.55, h, dtype=np.float32)[:, None]
            mask = np.maximum(mask * 0.95, grad)
            layer[mask == grad] = (40, 110, 50)
            self._garden = (layer.astype(np.float32), mask[..., None])
        return self._garden

    def draw_scene(self, img):
        W, H = self.W, self.H
        layer, mask = self._garden_layer()
        roi = img[H - layer.shape[0]:H].astype(np.float32)
        img[H - layer.shape[0]:H] = (roi * (1 - mask) + layer * mask).astype(np.uint8)
        # reach line
        y = int(self.hover_height() * H)
        for x in range(30, W - 30, 36):
            cv2.line(img, (x, y), (x + 18, y), (230, 230, 230), 2, cv2.LINE_AA)
        ui.text(img, "reach line", (W - 40, y - 8), 20, C.MUTED, False, "rb")
        for b in self.bubbles:
            draw_bubble(img, b.x, b.y, b.r, b.color, b.state == "hover", 1.0 if b.state == "hover" else 0.45)
        # allowed hand(s)
        for hand in getattr(self, "_hands_cache", []):
            col = self.color if self.allowed(hand) else C.LOCK
            ui.draw_hand(img, hand, col)
            if self.allowed(hand):
                p = (hand.palm * (W, H)).astype(int)
                cv2.circle(img, tuple(p), 26, C.WHITE, 3, cv2.LINE_AA)

    def update(self, frame, tracking, dt):
        self._hands_cache = tracking.hands
        super().update(frame, tracking, dt)

    def current_instruction(self):
        side = {"both": "either hand", "left": "your LEFT hand", "right": "your RIGHT hand"}[self.choice]
        return f"Pop the bubbles with {side}"

    def progress_info(self):
        return "Bubbles", self.pops, self.target

    def draw_demo(self, img, rect, t):
        x, y, w, h = rect
        demo_frame(img, rect)
        cyc = t % 3.0
        bx, by = x + w * 0.62, y + h * 0.32
        start = np.array([x + w * 0.4, y + h * 0.85])
        end = np.array([bx, by + 10])
        k = min(1.0, cyc / 1.6)
        k = 1 - (1 - k) ** 2
        pos = start + (end - start) * k
        if cyc < 1.6:
            draw_bubble(img, bx, by + math.sin(t * 3) * 5, 42, BUBBLE_COLORS[int(t / 3) % 6], True)
        elif cyc < 1.9:
            cv2.circle(img, (int(bx), int(by)), int(42 + (cyc - 1.6) * 200), C.WHITE, 3, cv2.LINE_AA)
            ui.text(img, "POP!", (int(bx), int(by)), 40, C.GOLD, True, "mm", stroke=3)
        world = hand_world(gesture_vector("open"), "Right")
        n = hand_norm(world, ((pos[0] - x) / w, (pos[1] - y) / h), aspect=w / h, scale=2.6)
        put_hand_in_rect(img, n[:, :2], rect, self.color)
        # arm line from shoulder to wrist
        sh = (int(x + w * 0.3), int(y + h - 8))
        cv2.line(img, sh, (int(x + n[0, 0] * w), int(y + n[0, 1] * h)), (120, 110, 100), 10, cv2.LINE_AA)

    def demo_intent(self):
        hover = sorted((b for b in self.bubbles if b.state == "hover"), key=lambda b: b.born)
        hands = []
        if hover:
            b = hover[0]
            side = b.side if self.choice == "both" else self.choice.capitalize()
            hands.append({"side": side, "pos": (b.x / self.W, b.y / self.H), "gesture": "open", "speed": 0.9})
        else:
            side = "Right" if self.choice != "left" else "Left"
            hands.append({"side": side, "pos": (0.62 if side == "Right" else 0.38, 0.8), "gesture": "open"})
        return {"hands": hands}

    # ------------------------------------------------------------------ results
    def accuracy(self):
        return self.pops / max(1, self.pops + self.escaped)

    def rom(self):
        allr = self.reach["Left"] + self.reach["Right"]
        return (100 * max(allr) if allr else 0.0), "max reach height (% of screen)"

    def extra(self):
        def avg(v):
            return round(100 * sum(v) / len(v), 1) if v else 0.0
        return {"pops": self.pops, "escaped": self.escaped, "hand_choice": self.choice,
                "pops_left": self.pops_side.get("Left", 0), "pops_right": self.pops_side.get("Right", 0),
                "reach_left": avg(self.reach["Left"]), "reach_right": avg(self.reach["Right"]),
                "reach_left_max": round(100 * max(self.reach["Left"]), 1) if self.reach["Left"] else 0.0,
                "reach_right_max": round(100 * max(self.reach["Right"]), 1) if self.reach["Right"] else 0.0}

    def summary(self):
        e = self.extra()
        rt = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0
        tot = e["pops_left"] + e["pops_right"]
        bal = f"L {e['pops_left']}  /  R {e['pops_right']}" if tot else "-"
        return [("Bubbles popped", f"{self.pops} / {self.target}"),
                ("Highest reach", f"{self.rom()[0]:.0f}% of screen"),
                ("Average reaction", f"{rt:.2f} s"),
                ("Left / right balance", bal)]


def draw_bubble(img, x, y, r, color, glow=False, opacity=1.0):
    x, y, r = int(x), int(y), int(r)
    H, W = img.shape[:2]
    if x + r < 0 or x - r > W or y + r < 0 or y - r > H:
        return
    if glow:
        ui.radial_glow(img, (x, y), int(r * 1.6), color, 0.35)
    x0, y0, x1, y1 = max(0, x - r - 2), max(0, y - r - 2), min(W, x + r + 2), min(H, y + r + 2)
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.circle(over, (x - x0, y - y0), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.35 * opacity, roi, 1 - 0.35 * opacity, 0, roi)
    if opacity < 1:
        cv2.circle(img, (x, y), r, color, 2, cv2.LINE_AA)
        return
    cv2.circle(img, (x, y), r, color, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), r - 3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.ellipse(img, (x - r // 3, y - r // 3), (r // 4, r // 7), -40, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x + r // 3, y + r // 3), max(2, r // 10), (255, 255, 255), -1, cv2.LINE_AA)
