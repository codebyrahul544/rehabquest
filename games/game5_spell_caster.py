"""GAME 5 (HARD) - Spell Caster: finger individuation, opposition and two-hand coordination.

A spellbook shows a sequence of finger "runes" (thumb-to-finger taps, single
finger raises, counting, and finally two different gestures with two hands).
Memorise it, then cast it in order before each rune's timer runs out. The
sequences get longer every round (memory + motor planning, "Simon says").
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from core import ui
from core.effects import clamp, ease_out_back
from core.gestures import FINGER_ORDER, GESTURE_INFO, matches, parse_dual
from core.ui import C
from games.base_game import BaseGame, demo_frame, put_hand_in_rect
from core.synth import gesture_vector, hand_norm, hand_world

MAGIC = (255, 120, 190)
MAGIC2 = (255, 220, 120)
DUALS = ["dual:open+fist", "dual:fist+open", "dual:peace+fist", "dual:point+open"]
ISOLATION = ["point", "pinky_only", "peace", "pinch_index", "pinch_middle", "pinch_ring", "pinch_pinky"]


def rune_name(g: str) -> str:
    d = parse_dual(g)
    if d:
        return f"L: {GESTURE_INFO[d[0]][0]}  +  R: {GESTURE_INFO[d[1]][0]}"
    return GESTURE_INFO[g][0]


def rune_say(g: str) -> str:
    d = parse_dual(g)
    if d:
        return f"BOTH hands: LEFT {GESTURE_INFO[d[0]][0].lower()}, RIGHT {GESTURE_INFO[d[1]][0].lower()}"
    return GESTURE_INFO[g][1]


def rune_fingers(g: str) -> list:
    d = parse_dual(g)
    if d:
        return sorted(set(GESTURE_INFO[d[0]][2]) | set(GESTURE_INFO[d[1]][2]))
    return GESTURE_INFO[g][2]


class SpellCaster(BaseGame):
    id = 5
    name = "Spell Caster"
    condition = "Advanced stroke recovery, hand surgery, carpal tunnel rehab"
    helps = "Finger individuation, thumb opposition, dexterity, two-hand coordination, memory"
    body_part = "Fingers (both hands)"
    difficulty = "HARD"
    needs = frozenset({"hands"})
    color = MAGIC
    instructions = ["Watch the spellbook: memorise the order of the finger runes.",
                    "Cast each rune with your fingers, in order, before its timer ends.",
                    "Spells get longer each round. The last one needs BOTH hands!"]
    fail_tip = "Hold each finger shape still for half a second. Keep your palm facing the camera."

    def on_start(self):
        c = self.cfg
        self.rounds = c.G5_ROUNDS
        self.max_mistakes = c.G5_MAX_MISTAKES + self.ease
        self.pass_text = f"Cast {self.rounds} spells with no more than {self.max_mistakes} mistakes"
        self.time_limit = 300.0
        rng = random.Random()
        plans = [["pinch_index", "pinch_middle"],
                 ["pinch_ring", "pinch_pinky", "point"],
                 rng.sample(ISOLATION, 4),
                 ["count_1", "count_2", "count_3", "count_4", "count_5"],
                 rng.sample(ISOLATION, 4) + [rng.choice(DUALS)]]
        while len(plans) < self.rounds:
            plans.append(rng.sample(ISOLATION, 4) + [rng.choice(DUALS)])
        self.plans = plans[:self.rounds]
        self.round = 0
        self.step = 0
        self.state = "show"
        self.state_t = 0.0
        self.step_t = 0.0
        self.hold = 0.0
        self.mistakes = 0
        self.marks: list = []            # per step in current round: None | "ok" | "miss"
        self.finger_stats = {f: {"tries": 0, "ok": 0, "time": []} for f in FINGER_ORDER}
        self.step_log: list = []
        self.correct = 0
        self.hands = []
        self.trail: list = []
        self._new_round()

    def _new_round(self):
        self.seq = self.plans[self.round]
        self.marks = [None] * len(self.seq)
        self.step = 0
        self.state = "show"
        self.state_t = 0.0

    @property
    def show_time(self):
        return 1.2 + 0.9 * len(self.seq)

    @property
    def step_time(self):
        k = self.round / max(1, self.rounds - 1)
        base = self.cfg.G5_STEP_TIME_START + (self.cfg.G5_STEP_TIME_END - self.cfg.G5_STEP_TIME_START) * k
        dual_bonus = 3.0 if parse_dual(self.seq[min(self.step, len(self.seq) - 1)]) else 0.0
        return base * (1 + 0.15 * self.ease) + dual_bonus

    # ------------------------------------------------------------------ logic
    def check(self, gesture, tracking) -> bool:
        d = parse_dual(gesture)
        if d:
            if len(tracking.hands) < 2:
                return False
            left, right = tracking.screen_sorted_hands()[:2]
            return matches(d[0], left.world, self.cfg) and matches(d[1], right.world, self.cfg)
        return any(matches(gesture, h.world, self.cfg) for h in tracking.hands)

    def on_update(self, tracking, dt):
        self.hands = tracking.hands
        for h in tracking.hands:
            tip = h.norm[8, :2] * (self.W, self.H)
            self.trail.append([tip[0], tip[1], 0.5])
        for p in self.trail:
            p[2] -= dt
        self.trail = [p for p in self.trail if p[2] > 0][-60:]
        self.state_t += dt
        if self.state == "show":
            if self.state_t >= self.show_time:
                self.state, self.state_t, self.step_t, self.hold = "cast", 0.0, 0.0, 0.0
                self.audio.play("go")
            return
        if self.state == "round_done":
            if self.state_t >= 2.0:
                self.round += 1
                if self.round < self.rounds:
                    self._new_round()
            return
        # cast
        g = self.seq[self.step]
        self.step_t += dt
        if self.check(g, tracking):
            self.hold += dt
        else:
            self.hold = max(0.0, self.hold - dt * 2)
        if self.hold >= self.cfg.G5_HOLD:
            self._step_done(True)
        elif self.step_t >= self.step_time:
            self._step_done(False)

    def _step_done(self, ok):
        g = self.seq[self.step]
        for f in rune_fingers(g):
            st = self.finger_stats[f]
            st["tries"] += 1
            if ok:
                st["ok"] += 1
                st["time"].append(self.step_t)
        self.step_log.append({"gesture": g, "ok": ok, "time": round(self.step_t, 2), "round": self.round + 1})
        self.marks[self.step] = "ok" if ok else "miss"
        x, y = self._hand_pos()
        if ok:
            self.correct += 1
            self.reaction_times.append(self.step_t)
            self.add_points(80 + int(40 * clamp(1 - self.step_t / self.step_time)), x, y, MAGIC2)
            self.fx.sparkle(x, y, MAGIC2, 22, 220)
            self.audio.play("chime")
        else:
            self.mistakes += 1
            self.break_combo()
            self.audio.play("miss")
            self.fx.flash_screen(C.BAD, 0.25)
            left = self.max_mistakes - self.mistakes
            self.say(f"Missed: {rune_name(g)}  -  {max(0, left)} mistakes left", 2.2)
        self.step += 1
        self.step_t = 0.0
        self.hold = 0.0
        if self.step >= len(self.seq):
            self._round_done()

    def _round_done(self):
        self.state, self.state_t = "round_done", 0.0
        cx, cy = self.W * 0.66, self.H * 0.45
        self.fx.burst(cx, cy, [MAGIC, MAGIC2, C.WHITE, C.ACCENT], 90, 520, 1.3, 8, 40)
        for k in range(3):
            self.fx.shockwave(cx, cy, [MAGIC, MAGIC2, C.ACCENT][k], 160 + 90 * k, 0.5 + 0.2 * k)
        self.fx.shake(12, 0.35)
        self.audio.play("magic")
        self.big_banner(f"Spell {self.round + 1} cast!", 1.6)

    def _hand_pos(self):
        if self.hands:
            p = self.hands[0].norm[8, :2]
            return float(p[0] * self.W), float(p[1] * self.H)
        return self.W * 0.66, self.H * 0.45

    def is_passed(self):
        return self.round >= self.rounds and self.mistakes <= self.max_mistakes

    def is_failed(self):
        return self.mistakes > self.max_mistakes

    # ------------------------------------------------------------------ drawing
    def draw_scene(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.5, (70, 20, 50))
        for x, y, life in self.trail:
            cv2.circle(img, (int(x), int(y)), int(3 + 10 * life), MAGIC2, -1, cv2.LINE_AA)
        for h in self.hands:
            ui.draw_hand(img, h, MAGIC)
        self._book(img)
        self._crystals(img)

    def _book(self, img):
        x, y, w, h = 24, 112, 500, 470
        ui.panel(img, (x - 6, y - 6, w + 12, h + 12), (40, 60, 110), 0.95, 26)
        ui.rounded_rect(img, (x, y, w // 2 - 4, h), 18, (170, 215, 235), -1)
        ui.rounded_rect(img, (x + w // 2 + 4, y, w // 2 - 4, h), 18, (170, 215, 235), -1)
        cv2.line(img, (x + w // 2, y + 10), (x + w // 2, y + h - 10), (90, 120, 150), 3, cv2.LINE_AA)
        ink = (60, 40, 50)
        r = min(self.round, self.rounds - 1)
        ui.text(img, f"Spell {r + 1} of {self.rounds}", (x + w // 2, y + 22), 26, ink, True, "mm")
        n = len(self.seq)
        showing = self.state == "show"
        revealed = int(self.state_t / 0.9) + 1 if showing else n
        cols = 2
        rows = math.ceil(n / cols)
        cell_h = min(120, (h - 70) // rows)
        for i, g in enumerate(self.seq):
            col, row = i // rows, i % rows
            cx = x + w // 4 + col * (w // 2)
            cy = y + 60 + row * cell_h + cell_h // 2
            visible = showing and i < revealed
            current = self.state == "cast" and i == self.step
            mark = self.marks[i]
            hint_now = current and self.step_t > self.step_time * 0.5
            rr = min(46, cell_h // 2 - 6)
            if current:
                ui.radial_glow(img, (cx, cy), rr + 40, MAGIC, 0.6)
                ui.ring(img, (cx, cy), rr + 8, 1 - self.step_t / self.step_time, MAGIC, 6, (120, 100, 110))
            cv2.circle(img, (cx, cy), rr, (240, 245, 250) if (visible or mark or hint_now) else (120, 140, 160), -1,
                       cv2.LINE_AA)
            cv2.circle(img, (cx, cy), rr, ink, 2, cv2.LINE_AA)
            if visible or mark or hint_now:
                d = parse_dual(g)
                if d:
                    ui.hand_icon(img, d[0], (cx - rr // 2, cy), rr, ink, "Left", 2)
                    ui.hand_icon(img, d[1], (cx + rr // 2, cy), rr, ink, "Right", 2)
                else:
                    ui.hand_icon(img, g if not g.startswith("count_") else
                                 {"count_1": "point", "count_2": "peace"}.get(g, g), (cx, cy), int(rr * 1.4), ink)
                label = rune_name(g) if not parse_dual(g) else "Two hands"
                ui.text(img, label, (cx, cy + rr + 12), 16, ink, True, "mm")
            else:
                ui.text(img, "?", (cx, cy), 44, (60, 70, 90), True, "mm")
                ui.text(img, str(i + 1), (cx, cy + rr + 12), 16, ink, True, "mm")
            if mark == "ok":
                ui.check(img, (cx + rr - 6, cy - rr + 8), 26, C.GOOD, 6)
            elif mark == "miss":
                ui.cross(img, (cx + rr - 6, cy - rr + 8), 24, C.BAD, 6)
        if showing:
            ui.text(img, "Memorise the runes!", (x + w // 2, y + h - 22), 24, (40, 30, 160), True, "mm")

    def _crystals(self, img):
        x0 = self.W - 60
        y = 140
        ui.text(img, "mistakes left", (x0 + 20, y - 18), 18, C.MUTED, True, "rm")
        for k in range(self.max_mistakes):
            x = x0 - k * 40
            alive = k < self.max_mistakes - self.mistakes
            pts = np.array([(x, y), (x + 13, y + 20), (x, y + 40), (x - 13, y + 20)], np.int32)
            cv2.fillPoly(img, [pts], MAGIC if alive else (70, 60, 70), cv2.LINE_AA)
            cv2.polylines(img, [pts], True, C.WHITE if alive else (110, 100, 110), 2, cv2.LINE_AA)

    def current_instruction(self):
        if self.state == "show":
            return "Watch and memorise the spell..."
        if self.state == "round_done":
            return "Relax your hands..."
        g = self.seq[self.step]
        if self.step_t > self.step_time * 0.5:
            return f"Rune {self.step + 1}: {rune_say(g)}"
        return f"Cast rune {self.step + 1} of {len(self.seq)} from memory!"

    def progress_info(self):
        return "Spells", min(self.round, self.rounds), self.rounds

    def draw_hud(self, img):
        super().draw_hud(img)
        if self.state == "cast" and self.phase == "play":
            p = self.hold / self.cfg.G5_HOLD
            if p > 0:
                ui.progress_bar(img, (self.W // 2 - 150, self.H - 112, 300, 12), p, MAGIC2)

    def draw_demo(self, img, rect, t):
        x, y, w, h = rect
        demo_frame(img, rect)
        order = ["pinch_index", "pinch_middle", "pinch_ring", "pinch_pinky", "point", "peace"]
        k = int(t / 1.6) % len(order)
        local = (t % 1.6) / 1.6
        a = clamp(local * 3)
        vec = gesture_vector("open") * (1 - a) + gesture_vector(order[k]) * a
        n = hand_norm(hand_world(vec, "Right"), (0.5, 0.5), aspect=w / h, scale=3.2)
        put_hand_in_rect(img, n[:, :2], rect, MAGIC)
        if a >= 1:
            tip = n[4, :2] * (w, h) + (x, y)
            ui.radial_glow(img, tip, 40, MAGIC2, 0.8)
        ui.text(img, GESTURE_INFO[order[k]][0], (x + w // 2, y + h - 36), 30, C.TEXT, True, "mm", stroke=2)

    def demo_intent(self):
        if self.state != "cast" or self.step_t < 0.6:
            return {"hands": [{"side": "Right", "pos": (0.68, 0.55), "gesture": "relaxed"}]}
        g = self.seq[self.step]
        d = parse_dual(g)
        if d:
            return {"hands": [{"side": "Left", "pos": (0.52, 0.55), "gesture": d[0]},
                              {"side": "Right", "pos": (0.82, 0.55), "gesture": d[1]}]}
        return {"hands": [{"side": "Right", "pos": (0.68, 0.55), "gesture": g}]}

    # ------------------------------------------------------------------ results
    def stars(self):
        if not self.success:
            return 0
        return 1 + int(self.mistakes <= 1) + int(self.mistakes == 0)

    def accuracy(self):
        return self.correct / max(1, self.correct + self.mistakes)

    def rom(self):
        rates = self.finger_rates()
        return (100 * min(rates.values()) if rates else 0.0), "weakest finger success (%)"

    def finger_rates(self):
        return {f: s["ok"] / s["tries"] for f, s in self.finger_stats.items() if s["tries"]}

    def weakest(self):
        rates = self.finger_rates()
        if not rates:
            return "-"
        times = {f: (np.mean(s["time"]) if s["time"] else 99) for f, s in self.finger_stats.items() if s["tries"]}
        return min(rates, key=lambda f: (rates[f], -times[f]))

    def extra(self):
        return {"rounds": min(self.round, self.rounds), "mistakes": self.mistakes,
                "fingers": {f: {"tries": s["tries"], "ok": s["ok"],
                                "mean_time_s": round(float(np.mean(s["time"])), 2) if s["time"] else None}
                            for f, s in self.finger_stats.items()},
                "weakest_finger": self.weakest(), "steps": self.step_log}

    def summary(self):
        times = {f: np.mean(s["time"]) for f, s in self.finger_stats.items() if s["time"]}
        slow = max(times, key=times.get) if times else "-"
        return [("Spells cast", f"{min(self.round, self.rounds)} / {self.rounds}"),
                ("Mistakes", f"{self.mistakes} (max {self.max_mistakes})"),
                ("Weakest finger", self.weakest()),
                ("Slowest finger", f"{slow} ({times[slow]:.1f} s)" if times else "-"),
                ("Runes correct", f"{self.correct}")]
