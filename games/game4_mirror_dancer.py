"""GAME 4 (MEDIUM) - Mirror Dancer: big movements for Parkinson's (LSVT-BIG style) and frozen shoulder.

A glowing ghost dancer shows a target pose over your own body. Match it before
the beat ring closes. Accuracy comes from joint angles (shoulder abduction,
elbow extension, trunk lean, shrug), not from pixel positions.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from core import ui
from core.effects import clamp, ease_out_back
from core.pose_metrics import POSES, features, pose_accuracy, posture_score, NEUTRAL
from core.synth import pose_landmarks
from core.ui import C
from games.base_game import BaseGame, demo_frame

GHOST = (255, 230, 120)
DANCE = (230, 120, 255)


class MirrorDancer(BaseGame):
    id = 4
    name = "Mirror Dancer"
    condition = "Parkinson's disease (small, slow movements), frozen shoulder, posture"
    helps = "Large-amplitude movement, shoulder range, trunk balance, posture"
    body_part = "Upper body"
    difficulty = "MEDIUM"
    needs = frozenset({"pose"})
    color = DANCE
    instructions = ["Sit or stand back so your head, shoulders and arms are visible.",
                    "Copy the glowing dancer's pose - move BIG and bold!",
                    "Match it before the beat ring closes, then hold it briefly."]
    gate_text = "Move back so your upper body is visible"
    lost_text = "I can't see your upper body - please move back a little"
    fail_tip = "Move bigger than feels natural - stretch all the way out. Sitting further back helps the camera."

    def on_start(self):
        c = self.cfg
        self.total = c.G4_TOTAL_POSES
        self.need = max(6, int(round(c.G4_PASS_POSES * self.ease_factor)))
        self.thr = c.G4_POSE_ACCURACY * (1 - 0.5 * c.EASE_STEP * self.ease)
        self.window = c.G4_POSE_WINDOW * (1 + 0.15 * self.ease)
        self.tol = c.G4_TOLERANCE_SCALE * (1 + 0.15 * self.ease)
        self.time_limit = self.total * (self.window + 1.4) + 10
        self.pass_text = f"Match {self.need} of {self.total} poses with at least {int(self.thr * 100)}% accuracy"
        ids = list(POSES)
        rng = random.Random()
        seq = ids + [rng.choice(["arms_up", "t_pose", "hands_head"])]
        rng.shuffle(seq)
        for i in range(1, len(seq)):                    # avoid the same pose twice in a row
            if seq[i] == seq[i - 1]:
                j = (i + 2) % len(seq)
                seq[i], seq[j] = seq[j], seq[i]
        self.seq = seq[:self.total]
        self.i = 0
        self.state = "transition"
        self.state_t = 0.0
        self.pose_t = 0.0
        self.hold = 0.0
        self.acc = 0.0
        self.hint = ""
        self.best_acc = 0.0
        self.outcomes: list = []        # dicts per pose
        self.matched = 0
        self.missed = 0
        self.rom_max: dict = {}
        self.posture: list = []
        self.neck_samples: list = []
        self.neck_baseline = self.calib.get("neck_ratio")
        self.feats = None
        self.user = None
        self._beat = -1
        self.posture_bonus_given = False

    @property
    def pdef(self):
        return POSES[self.seq[min(self.i, len(self.seq) - 1)]]

    def tracking_ok(self, tracking):
        p = tracking.pose
        return p is not None and min(p.vis[11], p.vis[12]) > 0.5

    # ------------------------------------------------------------------ logic
    def on_countdown(self, tracking, dt):
        if self.tracking_ok(tracking):
            f = features(tracking.pose, tracking.aspect)
            self.neck_samples.append(f["neck"])

    def on_update(self, tracking, dt):
        if self.neck_samples and self.neck_baseline is None:
            self.neck_baseline = float(np.median(self.neck_samples))
        self.user = tracking.pose
        self.feats = f = features(tracking.pose, tracking.aspect, self.neck_baseline)
        for k in ("abdL", "abdR"):
            self.rom_max[k] = max(self.rom_max.get(k, 0.0), f[k])
        self.rom_max["leanL"] = max(self.rom_max.get("leanL", 0.0), -f["lean"])
        self.rom_max["leanR"] = max(self.rom_max.get("leanR", 0.0), f["lean"])
        self.state_t += dt
        if self.state == "transition":
            self.posture.append(posture_score(f))
            if self.state_t >= 1.3:
                if self.i >= self.total:
                    return
                self.state, self.state_t, self.pose_t, self.hold, self.best_acc = "pose", 0.0, 0.0, 0.0, 0.0
                self._beat = -1
            return
        # active pose
        self.pose_t += dt
        self.acc, self.hint, _ = pose_accuracy(self.pdef, f, self.tol)
        self.best_acc = max(self.best_acc, self.acc)
        beat = int(self.pose_t)
        if beat != self._beat:
            self._beat = beat
            if self.window - self.pose_t < 3.2:
                self.audio.play("tick")
        if self.acc >= self.thr:
            self.hold += dt
        else:
            self.hold = max(0.0, self.hold - dt * 2)
        if self.hold >= self.cfg.G4_HOLD:
            self._resolve(True)
        elif self.pose_t >= self.window:
            self._resolve(False)

    def _resolve(self, ok):
        f = self.feats
        rec = {"pose": self.pdef.id, "matched": ok, "accuracy": round(self.best_acc, 3), "time": round(self.pose_t, 2),
               "rom": {k: round(f[k], 1) for k in self.pdef.rom}}
        self.outcomes.append(rec)
        cx, cy = self._ghost_center()
        if ok:
            self.matched += 1
            self.reaction_times.append(self.pose_t)
            speed = int(60 * clamp(1 - self.pose_t / self.window))
            self.add_points(int(100 * self.acc) + speed, cx, cy - 160, DANCE)
            self.fx.burst(cx, cy - 60, [DANCE, GHOST, C.WHITE], 50, 420, 0.9, 7, 150)
            self.fx.shockwave(cx, cy - 60, GHOST, 260)
            self.fx.shake(7, 0.2)
            self.audio.play("chime")
            if self.combo in (3, 6, 9):
                self.big_banner(f"{self.combo} in a row!")
        else:
            self.missed += 1
            self.break_combo()
            self.audio.play("miss")
            self.say("Missed that one - next pose coming!", 1.6)
        self.i += 1
        self.state, self.state_t = "transition", 0.0
        if self.i >= self.total and not self.posture_bonus_given:
            self.posture_bonus_given = True
            bonus = int(200 * self.posture_avg())
            if bonus:
                self.score += bonus
                self.fx.float_text(f"Posture bonus +{bonus}", self.W // 2, self.H // 2 + 80, C.TEAL, 36, 1.8)

    def posture_avg(self):
        return float(np.mean(self.posture)) if self.posture else 0.0

    def is_passed(self):
        return self.i >= self.total and self.matched >= self.need

    def is_failed(self):
        return self.missed > self.total - self.need or (self.i >= self.total and self.matched < self.need)

    # ------------------------------------------------------------------ drawing
    def _ghost_center(self):
        if self.user is not None:
            p = self.user.norm
            return int((p[11, 0] + p[12, 0]) / 2 * self.W), int((p[11, 1] + p[12, 1]) / 2 * self.H)
        return self.W // 2, int(self.H * 0.55)

    def _ghost_points(self, spec):
        """Target pose scaled and placed so its shoulders sit on the user's shoulders."""
        aspect = self.W / self.H
        g = pose_landmarks(spec, aspect, (0.5, 0.55), 0.25)
        if self.user is None:
            return g
        u = self.user.norm
        uw = np.linalg.norm((u[11, :2] - u[12, :2]) * (aspect, 1))
        s = clamp(uw / 0.25, 0.5, 2.5)
        umid = (u[11, :2] + u[12, :2]) / 2
        gmid = (g[11, :2] + g[12, :2]) / 2
        out = g.copy()
        out[:, 0] = umid[0] + (g[:, 0] - gmid[0]) * s
        out[:, 1] = umid[1] + (g[:, 1] - gmid[1]) * s
        return out

    def draw_scene(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.35, (60, 20, 40))
        # stage spotlights
        for i, x in enumerate((W * 0.2, W * 0.8)):
            a = 0.2 + 0.1 * math.sin(self.t * 2 + i * 2)
            ui.radial_glow(img, (int(x), 0), 380, (200, 120, 255), a)
        active = self.state == "pose" and self.phase == "play"
        if active:
            gp = self._ghost_points(self.pdef.spec)
            ui.draw_pose(img, gp, GHOST, 12, 22, core_color=(255, 245, 210))
        if self.user is not None:
            good = active and self.acc >= self.thr
            ui.draw_pose(img, self.user.norm, C.GOOD if good else DANCE, 7, 10, vis=self.user.vis)
        self._card(img)
        self._dots(img)

    def _card(self, img):
        W = self.W
        x, y, w, h = W - 330, 112, 306, 470
        ui.panel(img, (x, y, w, h), C.PANEL_DARK, 0.86, 26, border=DANCE)
        if self.i >= self.total:
            ui.text(img, "Finished!", (x + w // 2, y + 60), 36, C.TEXT, True, "mm")
            return
        is_pose = self.state == "pose"
        ui.text(img, f"Pose {self.i + 1} of {self.total}" + ("" if is_pose else "  -  next:"), (x + 20, y + 16), 20,
                C.MUTED, True)
        ui.text(img, self.pdef.name, (x + w // 2, y + 64), 34, C.TEXT, True, "mm", stroke=2)
        # mini silhouette + beat ring
        cx, cy, R = x + w // 2, y + 200, 92
        spec = self.pdef.spec
        aspect = 1.0
        g = pose_landmarks(spec, aspect, (0.5, 0.5), 0.3)
        mini = np.zeros_like(g)
        mini[:, 0] = cx + (g[:, 0] - 0.5) * 220
        mini[:, 1] = cy + (g[:, 1] - 0.5) * 220
        H = self.H
        mini_n = mini.copy()
        mini_n[:, 0] /= W
        mini_n[:, 1] /= H
        ui.draw_pose(img, mini_n, GHOST, 6, 8)
        if is_pose:
            left = clamp(1 - self.pose_t / self.window)
            pulse = 1 + 0.06 * max(0.0, 1 - (self.pose_t % 1.0) * 4)
            col = C.GOOD if left > 0.5 else (C.GOLD if left > 0.25 else C.BAD)
            cv2.circle(img, (cx, cy), int((R + 40 * left) * pulse), col, 5, cv2.LINE_AA)
            ui.text(img, f"{self.window - self.pose_t:.0f}s", (cx + R, cy - R), 22, col, True, "mm", stroke=2)
            ui.paragraph(img, self.pdef.say, (x + 20, y + 318), w - 40, 20, C.GOLD, True)
            ok = self.acc >= self.thr
            ui.text(img, "Accuracy", (x + 20, y + 380), 20, C.TEXT, True)
            ui.progress_bar(img, (x + 20, y + 406, w - 40, 24), self.acc, C.GOOD if ok else C.WARN, marker=self.thr,
                            label=f"{self.acc * 100:.0f}%")
            if ok:
                ui.progress_bar(img, (x + 20, y + 436, w - 40, 8), self.hold / self.cfg.G4_HOLD, C.GOOD)
        else:
            ui.text(img, "Get ready...", (x + w // 2, y + 330), 26, C.GOLD, True, "mm")
        # posture
        if self.feats is not None:
            ps = posture_score(self.feats)
            ui.text(img, f"Posture {ps * 100:.0f}%", (x + 20, y + h + 12), 20, C.TEAL if ps > 0.7 else C.WARN, True)

    def _dots(self, img):
        n = self.total
        for k in range(n):
            x = 40 + k * 34
            y = self.H - 130
            if k < len(self.outcomes):
                ok = self.outcomes[k]["matched"]
                cv2.circle(img, (x, y), 13, C.GOOD if ok else C.BAD, -1, cv2.LINE_AA)
                (ui.check if ok else ui.cross)(img, (x, y + 1), 12, C.WHITE, 3)
            else:
                cv2.circle(img, (x, y), 13, (90, 80, 80), 2, cv2.LINE_AA)
        ui.text(img, f"matched {self.matched}   need {self.need}", (40, self.H - 104), 20, C.MUTED, True)

    def current_instruction(self):
        if self.state == "pose":
            return self.hint if self.acc < self.thr else "Great - hold it!"
        return "Relax... sit tall, shoulders level"

    def countdown_tip(self):
        return "Sit tall and relax your shoulders"

    def progress_info(self):
        return "Poses matched", self.matched, self.need

    def draw_demo(self, img, rect, t):
        x, y, w, h = rect
        demo_frame(img, rect)
        order = ["arms_up", "t_pose", "reach_left", "hands_head", "lean_right", "elbows_ribs"]
        k = int(t / 2.0) % len(order)
        local = (t % 2.0) / 2.0
        a = clamp(local * 2.5)
        tgt = POSES[order[k]].spec
        spec = {key: NEUTRAL.get(key, 0) + (tgt[key] - NEUTRAL.get(key, 0)) * a for key in tgt}
        g = pose_landmarks(spec, w / h, (0.5, 0.48), 0.22)
        g[:, 0] = (x + g[:, 0] * w) / self.W
        g[:, 1] = (y + g[:, 1] * h) / self.H
        ui.draw_pose(img, g, GHOST, 8, 12)
        ui.text(img, POSES[order[k]].name, (x + w // 2, y + h - 36), 30, C.TEXT, True, "mm", stroke=2)

    def demo_intent(self):
        if self.state == "pose" and self.pose_t > 0.7:
            return {"pose": {"spec": self.pdef.spec}}
        return {"pose": {"spec": {}}}

    # ------------------------------------------------------------------ results
    def stars(self):
        if not self.success:
            return 0
        return 1 + int(self.matched >= self.total - 1) + int(self.matched == self.total and self.accuracy() >= 0.85)

    def accuracy(self):
        return float(np.mean([o["accuracy"] for o in self.outcomes])) if self.outcomes else 0.0

    def rom(self):
        return max(self.rom_max.get("abdL", 0), self.rom_max.get("abdR", 0)), "max shoulder abduction (deg)"

    def extra(self):
        return {"matched": self.matched, "total": self.total,
                "abd_left_max": round(self.rom_max.get("abdL", 0), 1), "abd_right_max": round(self.rom_max.get("abdR", 0), 1),
                "lean_left_max": round(self.rom_max.get("leanL", 0), 1), "lean_right_max": round(self.rom_max.get("leanR", 0), 1),
                "posture": round(self.posture_avg(), 3),
                "mean_match_time_s": round(float(np.mean(self.reaction_times)), 2) if self.reaction_times else 0.0,
                "poses": self.outcomes}

    def summary(self):
        e = self.extra()
        return [("Poses matched", f"{self.matched} / {self.total}"),
                ("Shoulder lift L / R", f"{e['abd_left_max']:.0f} / {e['abd_right_max']:.0f} deg"),
                ("Side lean L / R", f"{e['lean_left_max']:.0f} / {e['lean_right_max']:.0f} deg"),
                ("Average time to match", f"{e['mean_match_time_s']:.1f} s"),
                ("Posture score", f"{e['posture'] * 100:.0f}%")]
