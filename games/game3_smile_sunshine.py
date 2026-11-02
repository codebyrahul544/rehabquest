"""GAME 3 (EASY) - Smile Sunshine: facial exercises for Bell's palsy and post-stroke facial droop.

A cartoon sun shows an expression; the patient copies it and holds it for 2 s.
The live SYMMETRY bar (left vs right side) is the key clinical metric.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from core import ui
from core.effects import clamp, ease_out_back
from core.face_metrics import EXPRESSIONS, FACE_OVAL, LIPS_OUTER, expression_ok, sides
from core.ui import C
from games.base_game import BaseGame, demo_frame

SUN = (40, 200, 255)
SUN_DARK = (20, 140, 240)

# What the virtual patient (and the sun) does for each exercise. Slight right-side weakness = realistic.
DEMO_BLEND = {
    "smile": {"mouthSmileLeft": 0.82, "mouthSmileRight": 0.70},
    "brows": {"browInnerUp": 0.7, "browOuterUpLeft": 0.62, "browOuterUpRight": 0.52},
    "eyes": {"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.75},
    "pucker": {"mouthPucker": 0.8},
    "jaw": {"jawOpen": 0.65},
    "half_smile": {"mouthSmileLeft": 0.72, "mouthSmileRight": 0.12},
}


class SmileSunshine(BaseGame):
    id = 3
    name = "Smile Sunshine"
    condition = "Bell's palsy / facial nerve weakness, stroke facial droop"
    helps = "Facial muscle strength and left-right symmetry"
    body_part = "Face"
    difficulty = "EASY"
    needs = frozenset({"face"})
    color = SUN
    instructions = ["Look at the camera. The sun shows you a face exercise.",
                    "Copy it and HOLD for 2 seconds - the sun's rays grow.",
                    "Try to move both sides of your face evenly (watch the symmetry bar)."]
    gate_text = "Look at the camera to start"
    lost_text = "I can't see your face - please look at the camera"
    fail_tip = "Exaggerate each expression a little and hold it still. Use a mirror-like slow movement."

    def on_start(self):
        c = self.cfg
        self.time_limit = c.G3_TIME_LIMIT * (1 + 0.15 * self.ease)
        self.pass_text = f"Complete all {len(EXPRESSIONS)} expressions (each held 2 s) within {int(self.time_limit)} s"
        self.idx = 0
        self.hold = 0.0
        self.expr_t = 0.0
        self.extra_ease = 1.0
        self.hinted = False
        self.transition = 0.0
        self.records: list = []
        self.samples: list = []
        self.strength = 0.0
        self.sym = None
        self.lr = (0.0, 0.0)
        self.ok_s = self.ok_y = False
        self.face = None
        self.ray_burst = 0.0

    @property
    def expr(self):
        return EXPRESSIONS[min(self.idx, len(EXPRESSIONS) - 1)]

    def tracking_ok(self, tracking):
        return tracking.face is not None

    # ------------------------------------------------------------------ logic
    def on_update(self, tracking, dt):
        self.face = tracking.face
        self.ray_burst = max(0.0, self.ray_burst - dt)
        if self.transition > 0:
            self.transition -= dt
            return
        if self.idx >= len(EXPRESSIONS) or self.face is None:
            return
        self.expr_t += dt
        e = self.expr
        ease = self.ease_factor * self.extra_ease
        self.ok_s, self.ok_y, self.strength, self.sym = expression_ok(e["id"], self.face, self.cfg, ease, tracking.aspect)
        self.lr = sides(e["id"], self.face, tracking.aspect)
        if self.ok_s and self.ok_y:
            self.hold += dt
            if e["id"] != "half_smile":
                self.samples.append(self.sym)
        else:
            self.hold = max(0.0, self.hold - dt * 1.2)
        if self.expr_t > 20 and not self.hinted:
            self.hinted = True
            self.extra_ease = 0.85
            self.say("Tip: " + e["tip"], 4)
        if self.hold >= self.cfg.G3_HOLD:
            self._complete()

    def _complete(self):
        e = self.expr
        sym = float(np.mean(self.samples)) if self.samples else None
        self.records.append({"id": e["id"], "name": e["name"], "symmetry": sym, "time": round(self.expr_t, 2),
                             "strength": round(self.strength, 3), "control": round(self.sym, 3)
                             if e["id"] == "half_smile" else None})
        self.reaction_times.append(self.expr_t)
        bonus = int(50 * (sym if sym is not None else 1.0))
        cx, cy = self.sun_pos()
        self.add_points(100 + bonus, cx, cy + 150, SUN)
        self.fx.burst(cx, cy, [SUN, C.GOLD, C.WHITE], 46, 380, 1.0, 7, 60)
        self.fx.shockwave(cx, cy, SUN, 220)
        self.audio.play("chime")
        self.ray_burst = 1.0
        self.idx += 1
        self.hold, self.expr_t, self.samples, self.hinted, self.extra_ease = 0.0, 0.0, [], False, 1.0
        self.transition = 1.2
        if self.idx < len(EXPRESSIONS):
            self.big_banner(["Lovely!", "Beautiful!", "Super!", "Brilliant!", "Great job!"][self.idx % 5], 1.1)

    def is_passed(self):
        return self.idx >= len(EXPRESSIONS)

    def sun_pos(self):
        return self.W - 210, 330

    # ------------------------------------------------------------------ drawing
    def draw_scene(self, img):
        W, H = self.W, self.H
        ui.darken(img, 0.18, (60, 150, 230))          # warm sunny tint
        if self.face is not None:
            p = np.column_stack([self.face.norm[:, 0] * W, self.face.norm[:, 1] * H]).astype(np.int32)
            cv2.polylines(img, [p[FACE_OVAL]], True, (200, 230, 255), 2, cv2.LINE_AA)
            cv2.polylines(img, [p[LIPS_OUTER]], True, (150, 150, 255), 2, cv2.LINE_AA)
        done = self.idx >= len(EXPRESSIONS)
        target = DEMO_BLEND[self.expr["id"]] if not done else DEMO_BLEND["smile"]
        prog = self.hold / self.cfg.G3_HOLD
        cx, cy = self.sun_pos()
        draw_sun(img, (cx, cy), 120, target, prog + self.ray_burst, self.t)
        ui.ring(img, (cx, cy), 205, prog, C.GOLD, 10, None)
        self._panel(img)
        # progress suns
        n = len(EXPRESSIONS)
        for i in range(n):
            x = W - 400 + i * 66
            col = SUN if i < self.idx else (90, 90, 90)
            cv2.circle(img, (x, H - 150), 22, col, -1, cv2.LINE_AA)
            if i < self.idx:
                ui.check(img, (x, H - 148), 22, (30, 30, 30), 4)
            elif i == self.idx:
                cv2.circle(img, (x, H - 150), 27, C.WHITE, 3, cv2.LINE_AA)

    def _panel(self, img):
        x, y, w, h = 24, 112, 470, 470
        ui.panel(img, (x, y, w, h), C.PANEL_DARK, 0.84, 26, border=SUN)
        e = self.expr
        ui.text(img, f"Exercise {min(self.idx + 1, len(EXPRESSIONS))} of {len(EXPRESSIONS)}", (x + 24, y + 18), 22, C.MUTED, True)
        ui.text(img, e["name"], (x + 24, y + 48), 44, C.TEXT, True, stroke=2)
        ui.paragraph(img, e["say"], (x + 24, y + 110), w - 48, 26, C.GOLD, True)
        # strength
        thr = getattr(self.cfg, e["thr"]) * self.ease_factor * self.extra_ease
        yy = y + 190
        ui.text(img, "Strength", (x + 24, yy), 22, C.TEXT, True)
        self._status(img, (x + w - 30, yy + 12), self.ok_s, "strong enough" if self.ok_s else "a bit more")
        ui.progress_bar(img, (x + 24, yy + 34, w - 48, 24), clamp(self.strength / max(0.05, thr * 1.6)),
                        C.GOOD if self.ok_s else SUN, marker=1 / 1.6)
        # symmetry / control
        yy += 84
        if e["id"] == "half_smile":
            ui.text(img, "One-side control", (x + 24, yy), 22, C.TEXT, True)
            val = self.sym or 0.0
            goal = self.cfg.G3_HALF_SMILE_DIFF * self.ease_factor * self.extra_ease
            self._status(img, (x + w - 30, yy + 12), self.ok_y, "good control" if self.ok_y else "keep other side still")
            ui.progress_bar(img, (x + 24, yy + 34, w - 48, 24), clamp(val / 0.6), C.GOOD if self.ok_y else C.WARN,
                            marker=goal / 0.6, label=f"{val * 100:.0f}%")
        else:
            sym = self.sym if self.sym is not None else 0.0
            sthr = self.cfg.G3_SYMMETRY_THRESHOLD * self.ease_factor * self.extra_ease
            ui.text(img, "Symmetry (left vs right)", (x + 24, yy), 22, C.TEXT, True)
            self._status(img, (x + w - 30, yy + 12), self.ok_y, "even" if self.ok_y else "lift weaker side")
            ui.progress_bar(img, (x + 24, yy + 34, w - 48, 24), sym, C.GOOD if self.ok_y else C.WARN, marker=sthr,
                            label=f"{sym * 100:.0f}%")
        # L / R bars
        yy += 86
        l, r = self.lr
        top = max(l, r, 0.05)
        for i, (lab, v) in enumerate((("Your left", l), ("Your right", r))):
            bx = x + 24 + i * ((w - 48) // 2 + 6)
            bw = (w - 48) // 2 - 6
            ui.text(img, lab, (bx, yy), 20, C.MUTED, True)
            ui.progress_bar(img, (bx, yy + 26, bw, 18), v / top, C.ACCENT)
        ui.paragraph(img, e["tip"], (x + 24, yy + 64), w - 48, 20, C.MUTED)

    def _status(self, img, pos, ok, label):
        x, y = pos
        tw, _ = ui.text_size(label, 18, True)
        if ok:
            ui.check(img, (x - tw - 20, y), 18, C.GOOD, 3)
        else:
            ui.warn_icon(img, (x - tw - 20, y), 20)
        ui.text(img, label, (x, y), 18, C.GOOD if ok else C.WARN, True, "rm")

    def current_instruction(self):
        if self.transition > 0 or self.idx >= len(EXPRESSIONS):
            return "Relax your face..."
        return self.expr["say"]

    def countdown_tip(self):
        return "Relax your face and look at the camera"

    def progress_info(self):
        return "Expressions", self.idx, len(EXPRESSIONS)

    def draw_demo(self, img, rect, t):
        x, y, w, h = rect
        demo_frame(img, rect)
        k = int(t / 2.5) % len(EXPRESSIONS)
        e = EXPRESSIONS[k]
        local = (t % 2.5) / 2.5
        amt = clamp(local * 3) * clamp((1 - local) * 6)
        blend = {key: v * amt for key, v in DEMO_BLEND[e["id"]].items()}
        draw_sun(img, (x + w // 2, y + h // 2 - 10), 110, blend, amt, t)
        ui.text(img, e["name"], (x + w // 2, y + h - 36), 30, C.TEXT, True, "mm", stroke=2)

    def demo_intent(self):
        if self.transition > 0 or self.idx >= len(EXPRESSIONS) or self.expr_t < 0.5:
            return {"face": {"blend": {}}}
        return {"face": {"blend": DEMO_BLEND[self.expr["id"]]}}

    # ------------------------------------------------------------------ results
    def accuracy(self):
        syms = [r["symmetry"] for r in self.records if r["symmetry"] is not None]
        return float(np.mean(syms)) if syms else 0.0

    def rom(self):
        return 100 * self.accuracy(), "mean facial symmetry (%)"

    def extra(self):
        return {"completed": self.idx,
                "symmetry": {r["id"]: (round(r["symmetry"], 3) if r["symmetry"] is not None else None) for r in self.records},
                "times": {r["id"]: r["time"] for r in self.records},
                "half_smile_control": next((r["control"] for r in self.records if r["id"] == "half_smile"), None)}

    def summary(self):
        out = [("Expressions completed", f"{self.idx} / {len(EXPRESSIONS)}"),
               ("Average symmetry", f"{self.accuracy() * 100:.0f}%")]
        recs = [r for r in self.records if r["symmetry"] is not None]
        if recs:
            best = max(recs, key=lambda r: r["symmetry"])
            worst = min(recs, key=lambda r: r["symmetry"])
            out.append(("Most even", f"{best['name']} ({best['symmetry'] * 100:.0f}%)"))
            out.append(("Needs practice", f"{worst['name']} ({worst['symmetry'] * 100:.0f}%)"))
        return out


def draw_sun(img, center, r, blend, rays, t):
    """Cartoon sun whose face performs the given blendshapes."""
    g = lambda k: float(blend.get(k, 0.0))  # noqa: E731
    cx, cy = int(center[0]), int(center[1])
    ui.radial_glow(img, (cx, cy), int(r * 2.2), SUN, 0.35 + 0.25 * clamp(rays))
    n = 14
    ray_len = r * (0.28 + 0.45 * clamp(rays))
    for i in range(n):
        a = t * 0.3 + i * 2 * math.pi / n
        a1, a2 = a - 0.12, a + 0.12
        p0 = (cx + (r + 8) * math.cos(a1), cy + (r + 8) * math.sin(a1))
        p1 = (cx + (r + 8) * math.cos(a2), cy + (r + 8) * math.sin(a2))
        tip = (cx + (r + 8 + ray_len) * math.cos(a), cy + (r + 8 + ray_len) * math.sin(a))
        cv2.fillPoly(img, [np.array([p0, tip, p1], np.int32)], SUN, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, SUN, -1, cv2.LINE_AA)
    cv2.circle(img, (cx - r // 8, cy - r // 8), int(r * 0.85), (90, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, SUN_DARK, 3, cv2.LINE_AA)
    # cheeks
    for sx in (-1, 1):
        cv2.circle(img, (cx + sx * int(r * 0.52), cy + int(r * 0.22)), int(r * 0.14), (150, 150, 255), -1, cv2.LINE_AA)
    # eyes + brows (screen-left = the sun's... mirror: user's left)
    for sx, blink, brow in ((-1, g("eyeBlinkLeft"), g("browOuterUpLeft")), (1, g("eyeBlinkRight"), g("browOuterUpRight"))):
        ex, ey = cx + sx * int(r * 0.34), cy - int(r * 0.15)
        open_h = int(r * 0.13 * (1 - clamp(blink)))
        if open_h > 3:
            cv2.ellipse(img, (ex, ey), (int(r * 0.11), open_h), 0, 0, 360, C.WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, (ex, ey + 2), max(3, min(open_h, int(r * 0.07))), (40, 30, 30), -1, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (ex, ey), (int(r * 0.11), int(r * 0.05)), 0, 0, 180, (40, 30, 30), 4, cv2.LINE_AA)
        lift = int(r * 0.16 * max(brow, g("browInnerUp")))
        by = ey - int(r * 0.22) - lift
        cv2.ellipse(img, (ex, by + int(r * 0.06)), (int(r * 0.14), int(r * 0.07)), 0, 200, 340, (40, 60, 120), 5,
                    cv2.LINE_AA)
    # mouth
    my = cy + int(r * 0.38)
    jaw, pk = g("jawOpen"), g("mouthPucker")
    sl, sr = g("mouthSmileLeft"), g("mouthSmileRight")
    if pk > 0.3:
        rr = int(r * (0.07 + 0.05 * pk))
        cv2.circle(img, (cx, my), rr + 6, (110, 90, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, my), rr - 2, (40, 30, 90), -1, cv2.LINE_AA)
    elif jaw > 0.25:
        cv2.ellipse(img, (cx, my + int(r * 0.05)), (int(r * 0.2), int(r * (0.08 + 0.2 * jaw))), 0, 0, 360,
                    (40, 30, 90), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, my + int(r * (0.12 + 0.12 * jaw))), (int(r * 0.11), int(r * 0.06)), 0, 0, 360,
                    (120, 110, 240), -1, cv2.LINE_AA)
    else:
        hw = int(r * (0.26 + 0.08 * max(sl, sr)))
        lx, ly = cx - hw, my - int(r * 0.2 * sl)
        rx, ry = cx + hw, my - int(r * 0.2 * sr)
        depth = int(r * (0.04 + 0.16 * (sl + sr) / 2))
        pts = []
        for k in range(21):
            tt = k / 20
            x = lx + (rx - lx) * tt
            y = ly + (ry - ly) * tt + depth * math.sin(math.pi * tt)
            pts.append((x, y))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (40, 30, 90), 6, cv2.LINE_AA)
