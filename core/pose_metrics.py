"""Upper-body joint angles and the Mirror Dancer pose library (Parkinson's / frozen shoulder).

Coordinates are converted to an aspect-corrected 2-D space (x * width/height, y)
so that angles are true angles on screen. "Left" always means the user's own
left, which is the *screen* left in the mirrored view, so we sort shoulders by
x instead of trusting model labels.

Features (degrees unless noted)
  abdL/abdR   shoulder abduction: angle between the torso-down direction and the
              upper arm (0 = arm by side, 90 = T-pose, 180 = straight up)
  elbL/elbR   elbow extension (180 = straight). Foreshortened forearms count as bent.
  handL/handR wrist height above the shoulder line, in shoulder widths
  lean        trunk side-lean, + = towards the user's right
  tilt        shoulder-line tilt
  shrug       neck length relative to the relaxed baseline (1.0 = relaxed, <0.8 = shrugged)
  head_off    head offset from the shoulder midpoint, in shoulder widths
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gestures import angle_between

NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                    (15, 19), (16, 20)]
DEFAULT_NECK = 0.47


def _side_chains(p: np.ndarray):
    """Return index chains (shoulder, elbow, wrist, hip, ear) for screen-left and screen-right."""
    a = (L_SH, L_EL, L_WR, L_HIP, L_EAR)
    b = (R_SH, R_EL, R_WR, R_HIP, R_EAR)
    return (a, b) if p[L_SH, 0] <= p[R_SH, 0] else (b, a)


def features(pose, aspect: float = 16 / 9, neck_baseline: float | None = None) -> dict:
    p = pose.norm[:, :2] * np.array([aspect, 1.0])
    vis = pose.vis
    left, right = _side_chains(p)
    sl, sr = p[left[0]], p[right[0]]
    sv = sr - sl
    sw = max(1e-6, float(np.linalg.norm(sv)))
    down = np.array([-sv[1], sv[0]]) / sw          # perpendicular to the shoulder line, pointing down
    f: dict = {"sw": sw}
    for tag, (sh, el, wr, hip, ear) in (("L", left), ("R", right)):
        upper = p[el] - p[sh]
        fore = p[wr] - p[el]
        f["abd" + tag] = angle_between(down, upper)
        ang = 180.0 - angle_between(upper, fore)
        ratio = float(np.linalg.norm(fore)) / max(1e-6, 0.95 * float(np.linalg.norm(upper)))
        f["elb" + tag] = min(ang, 180.0 * min(1.0, ratio))
        f["hand" + tag] = float(p[sh, 1] - p[wr, 1]) / sw
        f["vis" + tag] = float(min(vis[sh], vis[el], vis[wr]))
    mid_sh = (sl + sr) / 2
    f["tilt"] = math.degrees(math.atan2(sv[1], sv[0]))
    hips_ok = vis[L_HIP] > 0.5 and vis[R_HIP] > 0.5
    if hips_ok:
        torso = mid_sh - (p[L_HIP] + p[R_HIP]) / 2
        f["lean"] = math.degrees(math.atan2(torso[0], -torso[1]))
    else:
        f["lean"] = f["tilt"]
    ear_y = (p[left[4], 1] + p[right[4], 1]) / 2
    neck = float(mid_sh[1] - ear_y) / sw
    f["neck"] = neck
    f["shrug"] = neck / (neck_baseline or DEFAULT_NECK)
    f["head_off"] = float(p[NOSE, 0] - mid_sh[0]) / sw
    f["upper_vis"] = float(min(vis[L_SH], vis[R_SH], vis[NOSE]))
    return f


def posture_score(f: dict) -> float:
    """0..1: level shoulders, head centred, neck not slumped/shrugged."""
    level = max(0.0, 1.0 - max(0.0, abs(f["tilt"]) - 4.0) / 10.0)
    centred = max(0.0, 1.0 - max(0.0, abs(f["head_off"]) - 0.10) / 0.25)
    neck = max(0.0, 1.0 - max(0.0, abs(f["shrug"] - 1.0) - 0.12) / 0.3)
    return (level + centred + neck) / 3.0


# --------------------------------------------------------------------------
# Pose library
# --------------------------------------------------------------------------
@dataclass
class Crit:
    feat: str
    kind: str           # "min" | "max" | "near"
    value: float
    tol: float
    hint: str
    weight: float = 1.0

    def score(self, v: float, scale: float = 1.0) -> float:
        tol = self.tol * scale
        if self.kind == "min":
            err = max(0.0, self.value - v)
        elif self.kind == "max":
            err = max(0.0, v - self.value)
        else:
            err = max(0.0, abs(v - self.value) - tol)
            return max(0.0, 1.0 - err / tol)
        return max(0.0, 1.0 - err / tol)


@dataclass
class PoseDef:
    id: str
    name: str
    say: str
    crits: list
    spec: dict = field(default_factory=dict)   # synthetic target used for the silhouette / demo
    rom: tuple = ()                            # features recorded as range of motion


NEUTRAL = {"abdL": 12, "abdR": 12, "elbL": 170, "elbR": 170, "fwdL": 0, "fwdR": 0, "lean": 0, "shrug": 0}


def _spec(**kw):
    s = dict(NEUTRAL)
    s.update(kw)
    return s


POSES = {p.id: p for p in [
    PoseDef("arms_up", "Arms Up High", "Reach BOTH arms up to the sky!", [
        Crit("abdL", "min", 145, 35, "Lift your LEFT arm higher", 2.0),
        Crit("abdR", "min", 145, 35, "Lift your RIGHT arm higher", 2.0),
        Crit("elbL", "min", 145, 40, "Straighten your LEFT elbow"),
        Crit("elbR", "min", 145, 40, "Straighten your RIGHT elbow")],
        _spec(abdL=165, abdR=165, elbL=176, elbR=176), ("abdL", "abdR")),
    PoseDef("t_pose", "Big T", "Arms straight out to the sides", [
        Crit("abdL", "near", 90, 20, "LEFT arm level with your shoulder", 2.0),
        Crit("abdR", "near", 90, 20, "RIGHT arm level with your shoulder", 2.0),
        Crit("elbL", "min", 150, 40, "Straighten your LEFT elbow"),
        Crit("elbR", "min", 150, 40, "Straighten your RIGHT elbow")],
        _spec(abdL=90, abdR=90, elbL=176, elbR=176), ("abdL", "abdR")),
    PoseDef("reach_left", "Reach Left", "Reach far out to your LEFT and lean", [
        Crit("abdL", "near", 100, 25, "Stretch your LEFT arm out sideways", 2.0),
        Crit("elbL", "min", 150, 40, "Straighten your LEFT arm"),
        Crit("lean", "max", -8, 8, "Lean your body to the LEFT", 2.0)],
        _spec(abdL=100, elbL=176, abdR=20, lean=-12), ("abdL",)),
    PoseDef("reach_right", "Reach Right", "Reach far out to your RIGHT and lean", [
        Crit("abdR", "near", 100, 25, "Stretch your RIGHT arm out sideways", 2.0),
        Crit("elbR", "min", 150, 40, "Straighten your RIGHT arm"),
        Crit("lean", "min", 8, 8, "Lean your body to the RIGHT", 2.0)],
        _spec(abdR=100, elbR=176, abdL=20, lean=12), ("abdR",)),
    PoseDef("hands_head", "Hands on Head", "Put both hands on top of your head", [
        Crit("handL", "min", 0.6, 0.4, "Bring your LEFT hand up to your head"),
        Crit("handR", "min", 0.6, 0.4, "Bring your RIGHT hand up to your head"),
        Crit("elbL", "max", 115, 40, "Bend your LEFT elbow", 2.0),
        Crit("elbR", "max", 115, 40, "Bend your RIGHT elbow", 2.0),
        Crit("abdL", "min", 90, 40, "Open your LEFT elbow out wide"),
        Crit("abdR", "min", 90, 40, "Open your RIGHT elbow out wide")],
        _spec(abdL=130, abdR=130, elbL=85, elbR=85), ("abdL", "abdR")),
    PoseDef("lean_left", "Lean Left", "Lean your upper body to the LEFT", [
        Crit("lean", "max", -12, 10, "Lean further to your LEFT")],
        _spec(lean=-18, abdL=18, abdR=18), ("lean",)),
    PoseDef("lean_right", "Lean Right", "Lean your upper body to the RIGHT", [
        Crit("lean", "min", 12, 10, "Lean further to your RIGHT")],
        _spec(lean=18, abdL=18, abdR=18), ("lean",)),
    PoseDef("shrug", "Shoulder Shrug", "Lift both shoulders up to your ears", [
        Crit("shrug", "max", 0.78, 0.2, "Lift your shoulders up towards your ears", 2.0),
        Crit("abdL", "max", 40, 30, "Keep your LEFT arm relaxed down"),
        Crit("abdR", "max", 40, 30, "Keep your RIGHT arm relaxed down")],
        _spec(shrug=1.0), ("shrug",)),
    PoseDef("elbows_ribs", "Elbows to Ribs", "Tuck elbows in, forearms forward, squeeze", [
        Crit("abdL", "max", 30, 25, "Bring your LEFT elbow to your side"),
        Crit("abdR", "max", 30, 25, "Bring your RIGHT elbow to your side"),
        Crit("elbL", "max", 110, 40, "Bend your LEFT elbow"),
        Crit("elbR", "max", 110, 40, "Bend your RIGHT elbow")],
        _spec(abdL=15, abdR=15, elbL=90, elbR=90, fwdL=1, fwdR=1), ("elbL", "elbR")),
    PoseDef("left_up", "Left Arm Up", "Raise only your LEFT arm high", [
        Crit("abdL", "min", 145, 35, "Lift your LEFT arm higher", 2.5),
        Crit("elbL", "min", 145, 40, "Straighten your LEFT elbow"),
        Crit("abdR", "max", 40, 30, "Keep your RIGHT arm down", 1.5)],
        _spec(abdL=165, elbL=176), ("abdL",)),
    PoseDef("right_up", "Right Arm Up", "Raise only your RIGHT arm high", [
        Crit("abdR", "min", 145, 35, "Lift your RIGHT arm higher", 2.5),
        Crit("elbR", "min", 145, 40, "Straighten your RIGHT elbow"),
        Crit("abdL", "max", 40, 30, "Keep your LEFT arm down", 1.5)],
        _spec(abdR=165, elbR=176), ("abdR",)),
]}


def pose_accuracy(pdef: PoseDef, f: dict, tol_scale: float = 1.0) -> tuple:
    """(accuracy 0..1, hint for the worst criterion, per-criterion scores)."""
    scores = [c.score(f[c.feat], tol_scale) for c in pdef.crits]
    wsum = sum(c.weight for c in pdef.crits)
    acc = sum(s * c.weight for s, c in zip(scores, pdef.crits)) / wsum
    worst = min(range(len(scores)), key=lambda i: scores[i] * 10 - pdef.crits[i].weight)
    hint = pdef.crits[worst].hint if scores[worst] < 0.999 else "Perfect - hold it!"
    return acc, hint, scores
