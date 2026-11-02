"""Synthetic landmarks: an anatomical hand / face / upper body generated from parameters.

Used for four things:
  * ``--demo`` mode (a virtual patient plays the games when no webcam is available),
  * tutorial animations and target silhouettes,
  * unit tests of the gesture / pose / face logic (no webcam in CI),
  * the headless smoke test.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .trackers import Face, Hand, Pose, Tracking

# ==========================================================================
# Hands
# ==========================================================================
# Canonical hand: palm facing the camera, thumb pointing to +x (a LEFT hand in mirror view).
_FINGER_GEOM = {
    "index": ((0.032, -0.088), (0.040, 0.024, 0.020)),
    "middle": ((0.010, -0.093), (0.044, 0.028, 0.022)),
    "ring": ((-0.011, -0.088), (0.040, 0.026, 0.021)),
    "pinky": ((-0.030, -0.076), (0.032, 0.020, 0.018)),
}
_FINGER_IDS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_JOINT_FLEX = (75.0, 100.0, 65.0)     # MCP, PIP, DIP flexion at full curl
_THUMB_CMC = np.array([0.022, -0.025, 0.0])
_THUMB_EXT_TIP = np.array([0.098, -0.082, 0.0])
_THUMB_TUCK_TIP = np.array([-0.002, -0.050, -0.030])
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# gesture -> (curl per finger thumb..pinky, pinch finger or None)
GESTURE_PARAMS = {
    "open": ([0, 0, 0, 0, 0], None),
    "relaxed": ([0.3, 0.25, 0.3, 0.35, 0.4], None),
    "fist": ([1, 1, 1, 1, 1], None),
    "point": ([1, 0, 1, 1, 1], None),
    "pinky_only": ([1, 1, 1, 1, 0], None),
    "peace": ([1, 0, 0, 1, 1], None),
    "count_1": ([1, 0, 1, 1, 1], None),
    "count_2": ([1, 0, 0, 1, 1], None),
    "count_3": ([1, 0, 0, 0, 1], None),
    "count_4": ([1, 0, 0, 0, 0], None),
    "count_5": ([0, 0, 0, 0, 0], None),
    "pinch_index": ([0, 0.3, 0, 0, 0], "index"),
    "pinch_middle": ([0, 0, 0.3, 0, 0], "middle"),
    "pinch_ring": ([0, 0, 0, 0.3, 0], "ring"),
    "pinch_pinky": ([0, 0, 0, 0, 0.3], "pinky"),
}


def gesture_vector(gesture: str) -> np.ndarray:
    """9 numbers: 5 curls + 4 pinch amounts (index..pinky). Interpolating these blends gestures."""
    curls, pinch = GESTURE_PARAMS.get(gesture, GESTURE_PARAMS["relaxed"])
    v = np.zeros(9)
    v[:5] = curls
    if pinch:
        v[5 + ["index", "middle", "ring", "pinky"].index(pinch)] = 1.0
    return v


def _rot2(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def hand_world(vec: np.ndarray, side: str = "Right", roll: float = 0.0) -> np.ndarray:
    """(21,3) world landmarks in metres for a gesture vector."""
    curls, pinches = vec[:5], vec[5:]
    w = np.zeros((21, 3))
    tips = {}
    for k, name in enumerate(["index", "middle", "ring", "pinky"]):
        (mx, my), lens = _FINGER_GEOM[name]
        mcp = np.array([mx, my, 0.0])
        d0 = mcp[:2] / np.linalg.norm(mcp[:2])
        c = max(0.03, float(curls[k + 1]))
        ids = _FINGER_IDS[name]
        w[ids] = mcp
        phi = 0.0
        pos = mcp.copy()
        for j in range(3):
            phi += _JOINT_FLEX[j] * c
            r = math.radians(phi)
            direction = np.array([d0[0] * math.cos(r), d0[1] * math.cos(r), -math.sin(r)])
            pos = pos + lens[j] * direction
            w[ids + j + 1] = pos
        tips[name] = pos.copy()
    # thumb
    tc = float(np.clip(curls[0], 0, 1))
    tip = _THUMB_EXT_TIP * (1 - tc) + _THUMB_TUCK_TIP * tc
    p_amt = float(np.max(pinches)) if len(pinches) else 0.0
    if p_amt > 0:
        target = tips[["index", "middle", "ring", "pinky"][int(np.argmax(pinches))]] + np.array([0.0, 0.004, -0.006])
        tip = tip * (1 - p_amt) + target * p_amt
    mcp_t = _THUMB_CMC + 0.035 * np.array([0.75, -0.66, 0.0])
    ip = (mcp_t + tip) / 2 + np.array([0.008, 0.004, 0.0])
    w[1], w[2], w[3], w[4] = _THUMB_CMC, mcp_t, ip, tip
    if roll:
        for i in range(21):
            w[i, :2] = _rot2(w[i, :2], roll)
    if side == "Right":
        w[:, 0] *= -1
    return w


def hand_norm(world: np.ndarray, center, aspect: float = 16 / 9, scale: float = 1.8) -> np.ndarray:
    """Project world landmarks to normalised screen coordinates around a palm centre."""
    palm = world[[0, 5, 9, 13, 17], :2].mean(axis=0)
    out = np.zeros((21, 3))
    out[:, 0] = center[0] + (world[:, 0] - palm[0]) * scale / aspect
    out[:, 1] = center[1] + (world[:, 1] - palm[1]) * scale
    out[:, 2] = world[:, 2]
    return out


@lru_cache(maxsize=64)
def _cached_icon_hand(gesture: str, side: str):
    w = hand_world(gesture_vector(gesture), side)
    return hand_norm(w, (0.0, 0.0), aspect=1.0, scale=1.0)


def icon_points(gesture: str, side: str = "Right") -> np.ndarray:
    """2-D hand shape for drawing gesture icons (unit-free, centred at the palm)."""
    return _cached_icon_hand(gesture, side)[:, :2].copy()


def make_hand(gesture: str, side: str = "Right", center=(0.5, 0.5), aspect: float = 16 / 9,
              roll: float = 0.0, rotation=None) -> Hand:
    world = hand_world(gesture_vector(gesture), side, roll)
    norm = hand_norm(world, center, aspect)
    if rotation is not None:
        world = world @ np.asarray(rotation).T
    return Hand(norm, world, side, 1.0)


# ==========================================================================
# Face
# ==========================================================================
from .face_metrics import (CHIN, FACE_OVAL, LEFT_BROW, LEFT_EYE, LIPS_OUTER, MOUTH_L, MOUTH_R,  # noqa: E402
                           NOSE_BRIDGE, RIGHT_BROW, RIGHT_EYE)

NEUTRAL_FACE = {"mouthSmileLeft": 0.03, "mouthSmileRight": 0.03, "browInnerUp": 0.05,
                "browOuterUpLeft": 0.03, "browOuterUpRight": 0.03, "eyeBlinkLeft": 0.05,
                "eyeBlinkRight": 0.05, "mouthPucker": 0.03, "jawOpen": 0.02}


def face_landmarks(blend: dict, center=(0.5, 0.42), aspect: float = 16 / 9, size: float = 0.30) -> np.ndarray:
    """(478,3) face landmarks responding to blendshapes. Screen-left = user's left (mirror)."""
    g = lambda k: float(blend.get(k, 0.0))  # noqa: E731
    pts = np.zeros((478, 3))
    cx, cy = center[0] * aspect, center[1]
    W, H = size * 0.72, size
    jaw = g("jawOpen")

    def put(i, x, y):
        pts[i, 0] = (cx + x * W) / aspect
        pts[i, 1] = cy + y * H

    pts[:, 0], pts[:, 1] = center[0], center[1]
    n = len(FACE_OVAL)
    for k, idx in enumerate(FACE_OVAL):
        a = 2 * math.pi * k / n - math.pi / 2
        yy = math.sin(a) * 0.5
        put(idx, math.cos(a) * 0.5, yy + (0.06 * jaw if yy > 0.2 else 0.0))
    put(NOSE_BRIDGE, 0.0, -0.12)
    put(1, 0.0, 0.03)
    put(CHIN, 0.0, 0.5 + 0.06 * jaw)
    sl, sr = g("mouthSmileLeft"), g("mouthSmileRight")
    pk = g("mouthPucker")
    half = 0.17 * (1 - 0.4 * pk)
    my = 0.24
    lx, ly = -half - 0.05 * sl, my - 0.06 * sl
    rx, ry = half + 0.05 * sr, my - 0.06 * sr
    m = len(LIPS_OUTER)
    for k, idx in enumerate(LIPS_OUTER):
        a = 2 * math.pi * k / m            # 0 -> left corner, pi -> right corner
        t = (1 - math.cos(a)) / 2         # 0 at left, 1 at right
        x = lx + (rx - lx) * t
        base = ly + (ry - ly) * t
        open_amt = 0.02 + 0.02 * pk + 0.16 * jaw
        y = base + math.sin(a) * open_amt * (1.3 if math.sin(a) > 0 else 0.6)
        put(idx, x, y)
    put(MOUTH_L, lx, ly)
    put(MOUTH_R, rx, ry)
    put(13, 0.0, my - 0.01)
    put(14, 0.0, my + 0.01 + 0.14 * jaw)
    for ids, sx, blink, brow in ((LEFT_EYE, -1, g("eyeBlinkLeft"), g("browOuterUpLeft")),
                                 (RIGHT_EYE, 1, g("eyeBlinkRight"), g("browOuterUpRight"))):
        ex = 0.2 * sx
        opening = 0.05 * (1 - blink)
        for k, idx in enumerate(ids):
            a = 2 * math.pi * k / len(ids)
            put(idx, ex + 0.09 * math.cos(a) * sx, -0.08 + opening * math.sin(a))
        lift = 0.06 * max(brow, g("browInnerUp"))
        for k, idx in enumerate(LEFT_BROW if sx < 0 else RIGHT_BROW):
            put(idx, ex + (k - 2) * 0.05 * sx, -0.2 - lift - 0.02 * math.sin(math.pi * k / 4))
    return pts


def make_face(blend: dict, center=(0.5, 0.42), aspect: float = 16 / 9) -> Face:
    full = dict(NEUTRAL_FACE)
    full.update(blend)
    return Face(face_landmarks(full, center, aspect), full)


# ==========================================================================
# Upper body
# ==========================================================================
def pose_landmarks(spec: dict, aspect: float = 16 / 9, center=(0.5, 0.55), sw: float = 0.25) -> np.ndarray:
    """(33,3) normalised pose landmarks. Index 11 (left shoulder) is on the screen-left (mirror)."""
    g = lambda k, d=0.0: float(spec.get(k, d))  # noqa: E731
    P = {}
    cx, cy = center[0] * aspect, center[1]
    shrug = g("shrug") * 0.15 * sw
    hip_y = cy + 1.3 * sw
    P[23] = np.array([cx - 0.35 * sw, hip_y])
    P[24] = np.array([cx + 0.35 * sw, hip_y])
    P[11] = np.array([cx - 0.5 * sw, cy - shrug])
    P[12] = np.array([cx + 0.5 * sw, cy - shrug])
    P[0] = np.array([cx, cy - 0.42 * sw])
    P[7] = np.array([cx - 0.17 * sw, cy - 0.47 * sw])
    P[8] = np.array([cx + 0.17 * sw, cy - 0.47 * sw])
    for i, (dx, dy) in zip(range(1, 7), [(-.05, -.5), (-.08, -.5), (-.11, -.5), (.05, -.5), (.08, -.5), (.11, -.5)]):
        P[i] = np.array([cx + dx * sw, cy + dy * sw])
    P[9] = np.array([cx - 0.05 * sw, cy - 0.33 * sw])
    P[10] = np.array([cx + 0.05 * sw, cy - 0.33 * sw])
    ua, fa = 0.62 * sw, 0.60 * sw
    for tag, sh, el, wr, out in (("L", 11, 13, 15, -1), ("R", 12, 14, 16, 1)):
        abd = math.radians(g("abd" + tag, 12))
        udir = np.array([out * math.sin(abd), math.cos(abd)])
        P[el] = P[sh] + ua * udir
        bend = 180.0 - g("elb" + tag, 170)
        c1, c2 = _rot2(udir, bend), _rot2(udir, -bend)
        fdir = c1 if c1[0] * out < c2[0] * out else c2       # bend towards the midline
        length = fa * (1.0 - 0.65 * g("fwd" + tag))           # forearm pointing at the camera looks short
        P[wr] = P[el] + length * fdir
        for j, off in zip((17, 19, 21) if tag == "L" else (18, 20, 22), (-0.04, 0.0, 0.04)):
            P[j] = P[wr] + fdir * 0.08 * sw + np.array([off * sw, 0])
    for i in range(25, 33):
        P[i] = np.array([cx + (-0.3 if i % 2 else 0.3) * sw, hip_y + 0.6 * sw])
    lean = g("lean")
    pivot = (P[23] + P[24]) / 2
    pts = np.zeros((33, 3))
    for i, v in P.items():
        if i < 23:
            v = pivot + _rot2(v - pivot, lean)
        pts[i, 0] = v[0] / aspect
        pts[i, 1] = v[1]
    return pts


def make_pose(spec: dict, aspect: float = 16 / 9, center=(0.5, 0.55), sw: float = 0.25) -> Pose:
    vis = np.ones(33)
    vis[25:] = 0.2
    return Pose(pose_landmarks(spec, aspect, center, sw), vis)


# ==========================================================================
# Virtual patient for --demo mode
# ==========================================================================
class SimTracker:
    """Produces Tracking snapshots that follow a scene's ``demo_intent()``.

    Movement is rate-limited and gestures blend smoothly, so the virtual patient
    behaves like a person: it takes time to reach, open, close and hold.
    """

    def __init__(self, cfg, aspect: float = 16 / 9, seed: int = 7):
        self.cfg = cfg
        self.aspect = aspect
        self.rng = np.random.default_rng(seed)
        self.hands: dict = {}
        self.blend = dict(NEUTRAL_FACE)
        self.spec = dict(NEUTRAL_SPEC)
        self.timings: dict = {}
        self.t = 0.0

    def prepare(self, *_, **__):
        return None

    def close(self):
        pass

    def process(self, frame, needs, t, intent=None, mouse=None, dt=1 / 30) -> Tracking:
        self.t += dt
        tr = Tracking(aspect=self.aspect, t=t)
        intent = intent or {}
        if "hands" in needs:
            targets = intent.get("hands")
            if targets is None:
                targets = [{"side": "Right", "pos": mouse, "gesture": "point", "anchor": "index", "speed": 8.0}] if mouse else []
            tr.hands = self._hands(targets, dt)
        if "face" in needs:
            tgt = dict(NEUTRAL_FACE)
            tgt.update((intent.get("face") or {}).get("blend", {}))
            k = min(1.0, dt * 4.0)
            for key in set(tgt) | set(self.blend):
                cur = self.blend.get(key, 0.0)
                self.blend[key] = cur + (tgt.get(key, 0.0) - cur) * k
            wob = 0.01 * math.sin(self.t * 0.9)
            tr.face = make_face(self.blend, center=(0.5 + wob, 0.42), aspect=self.aspect)
        if "pose" in needs:
            tgt = dict(NEUTRAL_SPEC)
            tgt.update((intent.get("pose") or {}).get("spec", {}))
            k = min(1.0, dt * 3.0)
            for key in tgt:
                self.spec[key] = self.spec.get(key, 0.0) + (tgt[key] - self.spec.get(key, 0.0)) * k
            sway = 0.006 * math.sin(self.t * 0.7)
            tr.pose = make_pose(self.spec, self.aspect, center=(0.5 + sway, 0.55))
        return tr

    def _hands(self, targets, dt) -> list:
        out = []
        seen = set()
        for tg in targets:
            side = tg.get("side", "Right")
            seen.add(side)
            pos = tg.get("pos") or (0.5, 0.7)
            st = self.hands.get(side)
            if st is None:
                start = np.array([pos[0], 1.05])
                st = {"pos": start, "vec": gesture_vector("relaxed")}
                self.hands[side] = st
            target = np.array(pos, dtype=float)
            delta = target - st["pos"]
            dist = float(np.linalg.norm(delta))
            step = tg.get("speed", 1.1) * dt
            st["pos"] = target if dist <= step else st["pos"] + delta / dist * step
            k = min(1.0, dt * tg.get("blend_rate", 5.0))
            st["vec"] = st["vec"] + (gesture_vector(tg.get("gesture", "relaxed")) - st["vec"]) * k
            jitter = np.array([math.sin(self.t * 2.1 + len(side)), math.cos(self.t * 1.7)]) * 0.002
            world = hand_world(st["vec"], side, roll=-8 if side == "Right" else 8)
            norm = hand_norm(world, st["pos"] + jitter, self.aspect)
            if tg.get("anchor") == "index":
                norm[:, :2] += (st["pos"] + jitter) - norm[8, :2]
            out.append(Hand(norm, world, side, 0.98))
        for side in list(self.hands):
            if side not in seen:
                del self.hands[side]
        return out


NEUTRAL_SPEC = {"abdL": 12, "abdR": 12, "elbL": 170, "elbR": 170, "fwdL": 0, "fwdR": 0, "lean": 0, "shrug": 0}
