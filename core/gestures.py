"""Hand-gesture maths. Pure functions on (21, 3) landmark arrays so they are unit-testable.

All measures use MediaPipe *world* landmarks (metres, hand-centred) and are
normalised by hand size, so they work at any distance from the camera and in
any hand orientation:

* finger bend  = sum of the three joint angles along a finger (MCP + PIP + DIP)
* openness     = mean fingertip-to-wrist distance / hand size
* pinch        = thumb-tip to finger-tip distance / hand size
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CFG

WRIST = 0
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
LONG_FINGERS = ["index", "middle", "ring", "pinky"]
TIPS = {name: ids[3] for name, ids in FINGERS.items()}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def hand_size(w: np.ndarray) -> float:
    return max(1e-6, float(np.linalg.norm(w[WRIST] - w[9])))


def finger_bend(w: np.ndarray, name: str) -> float:
    """Total flexion of a long finger in degrees (0 = straight, ~240 = tight fist)."""
    mcp, pip, dip, tip = FINGERS[name]
    return (angle_between(w[mcp] - w[WRIST], w[pip] - w[mcp])
            + angle_between(w[pip] - w[mcp], w[dip] - w[pip])
            + angle_between(w[dip] - w[pip], w[tip] - w[dip]))


def thumb_ratio(w: np.ndarray) -> float:
    """Thumb-tip to little-finger knuckle distance / hand size (big = thumb out)."""
    return float(np.linalg.norm(w[4] - w[17])) / hand_size(w)


def finger_states(w: np.ndarray, cfg=CFG) -> dict:
    """name -> 'ext' | 'curl' | 'mid' for all five digits."""
    out = {}
    for name in LONG_FINGERS:
        b = finger_bend(w, name)
        out[name] = "ext" if b <= cfg.FINGER_EXTENDED_MAX_DEG else ("curl" if b >= cfg.FINGER_CURLED_MIN_DEG else "mid")
    out["thumb"] = "ext" if thumb_ratio(w) >= cfg.THUMB_EXTENDED_RATIO else "curl"
    return out


def openness_ratio(w: np.ndarray) -> float:
    s = hand_size(w)
    return float(np.mean([np.linalg.norm(w[TIPS[f]] - w[WRIST]) for f in LONG_FINGERS])) / s


def normalized_openness(ratio: float, fist_ref: float, open_ref: float) -> float:
    span = max(0.2, open_ref - fist_ref)
    return float(np.clip((ratio - fist_ref) / span, 0.0, 1.0))


def pinch_ratios(w: np.ndarray) -> dict:
    s = hand_size(w)
    return {f: float(np.linalg.norm(w[4] - w[TIPS[f]])) / s for f in LONG_FINGERS}


def detect_pinch(w: np.ndarray, cfg=CFG, threshold: Optional[float] = None) -> Optional[str]:
    """Finger touching the thumb, or None. Must be clearly closer than the runner-up."""
    thr = cfg.PINCH_THRESHOLD if threshold is None else threshold
    # A finger curled into a fist is not "touching the thumb" even if its tip is close.
    d = {f: r for f, r in pinch_ratios(w).items() if finger_bend(w, f) < 165.0}
    if len(d) < 2:
        return None
    ranked = sorted(d.items(), key=lambda kv: kv[1])
    best, best_d = ranked[0]
    if best_d > thr:
        return None
    if ranked[1][1] < best_d * 1.35 and ranked[1][1] < thr:
        return None   # two fingers touching the thumb: ambiguous
    return best


def count_extended(w: np.ndarray, cfg=CFG) -> Optional[int]:
    """Number of raised digits (thumb included), None if any finger is half-bent."""
    st = finger_states(w, cfg)
    if any(st[f] == "mid" for f in LONG_FINGERS):
        return None
    return sum(1 for f in FINGER_ORDER if st[f] == "ext")


# Gesture library ------------------------------------------------------------
GESTURE_INFO = {
    # id: (display name, instruction, fingers trained)
    "open": ("Open Hand", "Spread all five fingers wide", ["thumb", "index", "middle", "ring", "pinky"]),
    "fist": ("Gentle Fist", "Curl all fingers into a soft fist", ["index", "middle", "ring", "pinky"]),
    "pinch_index": ("Thumb + Index", "Touch your THUMB to your INDEX finger", ["index"]),
    "pinch_middle": ("Thumb + Middle", "Touch your THUMB to your MIDDLE finger", ["middle"]),
    "pinch_ring": ("Thumb + Ring", "Touch your THUMB to your RING finger", ["ring"]),
    "pinch_pinky": ("Thumb + Pinky", "Touch your THUMB to your LITTLE finger", ["pinky"]),
    "point": ("Index Only", "Raise ONLY your index finger", ["index"]),
    "pinky_only": ("Pinky Only", "Raise ONLY your little finger", ["pinky"]),
    "peace": ("Peace Sign", "Raise index + middle fingers (V)", ["index", "middle"]),
    "count_1": ("Count 1", "Show ONE finger", ["index"]),
    "count_2": ("Count 2", "Show TWO fingers", ["index", "middle"]),
    "count_3": ("Count 3", "Show THREE fingers", ["index", "middle", "ring"]),
    "count_4": ("Count 4", "Show FOUR fingers", ["index", "middle", "ring", "pinky"]),
    "count_5": ("Count 5", "Show all FIVE fingers", ["thumb", "index", "middle", "ring", "pinky"]),
}

_SHAPES = {
    # gesture: (must be extended, must be curled)
    "point": ({"index"}, {"middle", "ring", "pinky"}),
    "pinky_only": ({"pinky"}, {"index", "middle", "ring"}),
    "peace": ({"index", "middle"}, {"ring", "pinky"}),
    "open": ({"index", "middle", "ring", "pinky", "thumb"}, set()),
    "fist": (set(), {"index", "middle", "ring", "pinky"}),
}


def matches(gesture: str, w: np.ndarray, cfg=CFG) -> bool:
    if gesture.startswith("pinch_"):
        return detect_pinch(w, cfg) == gesture[6:]
    if gesture.startswith("count_"):
        return count_extended(w, cfg) == int(gesture[6:])
    ext, curl = _SHAPES[gesture]
    st = finger_states(w, cfg)
    return all(st[f] == "ext" for f in ext) and all(st[f] == "curl" for f in curl)


def classify(w: np.ndarray, cfg=CFG) -> str:
    """Best single label (for the debug overlay)."""
    p = detect_pinch(w, cfg)
    if p:
        return "pinch_" + p
    for g in ("point", "pinky_only", "peace", "fist", "open"):
        if matches(g, w, cfg):
            return g
    n = count_extended(w, cfg)
    return f"count_{n}" if n else "unknown"


def parse_dual(gesture: str):
    """'dual:open+fist' -> ('open', 'fist') for (user's left, user's right)."""
    if not gesture.startswith("dual:"):
        return None
    a, b = gesture[5:].split("+")
    return a, b


def crossed_hands(hands) -> bool:
    """Pain-stop gesture: two raised hands crossed over each other (wrists close, labels swapped)."""
    if len(hands) != 2:
        return False
    a, b = sorted(hands, key=lambda h: h.palm[0])
    raised = a.norm[0, 1] < 0.6 and b.norm[0, 1] < 0.6
    close = abs(a.norm[0, 0] - b.norm[0, 0]) < 0.15
    swapped = a.side == "Right" and b.side == "Left"   # right hand is on the left of the screen
    return raised and close and swapped
