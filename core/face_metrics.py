"""Facial-exercise metrics for Bell's palsy / stroke facial weakness.

Strength comes from MediaPipe FaceLandmarker blendshapes (0..1).
Symmetry (the key clinical number) compares the left and right side:

    symmetry = min(L, R) / max(L, R)        (1.0 = perfectly even)

For movements that have no left/right blendshape (pucker, mouth open) we use
geometry instead: the distance of each mouth corner from the face midline
(nose bridge -> chin), penalised by any height difference between corners.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CFG

NOSE_BRIDGE, CHIN = 6, 152
MOUTH_L, MOUTH_R = 61, 291
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [33, 160, 159, 158, 133, 153, 145, 144]
RIGHT_EYE = [362, 385, 386, 387, 263, 373, 374, 380]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]

EXPRESSIONS = [
    {"id": "smile", "name": "Big Smile", "say": "Give me your BIGGEST smile!",
     "tip": "Show your teeth and lift both cheeks evenly.", "thr": "G3_SMILE_THRESHOLD"},
    {"id": "brows", "name": "Raise Eyebrows", "say": "Raise BOTH eyebrows up high",
     "tip": "Look surprised! Lift your forehead.", "thr": "G3_BROW_THRESHOLD"},
    {"id": "eyes", "name": "Close Eyes Gently", "say": "Close both eyes gently",
     "tip": "Soft and slow - no squeezing. The sun will tell you when.", "thr": "G3_EYES_THRESHOLD"},
    {"id": "pucker", "name": "Kiss Lips", "say": "Pucker your lips like a KISS",
     "tip": "Push your lips forward into a small 'O'.", "thr": "G3_PUCKER_THRESHOLD"},
    {"id": "jaw", "name": "Open Wide", "say": "Open your mouth WIDE - say 'Ahh'",
     "tip": "Drop your jaw slowly, keep it centred.", "thr": "G3_JAW_THRESHOLD"},
    {"id": "half_smile", "name": "One-Side Smile", "say": "Smile with ONE side only",
     "tip": "Lift one mouth corner, keep the other side still.", "thr": "G3_SMILE_THRESHOLD"},
]
EXPR_BY_ID = {e["id"]: e for e in EXPRESSIONS}


def b(blend: dict, key: str) -> float:
    return float(blend.get(key, 0.0))


def pair_symmetry(left: float, right: float, floor: float = 0.05) -> float:
    hi = max(left, right)
    if hi < floor:
        return 1.0
    return float(np.clip(min(left, right) / hi, 0.0, 1.0))


def mouth_geometry(norm: np.ndarray, aspect: float = 16 / 9) -> tuple:
    """(left corner distance, right corner distance, height asymmetry) from the face midline."""
    p = norm[:, :2] * np.array([aspect, 1.0])
    a, c = p[NOSE_BRIDGE], p[CHIN]
    axis = c - a
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return 0.0, 0.0, 0.0
    axis /= n
    perp = np.array([-axis[1], axis[0]])
    dl = abs(float(np.dot(p[MOUTH_L] - a, perp)))
    dr = abs(float(np.dot(p[MOUTH_R] - a, perp)))
    hl = float(np.dot(p[MOUTH_L] - a, axis))
    hr = float(np.dot(p[MOUTH_R] - a, axis))
    asym = abs(hl - hr) / max(1e-6, dl + dr)
    return dl, dr, asym


def geometric_symmetry(norm: np.ndarray, aspect: float = 16 / 9) -> float:
    dl, dr, asym = mouth_geometry(norm, aspect)
    width = pair_symmetry(dl, dr, floor=1e-4)
    return float(np.clip(width * (1.0 - min(1.0, 2.0 * asym)), 0.0, 1.0))


def sides(expr_id: str, face, aspect: float = 16 / 9) -> tuple:
    """(left, right) activation shown on the L/R bars."""
    bl = face.blend
    if expr_id in ("smile", "half_smile"):
        return b(bl, "mouthSmileLeft"), b(bl, "mouthSmileRight")
    if expr_id == "brows":
        return b(bl, "browOuterUpLeft"), b(bl, "browOuterUpRight")
    if expr_id == "eyes":
        return b(bl, "eyeBlinkLeft"), b(bl, "eyeBlinkRight")
    dl, dr, _ = mouth_geometry(face.norm, aspect)
    m = max(dl, dr, 1e-6)
    return dl / m, dr / m


def strength(expr_id: str, face) -> float:
    bl = face.blend
    if expr_id == "smile":
        return (b(bl, "mouthSmileLeft") + b(bl, "mouthSmileRight")) / 2
    if expr_id == "brows":
        return max(b(bl, "browInnerUp"), (b(bl, "browOuterUpLeft") + b(bl, "browOuterUpRight")) / 2)
    if expr_id == "eyes":
        return (b(bl, "eyeBlinkLeft") + b(bl, "eyeBlinkRight")) / 2
    if expr_id == "pucker":
        return b(bl, "mouthPucker")
    if expr_id == "jaw":
        return b(bl, "jawOpen")
    if expr_id == "half_smile":
        return max(b(bl, "mouthSmileLeft"), b(bl, "mouthSmileRight"))
    raise KeyError(expr_id)


def symmetry(expr_id: str, face, aspect: float = 16 / 9) -> Optional[float]:
    """Left/right symmetry 0..1, or None where symmetry is not the goal (one-side smile)."""
    bl = face.blend
    if expr_id == "smile":
        return pair_symmetry(b(bl, "mouthSmileLeft"), b(bl, "mouthSmileRight"))
    if expr_id == "brows":
        l, r = b(bl, "browOuterUpLeft"), b(bl, "browOuterUpRight")
        return pair_symmetry(l, r) if max(l, r) > 0.1 else 1.0
    if expr_id == "eyes":
        return pair_symmetry(b(bl, "eyeBlinkLeft"), b(bl, "eyeBlinkRight"))
    if expr_id in ("pucker", "jaw"):
        return geometric_symmetry(face.norm, aspect)
    return None


def one_side_control(face) -> float:
    """How well one side moves without the other (0..1) for the one-side smile."""
    l, r = b(face.blend, "mouthSmileLeft"), b(face.blend, "mouthSmileRight")
    return float(abs(l - r))


def expression_ok(expr_id: str, face, cfg=CFG, ease: float = 1.0, aspect: float = 16 / 9) -> tuple:
    """(strength_ok, symmetry_ok, strength, symmetry_or_control)."""
    thr = getattr(cfg, EXPR_BY_ID[expr_id]["thr"]) * ease
    s = strength(expr_id, face)
    if expr_id == "half_smile":
        ctrl = one_side_control(face)
        return s >= thr * 0.9, ctrl >= cfg.G3_HALF_SMILE_DIFF * ease, s, ctrl
    sym = symmetry(expr_id, face, aspect)
    return s >= thr, sym >= cfg.G3_SYMMETRY_THRESHOLD * ease, s, sym
