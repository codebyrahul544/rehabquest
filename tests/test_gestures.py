"""Hand-gesture classification on synthetic landmarks (no webcam needed)."""
import math

import numpy as np
import pytest

from core import gestures as G
from core.config import Config
from core.synth import GESTURE_PARAMS, make_hand

CFG = Config()
EXPECTED = {
    "open": "open", "fist": "fist", "point": "point", "pinky_only": "pinky_only", "peace": "peace",
    "count_3": "count_3", "count_4": "count_4",
    "pinch_index": "pinch_index", "pinch_middle": "pinch_middle", "pinch_ring": "pinch_ring", "pinch_pinky": "pinch_pinky",
}


def rotation(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize("gesture,label", EXPECTED.items())
@pytest.mark.parametrize("side", ["Left", "Right"])
def test_classify_each_gesture(gesture, label, side):
    assert G.classify(make_hand(gesture, side).world, CFG) == label


@pytest.mark.parametrize("gesture", list(GESTURE_PARAMS))
def test_matches_itself(gesture):
    if gesture == "relaxed":
        pytest.skip("relaxed is an in-between hand, not a target")
    assert G.matches(gesture, make_hand(gesture).world, CFG)


@pytest.mark.parametrize("seed", range(8))
def test_rotation_invariance(seed):
    """Angles + normalised distances must not depend on hand orientation."""
    rng = np.random.default_rng(seed)
    R = rotation(*rng.uniform(-math.pi, math.pi, 3))
    for gesture, label in EXPECTED.items():
        w = make_hand(gesture, rotation=R).world
        assert G.classify(w, CFG) == label, gesture


@pytest.mark.parametrize("scale", [0.7, 1.0, 1.4])
def test_scale_invariance(scale):
    """A small (child) or large hand gives the same answer."""
    for gesture, label in EXPECTED.items():
        assert G.classify(make_hand(gesture).world * scale, CFG) == label


def test_count_fingers():
    for n in range(1, 6):
        assert G.count_extended(make_hand(f"count_{n}").world, CFG) == n


def test_openness_ordering_and_normalisation():
    o = G.openness_ratio(make_hand("open").world)
    f = G.openness_ratio(make_hand("fist").world)
    r = G.openness_ratio(make_hand("relaxed").world)
    assert f < r < o
    assert G.normalized_openness(o, f, o) == pytest.approx(1.0)
    assert G.normalized_openness(f, f, o) == pytest.approx(0.0)
    assert G.normalized_openness(o * 2, f, o) == 1.0      # clamped
    assert G.normalized_openness(0.0, f, o) == 0.0


def test_pinch_needs_clear_winner():
    w = make_hand("pinch_middle").world
    d = G.pinch_ratios(w)
    assert d["middle"] < CFG.PINCH_THRESHOLD
    assert G.detect_pinch(w, CFG) == "middle"
    assert G.detect_pinch(make_hand("open").world, CFG) is None
    assert G.detect_pinch(make_hand("fist").world, CFG) is None   # tucked thumb is not a pinch


def test_finger_bend_extremes():
    assert G.finger_bend(make_hand("open").world, "index") < CFG.FINGER_EXTENDED_MAX_DEG
    assert G.finger_bend(make_hand("fist").world, "index") > CFG.FINGER_CURLED_MIN_DEG


def test_dual_parse():
    assert G.parse_dual("dual:open+fist") == ("open", "fist")
    assert G.parse_dual("open") is None


def test_crossed_hands():
    # right hand drawn on the screen-left, left hand on the screen-right, raised and close together
    r = make_hand("open", "Right", center=(0.46, 0.3))
    l = make_hand("open", "Left", center=(0.54, 0.3))
    assert G.crossed_hands([r, l])
    normal_l = make_hand("open", "Left", center=(0.3, 0.3))
    normal_r = make_hand("open", "Right", center=(0.7, 0.3))
    assert not G.crossed_hands([normal_l, normal_r])
    assert not G.crossed_hands([r])
