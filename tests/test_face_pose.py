"""Facial symmetry and pose-angle logic on synthetic landmarks."""
import pytest

from core import face_metrics as FM
from core import pose_metrics as PM
from core.config import Config
from core.synth import make_face, make_pose
from games.game3_smile_sunshine import DEMO_BLEND

CFG = Config()


# ---------------------------------------------------------------- face
def test_pair_symmetry_formula():
    assert FM.pair_symmetry(0.8, 0.8) == 1.0
    assert FM.pair_symmetry(0.8, 0.4) == pytest.approx(0.5)
    assert FM.pair_symmetry(0.4, 0.8) == pytest.approx(0.5)
    assert FM.pair_symmetry(0.0, 0.01) == 1.0          # both at rest -> no asymmetry to measure


def test_geometric_symmetry():
    sym = make_face({"mouthPucker": 0.8})
    lop = make_face({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.1})
    assert FM.geometric_symmetry(sym.norm) > 0.95
    assert FM.geometric_symmetry(lop.norm) < 0.8


@pytest.mark.parametrize("expr", [e["id"] for e in FM.EXPRESSIONS])
def test_demo_expressions_pass(expr):
    face = make_face(DEMO_BLEND[expr])
    ok_s, ok_y, strength, sym = FM.expression_ok(expr, face, CFG)
    assert ok_s and ok_y, (expr, strength, sym)


@pytest.mark.parametrize("expr", [e["id"] for e in FM.EXPRESSIONS])
def test_neutral_face_does_not_pass(expr):
    ok_s, ok_y, *_ = FM.expression_ok(expr, make_face({}), CFG)
    assert not (ok_s and ok_y)


def test_lopsided_smile_fails_symmetry():
    face = make_face({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.3})
    ok_s, ok_y, _, sym = FM.expression_ok("smile", face, CFG)
    assert ok_s and not ok_y and sym == pytest.approx(1 / 3, abs=0.01)


def test_ease_relaxes_threshold():
    face = make_face({"mouthSmileLeft": 0.42, "mouthSmileRight": 0.42})
    assert not FM.expression_ok("smile", face, CFG, ease=1.0)[0]
    assert FM.expression_ok("smile", face, CFG, ease=0.9)[0]


# ---------------------------------------------------------------- pose
def test_features_basic_angles():
    f = PM.features(make_pose({"abdL": 90, "abdR": 90, "elbL": 178, "elbR": 178}))
    assert f["abdL"] == pytest.approx(90, abs=2) and f["abdR"] == pytest.approx(90, abs=2)
    assert f["elbL"] > 170
    lean = PM.features(make_pose({"lean": 15}))
    assert lean["lean"] == pytest.approx(15, abs=1)
    assert PM.features(make_pose({"lean": -15}))["lean"] == pytest.approx(-15, abs=1)


def test_left_right_follow_screen_not_labels():
    """Swapping MediaPipe's left/right indices must not change which arm is 'left'."""
    p = make_pose({"abdL": 160, "abdR": 10})
    swapped = p.norm.copy()
    for a, b in ((11, 12), (13, 14), (15, 16), (23, 24), (7, 8)):
        swapped[[a, b]] = swapped[[b, a]]
    p2 = type(p)(swapped, p.vis)
    f1, f2 = PM.features(p), PM.features(p2)
    assert f1["abdL"] == pytest.approx(f2["abdL"]) and f1["abdL"] > 150


@pytest.mark.parametrize("pid", list(PM.POSES))
def test_each_pose_matches_itself(pid):
    f = PM.features(make_pose(PM.POSES[pid].spec))
    acc, hint, _ = PM.pose_accuracy(PM.POSES[pid], f)
    assert acc >= 0.95, (pid, acc, hint)


@pytest.mark.parametrize("pid", list(PM.POSES))
def test_neutral_does_not_match(pid):
    f = PM.features(make_pose(PM.NEUTRAL))
    assert PM.pose_accuracy(PM.POSES[pid], f)[0] < CFG.G4_POSE_ACCURACY


def test_shrug_and_posture():
    neutral = PM.features(make_pose(PM.NEUTRAL))
    shrug = PM.features(make_pose({"shrug": 1.0}), neck_baseline=neutral["neck"])
    assert shrug["shrug"] < 0.8
    assert PM.posture_score(neutral) > 0.95
    tilted = PM.features(make_pose({"lean": 20}))
    assert PM.posture_score(tilted) < PM.posture_score(neutral)


def test_hint_names_the_problem():
    f = PM.features(make_pose({"abdL": 165, "abdR": 30, "elbL": 176, "elbR": 176}))
    acc, hint, _ = PM.pose_accuracy(PM.POSES["arms_up"], f)
    assert acc < 0.75 and "RIGHT" in hint
