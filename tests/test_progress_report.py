"""Progression rules, session log and report generation."""
import json

from core.progress import GameResult, ProgressStore, SessionLog
from core.report import generate_report, insights, load_sessions


def res(gid, passed, score=100, stars=1, **extra):
    return GameResult(gid, f"G{gid}", "EASY", score, 0.8, passed, stars if passed else 0, 30.0, 0.5, 50.0, "x",
                      extra=extra)


def test_starts_with_only_game_one(tmp_path):
    p = ProgressStore(tmp_path)
    assert p.is_unlocked(1) and not p.is_unlocked(2)
    assert p.next_game() == 1 and p.first_run


def test_promotion_only_after_pass(tmp_path):
    p = ProgressStore(tmp_path)
    flags = p.record(res(1, False))
    assert not flags["promoted"] and not p.is_unlocked(2)
    flags = p.record(res(1, True, 500, 2))
    assert flags["promoted"] and p.is_unlocked(2) and not p.is_unlocked(3)
    assert p.game(1)["best_score"] == 500 and p.game(1)["stars"] == 2
    # replaying a passed game does not unlock anything new
    assert not p.record(res(1, True))["promoted"]


def test_two_fails_ease_then_three_stars_restore(tmp_path):
    p = ProgressStore(tmp_path)
    p.record(res(1, False))
    flags = p.record(res(1, False))
    assert flags["eased"] and p.game(1)["ease"] == 1
    flags = p.record(res(1, True, stars=3))
    assert flags["ready_next"] and p.game(1)["ease"] == 0


def test_persistence_and_skip(tmp_path):
    p = ProgressStore(tmp_path)
    p.skip(1)
    p.skip(2)
    p.set_calibration({"affected_side": "left"})
    q = ProgressStore(tmp_path)
    assert q.unlocked == 3 and q.calibration["affected_side"] == "left" and not q.first_run
    json.loads((tmp_path / "progress.json").read_text())       # valid JSON on disk


def test_corrupt_progress_file_recovers(tmp_path):
    (tmp_path / "progress.json").write_text("{not json")
    p = ProgressStore(tmp_path)
    assert p.unlocked == 1


def test_session_log_and_report(tmp_path):
    log = SessionLog(tmp_path)
    for i, reach in enumerate((50, 55, 60, 66)):
        r = res(1, True, 100 + i)
        r.rom_value = reach
        r.extra = {"pops_left": 3, "pops_right": 9, "reach_left": reach - 5, "reach_right": reach}
        log.append(r)
    r5 = res(5, True)
    r5.extra = {"fingers": {"index": {"tries": 4, "ok": 4}, "ring": {"tries": 4, "ok": 2}}, "weakest_finger": "ring"}
    log.append(r5)
    assert len(log.rows()) == 5 and len(log.rows(1)) == 4
    df = load_sessions(tmp_path)
    text = " ".join(insights(df))
    assert "improved" in text and "ring" in text and "left hand did only" in text
    html, png = generate_report(tmp_path, tmp_path / "reports")
    assert html.exists() and png.exists() and png.stat().st_size > 10_000
    assert "therapist" in html.read_text(encoding="utf-8")


def test_report_with_no_data(tmp_path):
    html, png = generate_report(tmp_path, tmp_path / "reports")
    assert png.exists()
    assert "No sessions" in html.read_text(encoding="utf-8")
