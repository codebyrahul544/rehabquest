"""Headless smoke tests: every game is played to the end by the virtual patient."""
import numpy as np
import pytest

from core.trackers import Tracking
from games.registry import GAMES
from tests.conftest import run_game


@pytest.mark.parametrize("gid", list(GAMES))
def test_virtual_patient_passes(app, gid):
    game = GAMES[gid](app, 0, {"hand": "both"})
    res, frames = run_game(game, app)
    assert res.passed, (game.name, res.summary)
    assert 1 <= res.stars <= 3 and res.score > 0 and 0 <= res.accuracy <= 1
    assert res.summary and isinstance(res.extra, dict)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_bubble_garden_single_hand(app, hand):
    res, _ = run_game(GAMES[1](app, 0, {"hand": hand}), app)
    assert res.passed
    other = "pops_right" if hand == "left" else "pops_left"
    assert res.extra[other] == 0


@pytest.mark.parametrize("gid", list(GAMES))
def test_timer_pauses_when_tracking_lost(app, gid):
    game = GAMES[gid](app, 0)
    canvas = np.zeros((app.H, app.W, 3), np.uint8)
    empty = Tracking()
    for _ in range(30 * 6):                # 6 s with nobody in view
        game.update(canvas, empty, 1 / 30)
    assert game.phase == "play" and game.tracking_lost
    assert game.elapsed < 0.5             # timer did not run
    img = canvas.copy()
    game.draw(img)                         # "I can't see you" overlay renders


@pytest.mark.parametrize("gid", list(GAMES))
def test_tutorial_demo_animation_draws(app, gid):
    game = GAMES[gid](app, 0)
    img = np.zeros((app.H, app.W, 3), np.uint8)
    for t in np.linspace(0, 12, 25):
        game.draw_demo(img, (744, 24, 500, 452), float(t))
    assert img.any()
    info = game.intro()
    assert len(info["instructions"]) == 3 and info["condition"] and game.pass_text


def test_difficulty_structure():
    levels = [GAMES[g].difficulty for g in sorted(GAMES)]
    assert levels == ["EASY", "EASY", "EASY", "MEDIUM", "HARD"]


def test_ease_makes_targets_easier(app):
    for gid in GAMES:
        normal, eased = GAMES[gid](app, 0), GAMES[gid](app, 2)
        assert eased.ease_factor < normal.ease_factor
    assert GAMES[1](app, 2).target < GAMES[1](app, 0).target
    assert GAMES[2](app, 2).open_thr < GAMES[2](app, 0).open_thr
    assert GAMES[4](app, 2).thr < GAMES[4](app, 0).thr
    assert GAMES[5](app, 2).max_mistakes > GAMES[5](app, 0).max_mistakes


def test_spell_caster_fails_after_too_many_mistakes(app):
    game = GAMES[5](app, 0)
    canvas = np.zeros((app.H, app.W, 3), np.uint8)
    from core.synth import make_hand
    idle = Tracking(hands=[make_hand("relaxed")])
    t = 0
    while not game.done and t < 200:
        game.update(canvas, idle, 1 / 30)
        t += 1 / 30
    assert game.done and not game.success and game.mistakes > game.max_mistakes
