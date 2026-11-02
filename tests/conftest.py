"""Shared test helpers: a minimal fake App and a headless game runner driven by the virtual patient."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.effects import Effects  # noqa: E402
from core.progress import ProgressStore, SessionLog  # noqa: E402
from core.synth import SimTracker  # noqa: E402


class SilentAudio:
    muted = False

    def play(self, name):
        pass

    def toggle_mute(self):
        return True


class FakeApp:
    def __init__(self, data_dir, **cfg_overrides):
        self.cfg = Config(**cfg_overrides)
        self.W, self.H = 1280, 720
        self.audio = SilentAudio()
        self.fx = Effects()
        self.data_dir = Path(data_dir)
        self.progress = ProgressStore(self.data_dir)
        self.sessions = SessionLog(self.data_dir)
        self.demo = True
        self.autoplay = True
        self.headless = True
        self.click = None


def run_game(game, app, max_seconds=400.0, draw_every=20, dt=1 / 30):
    """Play a game with the virtual patient until it ends. Returns (result, frames)."""
    sim = SimTracker(app.cfg, app.W / app.H)
    canvas = np.zeros((app.H, app.W, 3), np.uint8)
    frames = 0
    t = 0.0
    while not game.done and t < max_seconds:
        tr = sim.process(canvas, game.needs, t, intent=game.demo_intent(), dt=dt)
        game.update(canvas, tr, dt)
        app.fx.update(dt)
        if frames % draw_every == 0:
            img = canvas.copy()
            game.draw(img)
            app.fx.draw(img)
        frames += 1
        t += dt
    return game.result(), frames


@pytest.fixture
def app(tmp_path):
    return FakeApp(tmp_path)
