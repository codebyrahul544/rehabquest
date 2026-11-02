"""RehabQuest - webcam-controlled rehabilitation game suite.

    python main.py              play with your webcam
    python main.py --demo       no webcam: a virtual patient plays (mouse = your hand in menus)
    python main.py --demo --autoplay   the virtual patient plays the whole journey by itself
    python main.py --selftest   headless run through all 5 games (used by the tests)
    python main.py --debug      show FPS and raw landmark values

State machine:  Loading -> (Calibration on first run) -> Menu -> Tutorial -> Game -> Result -> Menu
                plus Rest reminder, Pain stop, Progress report.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import ui  # noqa: E402
from core.audio import Audio  # noqa: E402
from core.camera import Camera, DemoCamera  # noqa: E402
from core.config import CFG  # noqa: E402
from core.effects import Effects  # noqa: E402
from core.gestures import classify, crossed_hands  # noqa: E402
from core.progress import ProgressStore, SessionLog  # noqa: E402
from core.scenes import (GameScene, MenuScene, MessageScene, PainScene, RestScene,  # noqa: E402
                         ResultScene, TutorialScene)
from core.synth import SimTracker  # noqa: E402
from core.ui import C  # noqa: E402
from games.registry import GAMES  # noqa: E402

WINDOW = "RehabQuest"
KEY_ESC = 27


def setup_logging() -> logging.Logger:
    (ROOT / "logs").mkdir(exist_ok=True)
    log = logging.getLogger("rehabquest")
    if not log.handlers:
        h = RotatingFileHandler(ROOT / "logs" / "error.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


class App:
    def __init__(self, args):
        self.args = args
        self.cfg = CFG
        if args.camera is not None:
            self.cfg.CAMERA_INDEX = args.camera
        self.W, self.H = 1280, 720
        self.demo = args.demo or args.selftest
        self.autoplay = args.autoplay or args.selftest
        self.headless = args.headless or args.selftest
        self.debug = args.debug
        self.fixed_dt = 1 / 30 if args.selftest else None
        self.log = setup_logging()
        self.fx = Effects()
        self.audio = Audio(self.cfg.SOUND and not self.headless, self.cfg.MUSIC, self.cfg.VOLUME, self.cfg.MUSIC_VOLUME)
        self._set_data_dir()
        self.running = True
        self.scene = None
        self.camera = None
        self.tracker = None
        self.mouse = None
        self.click = None
        self._pending_click = None
        self.toasts: list = []
        self.play_time = 0.0
        self.rest_due = False
        self.pain_t = 0.0
        self.fps = 0.0
        self.fullscreen = self.cfg.START_FULLSCREEN
        self.last_tracking = None
        self.results: list = []
        self.frames = 0
        self.selftest_done = False

    # ------------------------------------------------------------------ setup
    def _set_data_dir(self):
        if self.args.selftest:
            self.data_dir = ROOT / "data_selftest"
            shutil.rmtree(self.data_dir, ignore_errors=True)
        else:
            self.data_dir = ROOT / ("data_demo" if self.demo else "data")
        self.progress = ProgressStore(self.data_dir)
        self.sessions = SessionLog(self.data_dir)
        if self.args.reset:
            self.progress.reset()

    def enable_demo(self):
        self.demo = True
        self._set_data_dir()
        if self.camera:
            self.camera.release()
        self.camera = DemoCamera(self.W, self.H)
        self.tracker = SimTracker(self.cfg, self.W / self.H)
        self.after_loading()

    def start_loading(self):
        if self.demo:
            self.camera = DemoCamera(self.W, self.H)
            self.tracker = SimTracker(self.cfg, self.W / self.H)
            self.after_loading()
            return
        self.scene = MessageScene(self, "Starting up...", "Opening your camera and loading the body trackers",
                                  progress=0.0, icon="spin")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        scene = self.scene
        try:
            cam = Camera(self.cfg)
            scene.body = "Opening your camera..."
            if not cam.open():
                self.camera = None
                self.scene = MessageScene(self, "No webcam found", cam.error,
                                          ["Press R to retry", "Press D for demo mode (no camera)", "ESC to quit"])
                self.scene.kind = "nocam"
                return
            self.camera = cam

            def prog(msg, frac):
                scene.body = msg
                scene.progress = frac

            from core.trackers import AsyncTracker, MediaPipeTracker
            tr = MediaPipeTracker(self.cfg)
            tr.prepare(("hands", "face", "pose"), prog)
            self.tracker = AsyncTracker(tr)
            self.after_loading()
        except Exception as exc:
            self.log_exception(exc)
            self.scene = MessageScene(self, "Could not start the trackers", str(exc)[:400],
                                      ["Press D for demo mode (no camera needed)", "ESC to quit"])
            self.scene.kind = "nocam"

    def after_loading(self):
        if self.progress.first_run and not self.args.skip_calibration:
            self.start_calibration()
        else:
            self.open_menu()

    # ------------------------------------------------------------------ navigation
    def switch(self, scene):
        self.scene = scene
        self.fx.clear() if not isinstance(scene, ResultScene) else None

    def open_menu(self):
        if self.rest_due:
            self.rest_due = False
            self.play_time = 0.0
            self.switch(RestScene(self))
            return
        self.switch(MenuScene(self))

    def start_calibration(self):
        from core.calibration import CalibrationScene
        self.switch(CalibrationScene(self))

    def start_tutorial(self, gid):
        if self.rest_due:
            self.open_menu()
            return
        if not self.progress.is_unlocked(gid):
            self.toast(f"Game {gid} is locked - pass Game {gid - 1} first")
            return
        self.switch(TutorialScene(self, GAMES[gid]))

    def finish_game(self, game):
        res = game.result()
        self.sessions.append(res)
        flags = self.progress.record(res)
        self.results.append(res)
        history = self.sessions.rows(res.game_id)
        if flags["promoted"]:
            self.audio.play("success")
        self.switch(ResultScene(self, type(game), res, flags, history))

    def autoplay_finished(self, report_done=False):
        if not report_done and not self.args.selftest:
            from core.scenes import ReportScene
            self.switch(ReportScene(self))
            return
        self.selftest_done = True
        self.running = False

    def toast(self, msg, seconds=3.0):
        self.toasts.append([msg, seconds])

    def cursors(self, tracking):
        """Hand pointers (index fingertip, in pixels) for hover buttons."""
        return [(float(h.norm[8, 0] * self.W), float(h.norm[8, 1] * self.H)) for h in tracking.hands]

    def log_exception(self, exc):
        self.log.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    # ------------------------------------------------------------------ input
    def _on_mouse(self, event, x, y, flags, param):
        try:
            rect = cv2.getWindowImageRect(WINDOW)
            sx = self.W / max(1, rect[2])
            sy = self.H / max(1, rect[3])
        except Exception:
            sx = sy = 1.0
        pt = (x * sx, y * sy)
        self.mouse = pt
        if event == cv2.EVENT_LBUTTONUP:
            self._pending_click = pt

    def handle_key(self, key):
        if key < 0:
            return
        k = key & 0xFF
        if self.scene is not None and self.scene.on_key(k):
            return
        if k == KEY_ESC:
            if isinstance(self.scene, MenuScene) or isinstance(self.scene, MessageScene):
                self.running = False
            else:
                self.open_menu()
        elif k in (ord("m"), ord("M")):
            muted = self.audio.toggle_mute()
            self.toast("Sound off" if muted else "Sound on", 1.5)
        elif k in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif k in (ord("p"), ord("P")):
            self.switch(PainScene(self))
        elif k in (ord("d"), ord("D")):
            if isinstance(self.scene, MessageScene) and getattr(self.scene, "kind", "") == "nocam":
                self.enable_demo()
            else:
                self.debug = not self.debug
        elif k in (ord("r"), ord("R")) and isinstance(self.scene, MessageScene) \
                and getattr(self.scene, "kind", "") == "nocam":
            self.start_loading()

    # ------------------------------------------------------------------ main loop
    def run(self, max_frames=None):
        if not self.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, self.W, self.H)
            cv2.setMouseCallback(WINDOW, self._on_mouse)
            if self.fullscreen:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.start_loading()
        last = time.perf_counter()
        blank = np.zeros((self.H, self.W, 3), np.uint8)
        while self.running:
            now = time.perf_counter()
            dt = self.fixed_dt or min(0.1, now - last)
            last = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3))
            self.click, self._pending_click = self._pending_click, None
            canvas = self.step(dt, blank)
            self.frames += 1
            if max_frames and self.frames >= max_frames:
                break
            if self.headless:
                continue
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKeyEx(1)
            self.handle_key(key)
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
            except cv2.error:
                self.running = False
            spare = 1.0 / self.cfg.TARGET_FPS - (time.perf_counter() - now)
            if spare > 0.002:
                time.sleep(spare * 0.8)
        self.shutdown()

    def step(self, dt, blank):
        scene = self.scene
        frame = None
        if self.camera is not None:
            frame = self.camera.read(self.W, self.H)
            if frame is None and getattr(self.camera, "lost", False):
                self.toast("Camera disconnected - trying to reconnect...", 1.0)
        if frame is None:
            frame = blank.copy()
        tracking = None
        try:
            if self.tracker is not None and scene is not None and scene.needs:
                # demo: scenes without an intent (menus) get a virtual hand that follows the mouse
                intent = scene.demo_intent() if self.demo else None
                mouse = (self.mouse[0] / self.W, self.mouse[1] / self.H) if self.mouse else None
                tracking = self.tracker.process(frame, scene.needs, time.monotonic(), intent=intent, mouse=mouse, dt=dt)
            if tracking is None:
                from core.trackers import Tracking
                tracking = Tracking(aspect=self.W / self.H)
            self.last_tracking = tracking
            self._pain_check(tracking, dt)
            scene = self.scene
            scene.update(frame, tracking, dt)
            if scene.counts_as_play and getattr(scene, "game", None) and scene.game.phase == "play":
                self.play_time += dt
                if self.play_time >= self.cfg.REST_REMINDER_MINUTES * 60:
                    self.rest_due = True
            canvas = frame
            if not (self.headless and self.frames % 10):      # headless self-test: draw every 10th frame
                self.scene.draw(canvas)
            err = getattr(self.tracker, "error", None)
            if err is not None:
                self.tracker.error = None
                self.log_exception(err)
        except Exception as exc:
            self.log_exception(exc)
            print(f"[error] {exc!r} - see logs/error.log", file=sys.stderr)
            if self.args.selftest:
                raise
            self.toast("Oops - something went wrong. Back to the menu.", 4)
            self.switch(MenuScene(self))
            canvas = frame
        self.fx.update(dt)
        if self.headless and self.frames % 10:
            return canvas
        self.fx.draw(canvas)
        self._overlays(canvas, dt)
        ox, oy = self.fx.offset()
        if ox or oy:
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            canvas = cv2.warpAffine(canvas, M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
        return canvas

    def _pain_check(self, tracking, dt):
        if not getattr(self.scene, "allow_pain_gesture", False):
            self.pain_t = 0.0
            return
        crossed = crossed_hands(tracking.hands)
        if not crossed and tracking.pose is not None:
            from core.pose_metrics import features
            p = tracking.pose
            if min(p.vis[15], p.vis[16], p.vis[0]) > 0.6:
                f = features(p, tracking.aspect)
                wl, wr = p.norm[15, :2], p.norm[16, :2]
                close = np.linalg.norm((wl - wr) * (tracking.aspect, 1)) < 0.3 * f["sw"]
                crossed = close and f["handL"] > 0.8 and f["handR"] > 0.8
        self.pain_t = self.pain_t + dt if crossed else 0.0
        if self.pain_t >= self.cfg.PAIN_GESTURE_SECONDS:
            self.pain_t = 0.0
            self.switch(PainScene(self))

    def _overlays(self, img, dt):
        W, H = self.W, self.H
        if self.pain_t > 0.2:
            ui.panel(img, (W // 2 - 260, H // 2 + 120, 520, 70), C.PANEL_DARK, 0.9, 30, border=C.WARN)
            ui.ring(img, (W // 2 - 210, H // 2 + 155), 22, self.pain_t / self.cfg.PAIN_GESTURE_SECONDS, C.WARN, 6)
            ui.text(img, "Keep hands crossed to STOP", (W // 2 + 20, H // 2 + 155), 26, C.TEXT, True, "mm")
        y = 120
        for t in self.toasts:
            msg = t[0]
            tw, _ = ui.text_size(msg, 24, True)
            ui.panel(img, (W - tw - 70, y, tw + 44, 48), C.PANEL_DARK, 0.9, 24, border=C.ACCENT)
            ui.text(img, msg, (W - 48, y + 24), 24, C.TEXT, True, "rm")
            t[1] -= dt
            y += 58
        self.toasts = [t for t in self.toasts if t[1] > 0]
        if self.demo:
            ui.chip(img, "DEMO MODE" + (" - AUTOPLAY" if self.autoplay else ""), (W - 20 - 210, H - 50), C.PINK, 16)
        if self.audio.muted:
            ui.text(img, "muted (M)", (20, H - 40), 18, C.MUTED)
        if self.debug:
            self._debug(img)

    def _debug(self, img):
        tr = self.last_tracking
        lines = [f"FPS {self.fps:5.1f}   scene {type(self.scene).__name__}"]
        timings = getattr(self.tracker, "timings", {})
        if timings:
            lines.append("  ".join(f"{k} {v * 1000:.0f}ms" for k, v in timings.items()))
        if tr is not None:
            for h in tr.hands:
                lines.append(f"{h.side} hand: {classify(h.world, self.cfg)}  wrist=({h.norm[0, 0]:.2f},{h.norm[0, 1]:.2f})"
                             f"  index=({h.norm[8, 0]:.2f},{h.norm[8, 1]:.2f})")
            if tr.face is not None:
                top = sorted(tr.face.blend.items(), key=lambda kv: -kv[1])[:4]
                lines.append("face: " + "  ".join(f"{k}={v:.2f}" for k, v in top))
            if tr.pose is not None:
                from core.pose_metrics import features
                f = features(tr.pose, tr.aspect)
                lines.append("pose: " + "  ".join(f"{k}={f[k]:.0f}" for k in ("abdL", "abdR", "elbL", "elbR", "lean"))
                             + f"  shrug={f['shrug']:.2f}")
        ui.panel(img, (10, 110, 760, 30 + 26 * len(lines)), (0, 0, 0), 0.6, 10)
        for i, s in enumerate(lines):
            ui.text(img, s, (22, 124 + i * 26), 18, (120, 255, 120))

    def shutdown(self):
        try:
            if self.camera:
                self.camera.release()
            if self.tracker:
                self.tracker.close()
            self.audio.close()
        finally:
            if not self.headless:
                cv2.destroyAllWindows()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RehabQuest - webcam rehabilitation games")
    p.add_argument("--demo", action="store_true", help="no webcam: simulated patient")
    p.add_argument("--autoplay", action="store_true", help="with --demo: play the whole journey automatically")
    p.add_argument("--debug", action="store_true", help="show FPS and raw landmark values")
    p.add_argument("--selftest", action="store_true", help="headless demo run through all games, then exit")
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--camera", type=int, default=None, help="camera index (default from .env)")
    p.add_argument("--reset", action="store_true", help="reset game progress (keeps calibration)")
    p.add_argument("--skip-calibration", action="store_true", help="skip first-run calibration")
    p.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    app = App(args)
    try:
        app.run(max_frames=args.frames or (60 * 60 * 30 if args.selftest else None))
    except Exception as exc:
        app.log_exception(exc)
        print(f"Fatal error: {exc!r} (details in logs/error.log)", file=sys.stderr)
        if args.selftest:
            raise
        return 1
    if args.selftest:
        print("\nSELF-TEST RESULTS")
        print(f"{'Game':28s} {'Pass':5s} {'Score':>6s} {'Acc':>5s} {'Stars':>5s} {'Time':>6s}")
        for r in app.results:
            print(f"{r.game_id}. {r.name:25s} {'yes' if r.passed else 'NO':5s} {r.score:6d} {r.accuracy * 100:4.0f}% "
                  f"{r.stars:5d} {r.duration:5.0f}s")
        ok = app.progress.all_passed()
        print("ALL GAMES PASSED" if ok else "SOME GAMES FAILED")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
