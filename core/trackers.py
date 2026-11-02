"""Body tracking: MediaPipe Tasks (Hand / Face / Pose landmarkers) behind one small API.

Every frame the app asks for only the body parts the current screen needs
(``needs={"hands"}`` etc.) and gets back a :class:`Tracking` snapshot.
Landmarks are smoothed with an adaptive exponential moving average: heavy
smoothing when the hand is still (no jitter), light smoothing when it moves
fast (low latency).
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import ROOT

MODELS_DIR = ROOT / "models"
MODEL_URLS = {
    "hands": ("hand_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task"),
    "face": ("face_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task"),
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
}

PALM_IDS = (0, 5, 9, 13, 17)


# --------------------------------------------------------------------------
# Data classes handed to games
# --------------------------------------------------------------------------
@dataclass
class Hand:
    norm: np.ndarray            # (21, 3) x,y in 0..1 of the (mirrored) screen, z relative depth
    world: np.ndarray           # (21, 3) metres, hand-centred, orientation independent
    side: str = "Right"         # the user's own hand: "Left" or "Right"
    score: float = 1.0

    def px(self, w: int, h: int) -> np.ndarray:
        return np.column_stack([self.norm[:, 0] * w, self.norm[:, 1] * h]).astype(np.int32)

    @property
    def palm(self) -> np.ndarray:
        return self.norm[list(PALM_IDS), :2].mean(axis=0)

    @property
    def index_tip(self) -> np.ndarray:
        return self.norm[8, :2]

    @property
    def size(self) -> float:
        """Apparent hand size (wrist to middle knuckle) in normalised height units."""
        return float(np.linalg.norm(self.norm[0, :2] - self.norm[9, :2]))


@dataclass
class Face:
    norm: np.ndarray            # (478, 3)
    blend: dict = field(default_factory=dict)   # blendshape name -> 0..1

    @property
    def bbox_area(self) -> float:
        xy = self.norm[:, :2]
        span = xy.max(axis=0) - xy.min(axis=0)
        return float(span[0] * span[1])


@dataclass
class Pose:
    norm: np.ndarray            # (33, 3)
    vis: np.ndarray             # (33,) visibility 0..1


@dataclass
class Tracking:
    hands: list = field(default_factory=list)
    face: Optional[Face] = None
    pose: Optional[Pose] = None
    aspect: float = 16 / 9
    t: float = 0.0

    def hand(self, side: Optional[str] = None) -> Optional[Hand]:
        """Largest visible hand, optionally restricted to one side."""
        cands = [h for h in self.hands if side in (None, "both", h.side)]
        if not cands:
            return None
        return max(cands, key=lambda h: h.size)

    def preferred_hand(self, side: Optional[str]) -> Optional[Hand]:
        """The requested side if visible, otherwise any hand."""
        return self.hand(side) or self.hand()

    def screen_sorted_hands(self) -> list:
        """Hands sorted left-to-right on screen (= user's left then right in mirror view)."""
        return sorted(self.hands, key=lambda h: h.palm[0])


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
class AdaptiveEMA:
    """Per-key exponential moving average whose alpha rises with speed."""

    def __init__(self, alpha: float = 0.55, reset_after: float = 0.4):
        self.alpha = alpha
        self.reset_after = reset_after
        self._state: dict = {}
        self._last: dict = {}

    def __call__(self, key, arr: np.ndarray, t: float) -> np.ndarray:
        prev = self._state.get(key)
        if prev is None or prev.shape != arr.shape or t - self._last.get(key, -1e9) > self.reset_after:
            out = arr.astype(np.float64).copy()
        else:
            speed = float(np.mean(np.abs(arr[..., :2] - prev[..., :2]))) if arr.ndim > 1 else float(np.mean(np.abs(arr - prev)))
            a = min(1.0, self.alpha + speed * 8.0)
            out = a * arr + (1.0 - a) * prev
        self._state[key] = out
        self._last[key] = t
        return out


# --------------------------------------------------------------------------
# Model download
# --------------------------------------------------------------------------
class ModelDownloadError(RuntimeError):
    pass


def ensure_model(kind: str, progress: Optional[Callable[[str, float], None]] = None) -> Path:
    """Download the .task model on first run (with progress) and return its path."""
    name, url = MODEL_URLS[kind]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    if path.exists() and path.stat().st_size > 100_000:
        return path
    tmp = path.with_suffix(".part")
    msg = f"Downloading {kind} model ({name})"
    print(msg + " ...", flush=True)

    def hook(blocks, block_size, total):
        if progress and total > 0:
            progress(msg, min(1.0, blocks * block_size / total))

    try:
        urllib.request.urlretrieve(url, tmp, hook)
        tmp.replace(path)
    except Exception as exc:  # no internet, proxy, ...
        if tmp.exists():
            tmp.unlink()
        raise ModelDownloadError(
            f"Could not download the {kind} model.\nCheck your internet connection, "
            f"or download it manually from:\n{url}\ninto {MODELS_DIR}") from exc
    print(f"  saved {path}", flush=True)
    return path


# --------------------------------------------------------------------------
# MediaPipe tracker
# --------------------------------------------------------------------------
class MediaPipeTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._lm: dict = {}
        self._ts = 0
        self.ema = AdaptiveEMA(cfg.SMOOTHING_ALPHA)
        self.timings: dict = {}

    def prepare(self, kinds=("hands", "face", "pose"), progress=None):
        import mediapipe as mp  # noqa: F401  (import early so errors surface on the loading screen)
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        for kind in kinds:
            if kind in self._lm:
                continue
            path = ensure_model(kind, progress)
            if progress:
                progress(f"Starting {kind} tracker", 1.0)
            # Pass bytes, not a path: avoids MediaPipe problems with spaces / non-ASCII paths on Windows.
            base = mpp.BaseOptions(model_asset_buffer=path.read_bytes())
            mode = vision.RunningMode.VIDEO
            c = self.cfg
            if kind == "hands":
                opts = vision.HandLandmarkerOptions(
                    base_options=base, running_mode=mode, num_hands=2,
                    min_hand_detection_confidence=c.MIN_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=c.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=c.MIN_TRACKING_CONFIDENCE)
                self._lm[kind] = vision.HandLandmarker.create_from_options(opts)
            elif kind == "face":
                opts = vision.FaceLandmarkerOptions(
                    base_options=base, running_mode=mode, num_faces=2,
                    min_face_detection_confidence=c.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=c.MIN_TRACKING_CONFIDENCE,
                    output_face_blendshapes=True)
                self._lm[kind] = vision.FaceLandmarker.create_from_options(opts)
            elif kind == "pose":
                opts = vision.PoseLandmarkerOptions(
                    base_options=base, running_mode=mode, num_poses=1,
                    min_pose_detection_confidence=c.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=c.MIN_TRACKING_CONFIDENCE)
                self._lm[kind] = vision.PoseLandmarker.create_from_options(opts)

    def _timestamp(self) -> int:
        ts = int(time.monotonic() * 1000)
        self._ts = max(ts, self._ts + 1)   # VIDEO mode needs strictly increasing timestamps
        return self._ts

    def process(self, frame_bgr: np.ndarray, needs, t: float, **_) -> Tracking:
        import cv2
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        tr = Tracking(aspect=w / h, t=t)
        if not needs or frame_bgr is None:
            return tr
        dw = self.cfg.DETECT_WIDTH
        small = cv2.resize(frame_bgr, (dw, int(dw * h / w))) if w > dw else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if "hands" in needs and "hands" in self._lm:
            t0 = time.perf_counter()
            res = self._lm["hands"].detect_for_video(image, self._timestamp())
            tr.hands = self._hands(res, t)
            self.timings["hands"] = time.perf_counter() - t0
        if "face" in needs and "face" in self._lm:
            t0 = time.perf_counter()
            res = self._lm["face"].detect_for_video(image, self._timestamp())
            tr.face = self._face(res, t)
            self.timings["face"] = time.perf_counter() - t0
        if "pose" in needs and "pose" in self._lm:
            t0 = time.perf_counter()
            res = self._lm["pose"].detect_for_video(image, self._timestamp())
            tr.pose = self._pose(res, t)
            self.timings["pose"] = time.perf_counter() - t0
        return tr

    # -- converters --------------------------------------------------------
    def _hands(self, res, t) -> list:
        hands = []
        for i, lms in enumerate(res.hand_landmarks or []):
            norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            wl = res.hand_world_landmarks[i] if res.hand_world_landmarks else lms
            world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64)
            label, score = "Right", 1.0
            if res.handedness and res.handedness[i]:
                label = res.handedness[i][0].category_name
                score = res.handedness[i][0].score
            if self.cfg.HANDEDNESS_SWAP:
                label = "Left" if label == "Right" else "Right"
            hands.append(Hand(norm, world, label, score))
        if len(hands) == 2 and hands[0].side == hands[1].side:
            # Model gave both hands the same label: decide by screen position (mirror view).
            a, b = sorted(hands, key=lambda h: h.palm[0])
            a.side, b.side = "Left", "Right"
        for h in hands:
            h.norm = self.ema(("hn", h.side), h.norm, t)
            h.world = self.ema(("hw", h.side), h.world, t)
        hands.sort(key=lambda h: h.size, reverse=True)
        return hands

    def _face(self, res, t) -> Optional[Face]:
        faces = []
        for i, lms in enumerate(res.face_landmarks or []):
            norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            blend = {}
            if res.face_blendshapes and i < len(res.face_blendshapes):
                blend = {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
            faces.append(Face(norm, blend))
        if not faces:
            return None
        face = max(faces, key=lambda f: f.bbox_area)   # closest face = largest
        face.norm = self.ema("fn", face.norm, t)
        keys = sorted(face.blend)
        vals = self.ema("fb", np.array([face.blend[k] for k in keys]), t)
        face.blend = dict(zip(keys, vals.tolist()))
        if self.cfg.FACE_SIDE_SWAP:
            face.blend = swap_blend_sides(face.blend)
        return face

    def _pose(self, res, t) -> Optional[Pose]:
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
        vis = np.array([getattr(p, "visibility", None) or 0.0 for p in lms], dtype=np.float64)
        norm = self.ema("pn", norm, t)
        return Pose(norm, vis)

    def close(self):
        for lm in self._lm.values():
            try:
                lm.close()
            except Exception:
                pass
        self._lm.clear()


def swap_blend_sides(blend: dict) -> dict:
    """Swap ...Left/...Right blendshapes (mirrored input makes the model see the opposite side)."""
    out = {}
    for k, v in blend.items():
        if k.endswith("Left"):
            out[k[:-4] + "Right"] = v
        elif k.endswith("Right"):
            out[k[:-5] + "Left"] = v
        else:
            out[k] = v
    return out


class AsyncTracker:
    """Runs a tracker on a background thread so drawing never waits for detection.

    ``process`` hands over the newest frame and immediately returns the newest
    finished result (at most one frame old), keeping the UI at full frame rate.
    """

    def __init__(self, inner):
        import threading
        self.inner = inner
        self._cv = threading.Condition()
        self._job = None
        self._result = Tracking()
        self._running = True
        self.error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def timings(self):
        return self.inner.timings

    def _loop(self):
        while self._running:
            with self._cv:
                while self._job is None and self._running:
                    self._cv.wait(0.5)
                job, self._job = self._job, None
            if job is None:
                continue
            frame, needs, t = job
            try:
                res = self.inner.process(frame, needs, t)
                with self._cv:
                    self._result = res
            except Exception as exc:     # surfaced in the log by the app
                self.error = exc

    def process(self, frame, needs, t, **_):
        with self._cv:
            self._job = (frame.copy(), frozenset(needs), t)
            self._cv.notify()
            res = self._result
        # Drop body parts the current screen did not ask for (e.g. right after a screen change).
        if "hands" not in needs:
            res = Tracking(hands=[], face=res.face, pose=res.pose, aspect=res.aspect, t=res.t)
        return res

    def close(self):
        self._running = False
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout=1.0)
        self.inner.close()
