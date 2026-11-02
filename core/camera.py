"""Webcam capture in a background thread (always hands out the newest frame).

The image is mirrored, centre-cropped to 16:9 and resized to the game canvas,
so landmark coordinates map 1:1 onto what the patient sees.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np


def fit_16_9(frame: np.ndarray, out_w: int, out_h: int, mirror: bool = True) -> np.ndarray:
    if mirror:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    target = out_w / out_h
    if w / h > target + 1e-3:          # too wide -> crop sides
        nw = int(h * target)
        x0 = (w - nw) // 2
        frame = frame[:, x0:x0 + nw]
    elif w / h < target - 1e-3:        # too tall (4:3 webcams) -> crop top/bottom
        nh = int(w / target)
        y0 = (h - nh) // 2
        frame = frame[y0:y0 + nh]
    if frame.shape[1] != out_w or frame.shape[0] != out_h:
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return frame


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.failures = 0
        self.frame_id = 0
        self.error = ""

    def open(self) -> bool:
        indices = [self.cfg.CAMERA_INDEX] + [i for i in (0, 1, 2) if i != self.cfg.CAMERA_INDEX]
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for idx in indices:
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = False, None
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                if ok and frame is not None:
                    self.cap = cap
                    self._frame = frame
                    self._running = True
                    self._thread = threading.Thread(target=self._loop, daemon=True)
                    self._thread.start()
                    return True
                cap.release()
        self.error = ("No webcam found (or it is being used by another app).\n"
                      "Close Zoom/Teams/Camera, check the cable, then press R to retry.")
        return False

    def _loop(self):
        while self._running and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self.frame_id += 1
                self.failures = 0
            else:
                self.failures += 1
                time.sleep(0.02)

    @property
    def lost(self) -> bool:
        return self.failures > 60

    def read(self, out_w: int, out_h: int) -> Optional[np.ndarray]:
        with self._lock:
            frame = None if self._frame is None else self._frame
        if frame is None or self.lost:
            return None
        return fit_16_9(frame, out_w, out_h, self.cfg.MIRROR)

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class DemoCamera:
    """Stand-in for a webcam in --demo mode: a soft studio background."""

    def __init__(self, w: int = 1280, h: int = 720):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        top = np.array([70, 48, 38], np.float32)
        bottom = np.array([40, 30, 26], np.float32)
        g = (yy / h)[..., None]
        img = top * (1 - g) + bottom * g
        # soft vignette + "window light"
        d = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.45) / h) ** 2)
        img *= (1.15 - 0.6 * d)[..., None]
        cv2.rectangle(img, (int(w * 0.07), int(h * 0.1)), (int(w * 0.25), int(h * 0.5)), (110, 90, 80), -1)
        cv2.rectangle(img, (int(w * 0.07), int(h * 0.1)), (int(w * 0.25), int(h * 0.5)), (130, 110, 100), 4)
        cv2.line(img, (int(w * 0.16), int(h * 0.1)), (int(w * 0.16), int(h * 0.5)), (130, 110, 100), 4)
        cv2.rectangle(img, (0, int(h * 0.82)), (w, h), (48, 40, 36), -1)
        self.bg = np.clip(img, 0, 255).astype(np.uint8)
        self.lost = False
        self.error = ""

    def open(self) -> bool:
        return True

    def read(self, out_w: int, out_h: int) -> np.ndarray:
        if self.bg.shape[1] != out_w or self.bg.shape[0] != out_h:
            self.bg = cv2.resize(self.bg, (out_w, out_h))
        return self.bg.copy()

    def release(self):
        pass
