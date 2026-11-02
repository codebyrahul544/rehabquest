"""Settings for RehabQuest.

Every tunable constant lives here. Values are read (in priority order) from:
  1. real environment variables,
  2. the project's ``.env`` file,
  3. the defaults below.
So a therapist can tune thresholds by editing ``.env`` without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, object] = {
    # ---- camera / window -------------------------------------------------
    "CAMERA_INDEX": 0,
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "MIRROR": True,
    "WINDOW_WIDTH": 1280,
    "WINDOW_HEIGHT": 720,
    "START_FULLSCREEN": False,
    "TARGET_FPS": 30,
    "DETECT_WIDTH": 640,
    "MIN_BRIGHTNESS": 60,
    # ---- sound ---------------------------------------------------------------
    "SOUND": True,
    "MUSIC": True,
    "VOLUME": 0.7,
    "MUSIC_VOLUME": 0.10,
    # ---- session / interaction ---------------------------------------------
    "REST_REMINDER_MINUTES": 10.0,
    "REST_SECONDS": 60,
    "HOVER_CLICK_SECONDS": 1.5,
    "START_GATE_SECONDS": 2.0,
    "PAIN_GESTURE_SECONDS": 1.5,
    "EASE_STEP": 0.10,
    # ---- tracking ------------------------------------------------------------
    "SMOOTHING_ALPHA": 0.55,
    "TRACKING_LOST_GRACE": 0.35,
    "MIN_DETECTION_CONFIDENCE": 0.5,
    "MIN_TRACKING_CONFIDENCE": 0.5,
    "HANDEDNESS_SWAP": False,
    "FACE_SIDE_SWAP": True,
    # ---- hand geometry (shared by games 2 and 5) ----------------------------
    "OPEN_RATIO_DEFAULT": 1.85,
    "FIST_RATIO_DEFAULT": 1.05,
    "FINGER_EXTENDED_MAX_DEG": 80.0,
    "FINGER_CURLED_MIN_DEG": 110.0,
    "THUMB_EXTENDED_RATIO": 0.95,
    "PINCH_THRESHOLD": 0.35,
    # ---- Game 1: Bubble Garden ----------------------------------------------
    "G1_PASS_POPS": 15,
    "G1_TIME_LIMIT": 90.0,
    "G1_SPAWN_INTERVAL": 2.2,
    "G1_BUBBLE_RADIUS": 62,
    "G1_TOUCH_MARGIN": 28,
    "G1_HOVER_SECONDS": 6.0,
    # ---- Game 2: Lantern Lighter ---------------------------------------------
    "G2_PASS_LANTERNS": 8,
    "G2_TIME_LIMIT": 120.0,
    "G2_OPEN_HOLD": 2.0,
    "G2_FIST_HOLD": 1.0,
    "G2_OPEN_THRESHOLD": 0.72,
    "G2_FIST_THRESHOLD": 0.30,
    # ---- Game 3: Smile Sunshine -----------------------------------------------
    "G3_TIME_LIMIT": 150.0,
    "G3_HOLD": 2.0,
    "G3_SYMMETRY_THRESHOLD": 0.55,
    "G3_SMILE_THRESHOLD": 0.45,
    "G3_BROW_THRESHOLD": 0.35,
    "G3_EYES_THRESHOLD": 0.50,
    "G3_PUCKER_THRESHOLD": 0.45,
    "G3_JAW_THRESHOLD": 0.40,
    "G3_HALF_SMILE_DIFF": 0.20,
    # ---- Game 4: Mirror Dancer ------------------------------------------------
    "G4_POSE_ACCURACY": 0.75,
    "G4_PASS_POSES": 10,
    "G4_TOTAL_POSES": 12,
    "G4_POSE_WINDOW": 7.0,
    "G4_HOLD": 0.5,
    "G4_TOLERANCE_SCALE": 1.0,
    # ---- Game 5: Spell Caster -------------------------------------------------
    "G5_ROUNDS": 5,
    "G5_MAX_MISTAKES": 3,
    "G5_STEP_TIME_START": 9.0,
    "G5_STEP_TIME_END": 6.0,
    "G5_HOLD": 0.45,
}


def load_env_file(path: Path) -> dict[str, str]:
    """Tiny .env parser (KEY=VALUE, '#' comments) so no extra dependency is needed."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        values[key.strip()] = val
    return values


def _cast(raw: str | None, default):
    if raw is None or raw == "":
        return default
    try:
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(float(raw))
        if isinstance(default, float):
            return float(raw)
    except ValueError:
        return default
    return raw


class Config:
    def __init__(self, env_path: Path | None = None, **overrides):
        file_vals = load_env_file(env_path or ROOT / ".env")
        for key, default in DEFAULTS.items():
            raw = os.environ.get(key, file_vals.get(key))
            setattr(self, key, _cast(raw, default))
        for key, val in overrides.items():
            setattr(self, key, val)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in DEFAULTS}


CFG = Config()
