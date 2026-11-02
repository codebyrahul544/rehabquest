"""Progression (unlocks, stars, best scores, adaptive easing) and the session log.

progress.json   who has unlocked what + calibration baseline
sessions.csv    one row per finished game, used by the progress report
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

GAME_COUNT = 5
MAX_EASE = 2

SESSION_FIELDS = ["timestamp", "game_id", "game_name", "difficulty", "score", "accuracy", "reaction_time_s",
                  "rom_value", "rom_label", "duration_s", "passed", "stars", "ease_level", "extra"]


@dataclass
class GameResult:
    game_id: int
    name: str
    difficulty: str
    score: int
    accuracy: float              # 0..1
    passed: bool
    stars: int
    duration: float
    reaction_time: float = 0.0
    rom_value: float = 0.0
    rom_label: str = ""
    extra: dict = field(default_factory=dict)
    summary: list = field(default_factory=list)   # [(label, value)] shown on the result screen
    hint: str = ""
    ease: int = 0


def _default_game():
    return {"passed": False, "best_score": 0, "stars": 0, "attempts": 0, "passes": 0,
            "fails_in_row": 0, "ease": 0, "last_played": ""}


class ProgressStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "progress.json"
        self.data = self._load()

    def _load(self) -> dict:
        data = {"version": 1, "unlocked": 1, "games": {}, "calibration": None, "settings": {}}
        try:
            with open(self.path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data.update(loaded)
        except (OSError, ValueError):
            pass
        for gid in range(1, GAME_COUNT + 1):
            g = _default_game()
            g.update(data["games"].get(str(gid), {}))
            data["games"][str(gid)] = g
        data["unlocked"] = int(min(GAME_COUNT, max(1, data.get("unlocked", 1))))
        return data

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        os.replace(tmp, self.path)          # atomic: never a half-written file

    # -- queries ------------------------------------------------------------------
    def game(self, gid: int) -> dict:
        return self.data["games"][str(gid)]

    def is_unlocked(self, gid: int) -> bool:
        return gid <= self.data["unlocked"]

    @property
    def unlocked(self) -> int:
        return self.data["unlocked"]

    def next_game(self) -> int:
        """First unlocked game not yet passed (or the last one)."""
        for gid in range(1, GAME_COUNT + 1):
            if self.is_unlocked(gid) and not self.game(gid)["passed"]:
                return gid
        return min(self.unlocked, GAME_COUNT)

    def all_passed(self) -> bool:
        return all(self.game(g)["passed"] for g in range(1, GAME_COUNT + 1))

    @property
    def calibration(self) -> dict:
        return self.data.get("calibration") or {}

    @property
    def first_run(self) -> bool:
        return not self.data.get("calibration")

    def set_calibration(self, calib: dict):
        self.data["calibration"] = calib
        self.save()

    def setting(self, key, default=None):
        return self.data.setdefault("settings", {}).get(key, default)

    def set_setting(self, key, value):
        self.data.setdefault("settings", {})[key] = value
        self.save()

    # -- updates ------------------------------------------------------------------
    def record(self, res: GameResult) -> dict:
        """Apply a finished game. Returns flags for the result screen."""
        g = self.game(res.game_id)
        g["attempts"] += 1
        g["last_played"] = time.strftime("%Y-%m-%d %H:%M")
        new_best = res.score > g["best_score"]
        g["best_score"] = max(g["best_score"], res.score)
        promoted = False
        eased = False
        if res.passed:
            first_pass = not g["passed"]
            g["passed"] = True
            g["passes"] += 1
            g["stars"] = max(g["stars"], res.stars)
            g["fails_in_row"] = 0
            if res.stars >= 3 and g["ease"] > 0:
                g["ease"] -= 1           # doing great: remove one level of help
            if res.game_id < GAME_COUNT and self.data["unlocked"] <= res.game_id:
                self.data["unlocked"] = res.game_id + 1
                promoted = True
            promoted = promoted or (first_pass and res.game_id == GAME_COUNT)
        else:
            g["fails_in_row"] += 1
            if g["fails_in_row"] >= 2 and g["ease"] < MAX_EASE:
                g["ease"] += 1           # two fails in a row: slightly easier thresholds
                g["fails_in_row"] = 0
                eased = True
        self.save()
        return {"promoted": promoted, "new_best": new_best, "eased": eased,
                "ready_next": res.passed and res.stars >= 3}

    def skip(self, gid: int):
        """Teacher-demo shortcut (key N): mark a game as passed and unlock the next."""
        g = self.game(gid)
        g["passed"] = True
        g["stars"] = max(1, g["stars"])
        if gid < GAME_COUNT:
            self.data["unlocked"] = max(self.data["unlocked"], gid + 1)
        self.save()

    def reset(self):
        calib = self.data.get("calibration")
        self.data = {"version": 1, "unlocked": 1, "games": {}, "calibration": calib, "settings": {}}
        for gid in range(1, GAME_COUNT + 1):
            self.data["games"][str(gid)] = _default_game()
        self.save()


class SessionLog:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "sessions.csv"

    def append(self, res: GameResult):
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SESSION_FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "game_id": res.game_id, "game_name": res.name, "difficulty": res.difficulty,
                "score": res.score, "accuracy": round(res.accuracy, 4),
                "reaction_time_s": round(res.reaction_time, 3), "rom_value": round(res.rom_value, 3),
                "rom_label": res.rom_label, "duration_s": round(res.duration, 1),
                "passed": int(res.passed), "stars": res.stars, "ease_level": res.ease,
                "extra": json.dumps(res.extra, separators=(",", ":")),
            })

    def rows(self, game_id: int | None = None) -> list:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if game_id is not None:
            rows = [r for r in rows if str(r.get("game_id")) == str(game_id)]
        return rows
