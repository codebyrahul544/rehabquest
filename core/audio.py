"""Procedurally generated sound effects (NumPy -> pygame.mixer). No audio files needed.

If audio cannot start (no sound card, pygame missing, headless test) every call
quietly does nothing.
"""
from __future__ import annotations

import os

import numpy as np

SR = 44100


def _env(n, attack=0.005, release=0.15):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.clip((n / SR - t) / max(release, 1e-4), 0, 1)
    return a * r


def tone(freq, dur, wave="sine", vol=0.5, attack=0.005, release=None, decay=0.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = np.asarray(freq) if np.ndim(freq) else freq
    phase = 2 * np.pi * (np.cumsum(f) / SR if np.ndim(f) else f * t)
    if wave == "sine":
        y = np.sin(phase)
    elif wave == "tri":
        y = 2 / np.pi * np.arcsin(np.sin(phase))
    elif wave == "square":
        y = np.sign(np.sin(phase)) * 0.5
    else:
        y = np.sin(phase)
    y = y * _env(n, attack, release if release is not None else dur * 0.5)
    if decay:
        y *= np.exp(-t * decay)
    return y * vol


def sweep(f0, f1, dur, **kw):
    n = int(SR * dur)
    return tone(np.geomspace(f0, f1, n), dur, **kw)


def seq(parts, gap=0.0):
    out = []
    for p in parts:
        out.append(p)
        if gap:
            out.append(np.zeros(int(SR * gap)))
    return np.concatenate(out)


def mix(*tracks):
    n = max(len(t) for t in tracks)
    out = np.zeros(n)
    for t in tracks:
        out[:len(t)] += t
    return out


def note(name):
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    semis = names[name[0]] + (int(name[-1]) - 4) * 12
    return 440.0 * 2 ** (semis / 12)


def build_sounds() -> dict:
    rng = np.random.default_rng(3)
    s = {}
    s["pop"] = mix(sweep(1100, 260, 0.11, vol=0.55, release=0.08),
                   rng.normal(0, 0.18, int(SR * 0.025)) * _env(int(SR * 0.025), 0.001, 0.02))
    s["click"] = tone(1800, 0.03, vol=0.25, release=0.02)
    s["hover"] = tone(660, 0.05, "tri", vol=0.18, release=0.04)
    s["select"] = seq([tone(880, 0.07, "tri", 0.3), tone(1320, 0.12, "tri", 0.3)])
    s["tick"] = tone(760, 0.07, "tri", vol=0.35, release=0.05)
    s["go"] = mix(tone(1046, 0.35, "tri", 0.4, decay=4), tone(1568, 0.35, "sine", 0.2, decay=4))
    s["chime"] = mix(tone(1318, 0.7, "sine", 0.35, decay=5), tone(2636, 0.5, "sine", 0.12, decay=7),
                     tone(1975, 0.6, "sine", 0.1, decay=6))
    s["charge"] = sweep(300, 900, 0.35, wave="tri", vol=0.18)
    s["success"] = seq([tone(note(n), d, "tri", 0.35, release=d * 0.6)
                        for n, d in (("C5", 0.11), ("E5", 0.11), ("G5", 0.11), ("C6", 0.45))])
    s["fanfare"] = mix(s["success"], np.concatenate([np.zeros(int(SR * 0.33)),
                                                     tone(note("G5"), 0.5, "sine", 0.2, decay=3)]))
    s["fail"] = seq([tone(note(n), 0.18, "tri", 0.28) for n in ("G4", "E4", "C4")])
    s["miss"] = sweep(420, 220, 0.2, wave="tri", vol=0.22)
    shimmer = mix(*[tone(f, 0.8, "sine", 0.06, decay=4) for f in rng.uniform(1800, 4200, 8)])
    s["magic"] = mix(sweep(380, 1700, 0.55, wave="tri", vol=0.3), shimmer)
    s["whoosh"] = rng.normal(0, 0.25, int(SR * 0.4)) * _env(int(SR * 0.4), 0.15, 0.25)
    return s


def build_music(seconds=16.0) -> np.ndarray:
    """Soft ambient pad (C major 7 / A minor 7) with slow tremolo; loops seamlessly."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    out = np.zeros(n)
    chords = [("C3", "E3", "G3", "B3"), ("A2", "C3", "E3", "G3")]
    half = n // 2
    for ci, chord in enumerate(chords):
        seg = slice(ci * half, (ci + 1) * half)
        tt = t[seg]
        for k, nn in enumerate(chord):
            f = note(nn)
            out[seg] += np.sin(2 * np.pi * f * tt) * (0.18 / (k + 1) ** 0.3)
            out[seg] += np.sin(2 * np.pi * f * 2.003 * tt) * 0.03
        fade = np.minimum(1, np.minimum((tt - tt[0]) / 1.5, (tt[-1] - tt) / 1.5))
        out[seg] *= fade
    out *= 0.75 + 0.25 * np.sin(2 * np.pi * t / 4.0)
    return out * 0.6


class Audio:
    def __init__(self, enabled=True, music=True, volume=0.7, music_volume=0.1):
        self.ok = False
        self.muted = False
        self.sounds = {}
        self.music_channel = None
        self.music_volume = music_volume
        if not enabled:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.pre_init(SR, -16, 2, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self._pg = pygame
            for name, wave in build_sounds().items():
                self.sounds[name] = self._make(wave, volume)
            self.ok = True
            if music:
                snd = self._make(build_music(), 1.0)
                self.music_channel = pygame.mixer.Channel(15)
                self.music_channel.play(snd, loops=-1)
                self.music_channel.set_volume(music_volume)
        except Exception as exc:  # pragma: no cover - depends on hardware
            print(f"[audio] disabled: {exc}")
            self.ok = False

    def _make(self, wave, volume):
        data = np.clip(wave * volume, -1, 1)
        pcm = (data * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([pcm, pcm]))
        return self._pg.sndarray.make_sound(stereo)

    def play(self, name):
        if self.ok and not self.muted and name in self.sounds:
            try:
                self.sounds[name].play()
            except Exception:
                pass

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.ok and self.music_channel is not None:
            self.music_channel.set_volume(0 if self.muted else self.music_volume)
        return self.muted

    def close(self):
        if self.ok:
            try:
                self._pg.mixer.quit()
            except Exception:
                pass
