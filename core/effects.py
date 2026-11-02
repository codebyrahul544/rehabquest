"""Juice: particles (pops, sparkles, confetti), shockwave rings, floating score text, screen shake."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from . import ui
from .ui import C


# Easing -------------------------------------------------------------------
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_in_out_sine(t):
    return -(math.cos(math.pi * clamp(t)) - 1) / 2


CONFETTI_COLORS = [C.GOLD, C.PINK, C.ACCENT, C.GOOD, C.WARN, C.PURPLE]


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size", "kind", "gravity", "drag", "rot", "vr")

    def __init__(self, x, y, vx, vy, life, color, size, kind="dot", gravity=0.0, drag=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size, self.kind = color, size, kind
        self.gravity, self.drag = gravity, drag
        self.rot = random.uniform(0, math.pi)
        self.vr = random.uniform(-8, 8)


class Effects:
    def __init__(self):
        self.particles: list = []
        self.rings: list = []      # [x, y, r, max_r, life, max_life, color]
        self.texts: list = []      # [text, x, y, vy, life, max_life, color, size]
        self.shake_t = 0.0
        self.shake_amp = 0.0
        self.flash = 0.0
        self.flash_color = C.WHITE

    def clear(self):
        self.particles.clear()
        self.rings.clear()
        self.texts.clear()

    # -- emitters ---------------------------------------------------------------
    def burst(self, x, y, color, n=24, speed=260, life=0.8, size=6, gravity=300, kind="dot"):
        colors = color if isinstance(color, list) else [color]
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.35, 1.0) * speed
            self.particles.append(Particle(x, y, math.cos(a) * s, math.sin(a) * s, random.uniform(0.6, 1.0) * life,
                                           random.choice(colors), random.uniform(0.6, 1.0) * size, kind, gravity, 1.8))
        if len(self.particles) > 900:
            del self.particles[:len(self.particles) - 900]

    def sparkle(self, x, y, color=C.GOLD, n=14, speed=160):
        self.burst(x, y, [color, C.WHITE], n, speed, 0.9, 8, 0, "spark")

    def confetti(self, w, n=140):
        for _ in range(n):
            self.particles.append(Particle(random.uniform(0, w), random.uniform(-200, -10), random.uniform(-80, 80),
                                           random.uniform(80, 260), random.uniform(2.0, 3.5),
                                           random.choice(CONFETTI_COLORS), random.uniform(7, 13), "confetti", 120, 0.2))

    def shockwave(self, x, y, color=C.WHITE, max_r=120, life=0.45):
        self.rings.append([x, y, 5.0, max_r, life, life, color])

    def float_text(self, s, x, y, color=C.GOLD, size=40, life=1.0):
        self.texts.append([s, x, y, -70.0, life, life, color, size])

    def shake(self, amp=10.0, duration=0.3):
        self.shake_amp = max(self.shake_amp, amp)
        self.shake_t = max(self.shake_t, duration)

    def flash_screen(self, color=C.WHITE, amount=0.35):
        self.flash = amount
        self.flash_color = color

    # -- simulation ---------------------------------------------------------------
    def update(self, dt):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vy += p.gravity * dt
            f = max(0.0, 1 - p.drag * dt)
            p.vx *= f
            p.vy *= f
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += p.vr * dt
            alive.append(p)
        self.particles = alive
        for r in self.rings:
            r[4] -= dt
            r[2] = r[3] * ease_out_cubic(1 - r[4] / r[5])
        self.rings = [r for r in self.rings if r[4] > 0]
        for t in self.texts:
            t[4] -= dt
            t[2] += t[3] * dt
            t[3] *= 0.96
        self.texts = [t for t in self.texts if t[4] > 0]
        self.shake_t = max(0.0, self.shake_t - dt)
        if self.shake_t == 0:
            self.shake_amp = 0.0
        self.flash = max(0.0, self.flash - dt * 1.5)

    def offset(self):
        if self.shake_t <= 0:
            return 0, 0
        a = self.shake_amp * min(1.0, self.shake_t / 0.15)
        return int(random.uniform(-a, a)), int(random.uniform(-a, a))

    def draw(self, img):
        for r in self.rings:
            k = r[4] / r[5]
            cv2.circle(img, (int(r[0]), int(r[1])), int(r[2]), r[6], max(1, int(8 * k)), cv2.LINE_AA)
        for p in self.particles:
            k = p.life / p.max_life
            s = max(1, int(p.size * (0.4 + 0.6 * k)))
            x, y = int(p.x), int(p.y)
            if p.kind == "confetti":
                c, sn = math.cos(p.rot), math.sin(p.rot)
                w, h = s, s * 0.45 * abs(math.sin(p.rot * 1.3)) + 2
                pts = np.array([(x + c * dx - sn * dy, y + sn * dx + c * dy)
                                for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h))], np.int32)
                cv2.fillPoly(img, [pts], p.color, cv2.LINE_AA)
            elif p.kind == "spark":
                cv2.line(img, (x - s, y), (x + s, y), p.color, 2, cv2.LINE_AA)
                cv2.line(img, (x, y - s), (x, y + s), p.color, 2, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(1, s // 3), C.WHITE, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (x, y), s, p.color, -1, cv2.LINE_AA)
        for s, x, y, _, life, max_life, color, size in self.texts:
            k = life / max_life
            scale = ease_out_back(min(1.0, (1 - k) * 5))
            ui.text(img, s, (x, y), max(10, int(size * (0.6 + 0.4 * scale))), color, True, "mm",
                    stroke=3, alpha=min(1.0, k * 2.5))
        if self.flash > 0:
            ui.darken(img, self.flash, self.flash_color)
