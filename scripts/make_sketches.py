"""Generate the hand-drawn-style input sketches used in the experiments.

The sketches are drawn procedurally so that they are fully reproducible and free
of any copyright question. Every stroke is a list of points in normalised
coordinates (0..1, y pointing down) that is passed through a "pen model":

* the path is resampled and perturbed with smooth low-frequency noise (hand wobble),
* both ends overshoot slightly, as quick pen strokes do,
* most strokes are drawn twice with small offsets (typical sketchy double lines),
* stroke width and darkness vary slightly (pen pressure),
* the result is put on a lightly textured, unevenly lit "paper" background, so the
  preprocessing stage has to do real cleanup, as it would with a phone photo.

Usage:  python scripts/make_sketches.py --out sketches
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Point = tuple[float, float]


class Pen:
    """A tiny procedural pen that draws wobbly, sketch-like strokes."""

    def __init__(self, size: int = 768, seed: int = 0, supersample: int = 2):
        self.size = size
        self.ss = supersample
        self.S = size * supersample
        self.rng = np.random.default_rng(seed)
        self.img = Image.new("L", (self.S, self.S), 255)
        self.draw = ImageDraw.Draw(self.img)

    # ------------------------------------------------------------------ core
    def _smooth_noise(self, n: int, amp: float, corr: int = 25) -> np.ndarray:
        noise = self.rng.normal(0, 1, n + 4 * corr)
        k = np.exp(-0.5 * (np.arange(-2 * corr, 2 * corr + 1) / corr) ** 2)
        k /= k.sum()
        smooth = np.convolve(noise, k, mode="same")[2 * corr : 2 * corr + n]
        smooth /= smooth.std() + 1e-8
        return smooth * amp

    def _resample(self, pts: np.ndarray, spacing_px: float = 3.0) -> np.ndarray:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        length = seg.sum()
        if length < 1e-9:
            return pts
        n = max(int(length * self.S / spacing_px), 2)
        cum = np.concatenate([[0], np.cumsum(seg)])
        t = np.linspace(0, length, n)
        return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], 1)

    def stroke(
        self,
        pts: list[Point] | np.ndarray,
        closed: bool = False,
        width: float = 2.2,
        passes: int = 2,
        wobble: float = 0.0025,
        overshoot: float = 0.008,
        tone: int = 35,
    ) -> None:
        p = np.asarray(pts, dtype=float)
        if closed:
            p = np.vstack([p, p[:1]])
        # short strokes are drawn more carefully: scale wobble, overshoot and
        # the double-line offset with stroke length
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        care = min(1.0, length / 0.25)
        wobble *= care
        overshoot *= min(1.0, length / 0.1)
        if length < 0.08:
            passes = 1
        for i in range(passes):
            q = p.copy()
            if not closed and overshoot > 0 and len(q) >= 2:
                for a, b in ((0, 1), (-1, -2)):
                    d = q[a] - q[b]
                    d /= np.linalg.norm(d) + 1e-9
                    q[a] = q[a] + d * overshoot * self.rng.uniform(0.3, 1.2)
            q = self._resample(q)
            amp = wobble * (1.0 if i == 0 else 1.6)
            q = q + np.stack([self._smooth_noise(len(q), amp), self._smooth_noise(len(q), amp)], 1)
            if i > 0:  # second pass: slightly offset
                q = q + self.rng.normal(0, 0.0015, 2)
            px = q * self.S
            w = width * self.ss * (1.0 if i == 0 else 0.7)
            shade = int(np.clip(tone + self.rng.normal(0, 8) + (0 if i == 0 else 45), 0, 200))
            chunk = 12
            for s in range(0, len(px) - 1, chunk):
                seg = [tuple(v) for v in px[s : s + chunk + 1]]
                ww = max(1, int(round(w * self.rng.uniform(0.8, 1.2))))
                if len(seg) >= 2:
                    self.draw.line(seg, fill=shade, width=ww, joint="curve")

    # ------------------------------------------------------------ primitives
    def line(self, a: Point, b: Point, **kw) -> None:
        self.stroke([a, b], **kw)

    def poly(self, pts: list[Point], closed: bool = False, **kw) -> None:
        self.stroke(pts, closed=closed, **kw)

    def rect(self, x0: float, y0: float, x1: float, y1: float, **kw) -> None:
        # drawn as four separate strokes, like a quick hand sketch
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            self.line(a, b, **kw)

    def ellipse(self, cx, cy, rx, ry, t0=0.0, t1=2 * math.pi, rot=0.0, **kw) -> None:
        t = np.linspace(t0, t1, max(int(abs(t1 - t0) * 40), 8))
        x, y = rx * np.cos(t), ry * np.sin(t)
        c, s = math.cos(rot), math.sin(rot)
        pts = np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)
        closed = abs(t1 - t0) >= 2 * math.pi - 1e-6
        kw.setdefault("overshoot", 0.0 if closed else 0.006)
        self.stroke(pts, closed=False, **kw)

    def bezier(self, p0, p1, p2, p3, **kw) -> None:
        self.stroke(bezier_pts(p0, p1, p2, p3), **kw)

    def spline(self, pts: list[Point], closed: bool = False, **kw) -> None:
        self.stroke(catmull_rom(pts, closed), **kw)

    def capsule(self, a: Point, b: Point, r: float, **kw) -> None:
        """Two parallel lines with rounded ends (arms, legs, lamp arms)."""
        a, b = np.asarray(a), np.asarray(b)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        self.line(tuple(a + n * r), tuple(b + n * r), **kw)
        self.line(tuple(a - n * r), tuple(b - n * r), **kw)
        ang = math.atan2(d[1], d[0])
        self.ellipse(*b, r, r, ang - math.pi / 2, ang + math.pi / 2, **kw)
        self.ellipse(*a, r, r, ang + math.pi / 2, ang + 3 * math.pi / 2, **kw)

    def hatch(self, x0, y0, x1, y1, n=6, slant=0.03, **kw) -> None:
        kw.setdefault("width", 1.3)
        kw.setdefault("passes", 1)
        kw.setdefault("tone", 90)
        for i in range(n):
            x = x0 + (x1 - x0) * (i + 0.5) / n
            self.line((x, y0), (x + slant, y1), **kw)

    # ----------------------------------------------------------------- paper
    def finish(self, seed: int = 0) -> Image.Image:
        lines = self.img.resize((self.size, self.size), Image.LANCZOS)
        rng = np.random.default_rng(seed + 1000)
        h = w = self.size
        yy, xx = np.mgrid[0:h, 0:w] / h
        # uneven lighting (phone photo of paper) + fibre noise
        light = 245 - 18 * ((xx - 0.3) ** 2 + (yy - 0.2) ** 2) - 6 * yy
        grain = rng.normal(0, 2.5, (h, w))
        paper = Image.fromarray(np.clip(light + grain, 0, 255).astype(np.uint8))
        paper = paper.filter(ImageFilter.GaussianBlur(0.6))
        ink = np.asarray(lines, dtype=float) / 255.0
        out = np.asarray(paper, dtype=float) * ink  # multiply blend
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB")


def bezier_pts(p0, p1, p2, p3, n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def catmull_rom(pts: list[Point], closed: bool = False, n: int = 20) -> np.ndarray:
    P = np.asarray(pts, dtype=float)
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            )
    out.append(P[-2])
    return np.asarray(out)


# =========================================================================
#                                SKETCHES
# =========================================================================

def futuristic_tower(p: Pen) -> None:
    p.line((0.05, 0.88), (0.95, 0.88), width=2.6)

    def left(y):  # bulging, tapering silhouette of the main tower
        s = (0.88 - y) / 0.76
        return 0.38 + 0.07 * s - 0.035 * math.sin(math.pi * s)

    def right(y):
        return 1.0 - left(y)

    ys = np.linspace(0.88, 0.12, 40)
    p.poly([(left(y), y) for y in ys], width=2.6)
    p.poly([(right(y), y) for y in ys], width=2.6)
    p.poly([(left(0.12), 0.12), (0.5, 0.03), (right(0.12), 0.12)], width=2.2)
    p.line((0.5, 0.03), (0.5, -0.0), width=1.6, passes=1)
    for y in np.arange(0.18, 0.86, 0.055):  # floor bands, slightly curved
        xl, xr = left(y), right(y)
        p.poly([(xl, y), (0.5, y + 0.008), (xr, y)], width=1.5, passes=1, tone=60)
    for k in range(3):  # twisting facade fins
        x0 = 0.40 + 0.06 * k
        p.bezier((x0, 0.86), (x0 + 0.1, 0.65), (x0 - 0.06, 0.35), (0.47 + 0.03 * k, 0.14), width=1.4, passes=1, tone=70)

    # left secondary tower with dome + sky bridge
    p.line((0.15, 0.88), (0.15, 0.42), width=2.2)
    p.line((0.30, 0.88), (0.30, 0.42), width=2.2)
    p.ellipse(0.225, 0.42, 0.075, 0.07, math.pi, 2 * math.pi, width=2.2)
    for y in np.arange(0.48, 0.85, 0.045):
        for x in (0.17, 0.215, 0.26):
            p.line((x, y), (x + 0.025, y), width=1.2, passes=1, tone=80, overshoot=0.0)
    p.line((0.30, 0.50), (left(0.50), 0.50), width=1.8)
    p.line((0.30, 0.535), (left(0.535), 0.535), width=1.8)

    # right secondary tower with slanted roof
    p.poly([(0.70, 0.88), (0.70, 0.60), (0.86, 0.52), (0.86, 0.88)], width=2.2)
    for y in np.arange(0.62, 0.86, 0.05):
        p.line((0.72, y + 0.01), (0.84, y - 0.01), width=1.2, passes=1, tone=80)
    p.hatch(0.80, 0.56, 0.86, 0.86, n=5, slant=0.0)

    # small trees and a drone
    for x in (0.08, 0.34, 0.92):
        p.ellipse(x, 0.84, 0.022, 0.03, width=1.5)
        p.line((x, 0.87), (x, 0.88), width=1.5, passes=1)
    p.ellipse(0.80, 0.22, 0.03, 0.008, width=1.4)
    p.line((0.765, 0.21), (0.835, 0.21), width=1.2, passes=1)


def robot_character(p: Pen) -> None:
    p.ellipse(0.5, 0.22, 0.14, 0.11, width=2.6)  # head
    p.ellipse(0.5, 0.225, 0.10, 0.045, width=2.0)  # visor
    p.ellipse(0.46, 0.225, 0.014, 0.014, width=2.4, passes=1)
    p.ellipse(0.54, 0.225, 0.014, 0.014, width=2.4, passes=1)
    p.ellipse(0.5, 0.275, 0.03, 0.012, 0.2, math.pi - 0.2, width=1.5, passes=1)  # smile
    p.line((0.5, 0.11), (0.5, 0.045), width=1.8)
    p.ellipse(0.5, 0.035, 0.014, 0.014, width=1.8)
    p.ellipse(0.355, 0.22, 0.012, 0.035, width=1.8)  # ear bolts
    p.ellipse(0.645, 0.22, 0.012, 0.035, width=1.8)
    p.line((0.48, 0.33), (0.48, 0.37), width=1.8)
    p.line((0.52, 0.33), (0.52, 0.37), width=1.8)

    # torso: rounded box
    p.spline([(0.37, 0.39), (0.5, 0.37), (0.63, 0.39), (0.645, 0.5), (0.62, 0.62), (0.5, 0.635), (0.38, 0.62), (0.355, 0.5)], closed=True, width=2.6)
    p.ellipse(0.5, 0.47, 0.04, 0.04, width=2.0)  # chest light
    p.ellipse(0.5, 0.47, 0.018, 0.018, width=1.4, passes=1)
    for y in (0.555, 0.58):
        p.line((0.44, y), (0.56, y), width=1.3, passes=1, tone=70)
    p.hatch(0.59, 0.42, 0.63, 0.6, n=4, slant=-0.01)

    # arms: left resting, right waving
    p.ellipse(0.34, 0.41, 0.03, 0.03, width=2.0)
    p.capsule((0.335, 0.43), (0.27, 0.52), 0.022, width=2.0)
    p.capsule((0.27, 0.52), (0.29, 0.63), 0.02, width=2.0)
    p.ellipse(0.29, 0.655, 0.03, 0.025, width=2.0)
    p.ellipse(0.66, 0.41, 0.03, 0.03, width=2.0)
    p.capsule((0.67, 0.40), (0.76, 0.33), 0.022, width=2.0)
    p.capsule((0.76, 0.33), (0.79, 0.22), 0.02, width=2.0)
    p.ellipse(0.795, 0.19, 0.028, 0.03, width=2.0)
    for a in (-0.5, 0.0, 0.5):  # motion marks of the wave
        p.ellipse(0.795, 0.19, 0.07, 0.07, -math.pi / 2 + a - 0.15, -math.pi / 2 + a + 0.15, width=1.2, passes=1, tone=90)

    # legs and feet
    p.capsule((0.44, 0.635), (0.43, 0.80), 0.026, width=2.2)
    p.capsule((0.56, 0.635), (0.57, 0.80), 0.026, width=2.2)
    p.ellipse(0.42, 0.835, 0.065, 0.035, math.pi, 2 * math.pi, width=2.2)
    p.line((0.355, 0.835), (0.485, 0.835), width=2.2)
    p.ellipse(0.58, 0.835, 0.065, 0.035, math.pi, 2 * math.pi, width=2.2)
    p.line((0.515, 0.835), (0.645, 0.835), width=2.2)
    p.ellipse(0.5, 0.875, 0.22, 0.022, width=1.3, passes=1, tone=110)  # ground shadow


def hover_car(p: Pen) -> None:
    body = [(0.08, 0.56), (0.10, 0.48), (0.25, 0.44), (0.38, 0.42), (0.48, 0.32), (0.62, 0.31),
            (0.74, 0.40), (0.88, 0.46), (0.93, 0.52), (0.88, 0.575), (0.6, 0.60), (0.3, 0.605), (0.12, 0.595)]
    p.spline(body, closed=True, width=2.7)
    p.spline([(0.42, 0.42), (0.49, 0.345), (0.61, 0.335), (0.70, 0.41)], width=2.0)  # canopy
    p.line((0.42, 0.42), (0.70, 0.41), width=1.8)
    p.line((0.555, 0.34), (0.55, 0.415), width=1.4, passes=1)
    p.bezier((0.40, 0.43), (0.39, 0.48), (0.39, 0.54), (0.40, 0.59), width=1.5, passes=1)  # door
    p.bezier((0.14, 0.52), (0.4, 0.49), (0.7, 0.49), (0.86, 0.505), width=1.4, passes=1, tone=70)  # side crease
    p.ellipse(0.885, 0.505, 0.022, 0.012, width=1.8)  # headlight
    p.poly([(0.10, 0.49), (0.06, 0.43), (0.15, 0.43), (0.20, 0.46)], width=1.8)  # rear fin
    for cx in (0.28, 0.71):  # hover pads with energy rings
        p.ellipse(cx, 0.625, 0.09, 0.02, width=2.0)
        for k, r in enumerate((0.07, 0.05)):
            p.ellipse(cx, 0.655 + 0.025 * k, r, 0.012, 0, math.pi, width=1.2, passes=1, tone=90)
    for y in (0.46, 0.50, 0.54):  # speed lines
        p.line((0.0, y), (0.06, y), width=1.3, passes=1, tone=80)
    p.ellipse(0.5, 0.80, 0.36, 0.03, width=1.3, passes=1, tone=110)  # shadow on ground
    p.line((0.02, 0.84), (0.98, 0.84), width=1.5, passes=1, tone=90)
    p.hatch(0.15, 0.53, 0.35, 0.59, n=8, slant=-0.015)


def desk_lamp(p: Pen) -> None:
    p.line((0.03, 0.88), (0.97, 0.88), width=2.0, tone=70)
    p.ellipse(0.30, 0.835, 0.14, 0.032, width=2.5)  # base top
    p.ellipse(0.30, 0.86, 0.14, 0.032, 0, math.pi, width=2.5)  # base bottom edge
    p.line((0.16, 0.835), (0.16, 0.86), width=2.2, passes=1)
    p.line((0.44, 0.835), (0.44, 0.86), width=2.2, passes=1)
    p.ellipse(0.30, 0.80, 0.025, 0.025, width=2.2)  # base joint
    p.capsule((0.30, 0.80), (0.42, 0.45), 0.016, width=2.2)  # lower arm
    zig = [(0.30 + 0.12 * t + (0.012 if i % 2 else -0.012), 0.78 - 0.33 * t) for i, t in enumerate(np.linspace(0.08, 0.9, 18))]
    p.poly([(x + 0.03, y) for x, y in zig], width=1.1, passes=1, tone=80)  # spring
    p.ellipse(0.42, 0.45, 0.03, 0.03, width=2.2)  # elbow
    p.ellipse(0.42, 0.45, 0.01, 0.01, width=1.6, passes=1)
    p.capsule((0.42, 0.45), (0.66, 0.30), 0.015, width=2.2)  # upper arm
    p.ellipse(0.67, 0.30, 0.02, 0.02, width=2.0)  # head joint

    # conical shade: axis points down-right at 60 deg, rims are perpendicular to it
    th = math.radians(60)
    ax = np.array([math.cos(th), math.sin(th)])
    u = np.array([math.cos(th - math.pi / 2), math.sin(th - math.pi / 2)])  # rim major axis
    c1 = np.array([0.69, 0.33])
    c2 = c1 + 0.2 * ax
    r1, r2 = 0.035, 0.11
    p.ellipse(*c1, r1, 0.014, rot=th - math.pi / 2, width=2.0)
    p.ellipse(*c2, r2, 0.04, rot=th - math.pi / 2, width=2.6)
    p.line(tuple(c1 + r1 * u), tuple(c2 + r2 * u), width=2.4)
    p.line(tuple(c1 - r1 * u), tuple(c2 - r2 * u), width=2.4)
    p.ellipse(*(c2 + 0.01 * ax), 0.04, 0.018, rot=th - math.pi / 2, t0=0, t1=math.pi, width=1.4, passes=1)  # bulb
    for k in range(1, 6):  # shading on the far side of the cone
        t = k / 6
        a = c1 - r1 * u * 0.6 + t * (c2 - c1) * 0.9
        b = a + u * (0.02 + 0.04 * t)
        p.line(tuple(a), tuple(b), width=1.2, passes=1, tone=90)
    for off in (-0.07, 0.0, 0.07):  # light rays falling on the desk
        start = c2 + off * u + 0.05 * ax
        for s in np.linspace(0.0, 0.18, 3):
            a = start + s * ax
            p.line(tuple(a), tuple(a + 0.035 * ax), width=1.1, passes=1, tone=110, overshoot=0.0)
    p.line((0.385, 0.70), (0.42, 0.72), width=1.5, passes=1)  # power switch
    p.bezier((0.16, 0.85), (0.06, 0.86), (0.08, 0.80), (0.0, 0.81), width=1.5, passes=1)  # cable


def floating_island(p: Pen) -> None:
    top = [(0.2, 0.50), (0.32, 0.465), (0.5, 0.455), (0.68, 0.465), (0.80, 0.50), (0.66, 0.53), (0.5, 0.54), (0.32, 0.53)]
    p.spline(top, closed=True, width=2.4)
    under = [(0.2, 0.505), (0.26, 0.60), (0.30, 0.58), (0.36, 0.70), (0.42, 0.68), (0.5, 0.87),
             (0.56, 0.70), (0.62, 0.72), (0.68, 0.60), (0.74, 0.62), (0.80, 0.505)]
    p.poly(under, width=2.5)
    for a, b in (((0.33, 0.56), (0.38, 0.64)), ((0.48, 0.57), (0.50, 0.75)), ((0.62, 0.58), (0.60, 0.66))):
        p.line(a, b, width=1.3, passes=1, tone=70)
    p.hatch(0.55, 0.56, 0.72, 0.62, n=6, slant=-0.01)

    # cabin
    p.rect(0.36, 0.40, 0.47, 0.48, width=2.0)
    p.poly([(0.34, 0.405), (0.415, 0.33), (0.49, 0.405)], width=2.2)
    p.rect(0.40, 0.435, 0.425, 0.48, width=1.5, passes=1)
    p.rect(0.37, 0.415, 0.39, 0.435, width=1.3, passes=1)
    p.rect(0.44, 0.345, 0.455, 0.375, width=1.5, passes=1)
    for k in range(3):  # chimney smoke
        p.ellipse(0.455 + 0.02 * k, 0.32 - 0.035 * k, 0.012 + 0.004 * k, 0.01, width=1.1, passes=1, tone=110)

    for x, h in ((0.60, 0.13), (0.67, 0.10), (0.27, 0.09)):  # pine trees
        y = 0.49
        p.line((x, y), (x, y - 0.02), width=1.6, passes=1)
        pts = []
        for k in range(4):
            yy = y - 0.02 - h * k / 4
            w = 0.035 * (1 - k / 4)
            pts += [(x - w, yy), (x, yy - h / 4 * 0.7)]
        pts2 = [(x + w, yy) for (xx, yy), w in zip(pts[::2], [0.035 * (1 - k / 4) for k in range(4)])]
        p.poly(pts + [(x, y - 0.02 - h)], width=1.6, passes=1)
        p.poly([(x, y - 0.02 - h)] + pts2[::-1], width=1.6, passes=1)

    for k in range(4):  # waterfall
        x0 = 0.765 + 0.008 * k
        p.poly([(x0 + 0.004 * math.sin(i), 0.52 + 0.04 * i) for i in range(11)], width=1.2, passes=1, tone=80)
    p.ellipse(0.785, 0.94, 0.04, 0.01, width=1.2, passes=1, tone=90)

    for cx, cy, s in ((0.15, 0.2, 1.0), (0.8, 0.14, 1.2)):  # clouds
        for dx, r in ((-0.05, 0.03), (0.0, 0.045), (0.055, 0.032)):
            p.ellipse(cx + dx * s, cy, r * s, r * s * 0.8, math.pi, 2 * math.pi, width=1.5, passes=1, tone=70)
        p.line((cx - 0.08 * s, cy), (cx + 0.087 * s, cy), width=1.5, passes=1, tone=70)
    for bx, by in ((0.55, 0.18), (0.6, 0.22), (0.52, 0.25)):  # birds
        p.poly([(bx - 0.015, by - 0.006), (bx, by), (bx + 0.015, by - 0.006)], width=1.3, passes=1)
    p.poly([(0.12, 0.64), (0.16, 0.62), (0.18, 0.66), (0.14, 0.70)], closed=True, width=1.6)  # floating rocks
    p.poly([(0.86, 0.68), (0.9, 0.67), (0.9, 0.71)], closed=True, width=1.6)


SKETCHES = {
    "01_futuristic_tower": futuristic_tower,
    "02_robot_character": robot_character,
    "03_hover_car": hover_car,
    "04_desk_lamp": desk_lamp,
    "05_floating_island": floating_island,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sketches")
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, fn) in enumerate(SKETCHES.items()):
        pen = Pen(args.size, seed=42 + i)
        fn(pen)
        img = pen.finish(seed=42 + i)
        img.save(out / f"{name}.png")
        print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()
