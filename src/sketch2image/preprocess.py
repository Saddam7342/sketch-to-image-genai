"""Sketch preprocessing and control-map extraction.

A raw sketch (scan or phone photo) is turned into a clean binary line drawing,
and from that into the *control image* that a specific ControlNet expects:

    raw sketch ──clean_sketch()──> binary lines (black on white, square, /8 size)
                                 └──make_control(mode)──> control image

ControlNet conventions (lllyasviel ControlNet v1.1):
  * scribble / lineart / softedge: white lines on a black background
  * canny: white 1-px edges on black, extracted with the Canny operator

The OpenCV-based extractors below need no extra model downloads. If
`controlnet_aux` is installed, the learned HED / PiDiNet / Lineart annotators
can be used instead (modes ending in "_aux"), which is closer to what the
ControlNets saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

CONTROL_MODES = ("scribble", "lineart", "softedge", "canny", "softedge_aux", "lineart_aux")


@dataclass
class CleanConfig:
    size: int = 512                # output side length (square); must be divisible by 8
    background_blur: int = 41      # kernel for illumination flattening (odd)
    block_size: int = 31           # adaptive-threshold neighbourhood (odd)
    c: int = 12                    # adaptive-threshold offset; higher = fewer, darker lines kept
    min_component_px: int = 30     # remove specks smaller than this (at working resolution)
    stroke_px: int = 0             # 0 = keep stroke width, >0 = dilate/thin to roughly this width
    margin: float = 0.06           # white border added around the cropped drawing


def load_image(path_or_img) -> Image.Image:
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")
    return Image.open(path_or_img).convert("RGB")


def clean_sketch(img, cfg: CleanConfig | None = None) -> Image.Image:
    """Return a clean binary sketch (black lines on white), square, cfg.size px.

    Steps
    1. grayscale
    2. flatten uneven lighting: divide by a heavily blurred copy (background estimate)
    3. adaptive threshold: robust to remaining shading and paper texture
    4. remove small connected components (dust, paper grain)
    5. crop to the drawing's bounding box, pad to a square with a margin, resize
    6. optionally normalise stroke width
    """
    cfg = cfg or CleanConfig()
    gray = cv2.cvtColor(np.asarray(load_image(img)), cv2.COLOR_RGB2GRAY)

    bg = cv2.medianBlur(gray, cfg.background_blur | 1)
    flat = cv2.divide(gray, bg, scale=255)

    binary = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg.block_size | 1, cfg.c
    )  # lines = 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component_px
    binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    binary = _crop_pad_square(binary, cfg.margin)
    binary = cv2.resize(binary, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
    binary = np.where(binary > 96, 255, 0).astype(np.uint8)

    if cfg.stroke_px > 0:
        binary = _normalise_stroke(binary, cfg.stroke_px)
    return Image.fromarray(255 - binary).convert("RGB")  # black lines on white


def _crop_pad_square(lines: np.ndarray, margin: float) -> np.ndarray:
    ys, xs = np.nonzero(lines)
    if len(xs) == 0:
        return lines
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = lines[y0:y1, x0:x1]
    h, w = crop.shape
    side = int(max(h, w) * (1 + 2 * margin))
    out = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy : oy + h, ox : ox + w] = crop
    return out


def _normalise_stroke(lines: np.ndarray, width: int) -> np.ndarray:
    from skimage.morphology import skeletonize

    skel = skeletonize(lines > 0).astype(np.uint8) * 255
    if width <= 1:
        return skel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(skel, k)


def make_control(clean: Image.Image, mode: str = "scribble") -> Image.Image:
    """Turn a clean sketch (black on white) into the control image for `mode`."""
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode {mode!r}; choose from {CONTROL_MODES}")
    g = cv2.cvtColor(np.asarray(clean.convert("RGB")), cv2.COLOR_RGB2GRAY)
    lines = 255 - g  # white lines on black

    if mode == "scribble":
        # the scribble ControlNet was trained on thick, synthesised scribbles
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.dilate(lines, k)
    elif mode == "lineart":
        out = lines
    elif mode == "softedge":
        # HED-like soft boundaries: blur and re-normalise
        out = cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 1.5)
        out = np.clip(out / (out.max() + 1e-6) * 255 * 1.4, 0, 255).astype(np.uint8)
    elif mode == "canny":
        out = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 100, 200)
    elif mode == "softedge_aux":
        from controlnet_aux import PidiNetDetector

        det = _aux_cache("pidi", lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"))
        res = det(clean, detect_resolution=clean.width, image_resolution=clean.width, safe=True)
        return res.resize(clean.size)
    elif mode == "lineart_aux":
        from controlnet_aux import LineartDetector

        det = _aux_cache("lineart", lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"))
        res = det(clean, detect_resolution=clean.width, image_resolution=clean.width)
        return res.resize(clean.size)
    return Image.fromarray(out).convert("RGB")


_AUX: dict = {}


def _aux_cache(key, factory):
    if key not in _AUX:
        _AUX[key] = factory()
    return _AUX[key]


def sketch_lines_mask(clean: Image.Image) -> np.ndarray:
    """Boolean mask of sketch line pixels from a clean (black-on-white) sketch."""
    g = np.asarray(clean.convert("L"))
    return g < 128
