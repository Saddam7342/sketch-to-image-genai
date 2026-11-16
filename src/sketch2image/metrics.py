"""Quantitative evaluation: sketch preservation and prompt alignment.

Sketch preservation (structure)
-------------------------------
We extract an edge map from the generated image and compare it to the
skeleton (1-px centre line) of the clean input sketch, at the sketch's
resolution. With distance transforms we get, for every sketch pixel, the
distance to the nearest generated edge and vice versa:

* edge_recall@tau    – fraction of sketch pixels with a generated edge within tau px.
                       The main *preservation* score: "is every sketched line still there?"
* edge_precision@tau – fraction of generated edges lying on a sketch line. This is
                       *expected* to be low: added texture and detail are not in the sketch.
* edge_f1@tau        – harmonic mean of the two
* chamfer_s2g        – mean distance (px) from sketch pixels to the nearest generated edge

Recall must be read together with F1 / Chamfer: an image full of high-frequency
texture has edges everywhere and gets a high recall by chance (pure noise ~0.49,
precision ~0.14). Edge-map SSIM was evaluated and dropped: it is dominated by the
empty background (a blank image scored higher than a different drawing).

Prompt alignment (semantics)
----------------------------
* clip_score – 100 * max(cos(CLIP_image, CLIP_text), 0) (Hessel et al., 2021)
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def generated_edges(img: Image.Image, size: int) -> np.ndarray:
    """Canny edges of a generated image with thresholds set from its median intensity."""
    g = cv2.cvtColor(np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS)), cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)  # suppress fine texture, keep object boundaries
    med = float(np.median(g))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    lo, hi = max(lo, 30), max(hi, 90)
    return cv2.Canny(g, lo, hi) > 0


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel in mask."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def preservation_metrics(sketch_mask: np.ndarray, generated: Image.Image, tau: float = 3.0) -> dict:
    size = sketch_mask.shape[0]
    sk = skeletonize(sketch_mask)
    ge = generated_edges(generated, size)

    d_s2g = _dist_to(ge)[sk]
    d_g2s = _dist_to(sk)[ge]
    recall = float((d_s2g <= tau).mean()) if d_s2g.size else 0.0
    precision = float((d_g2s <= tau).mean()) if d_g2s.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    chamfer = float(np.minimum(d_s2g, size).mean()) if d_s2g.size else float(size)
    return {
        "edge_recall": round(recall, 4),
        "edge_precision": round(precision, 4),
        "edge_f1": round(f1, 4),
        "chamfer_s2g": round(chamfer, 3),
    }


class ClipScorer:
    """Lazy CLIP wrapper; the model is downloaded on first use."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._proc = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self._proc = CLIPProcessor.from_pretrained(self.model_id)

    def __call__(self, image: Image.Image, text: str) -> float:
        import torch

        self._load()
        inputs = self._proc(text=[text], images=[image.convert("RGB")], return_tensors="pt",
                            padding=True, truncation=True, max_length=77).to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)
        img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return round(float(100.0 * torch.clamp((img * txt).sum(), min=0)), 3)
