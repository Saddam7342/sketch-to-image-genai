"""Prompt construction: subject (from the sketch) + style preset + negative prompt."""
from __future__ import annotations

DEFAULT_NEGATIVE = (
    "lowres, blurry, out of focus, deformed, disfigured, bad anatomy, extra limbs, "
    "watermark, text, signature, jpeg artifacts, worst quality, low quality, sketch, lineart"
)

STYLES: dict[str, str] = {
    "photoreal": "photorealistic photograph, natural lighting, 35mm lens, highly detailed, sharp focus, 8k",
    "concept_art": "digital concept art, cinematic lighting, dramatic atmosphere, matte painting, highly detailed",
    "watercolor": "watercolor painting, soft washes, visible paper texture, delicate ink outlines, pastel palette",
    "clay": "3d clay render, plasticine, soft studio lighting, pastel colors, cute, tilt-shift",
    "cyberpunk": "cyberpunk night scene, neon lights, rain reflections, volumetric fog, blade-runner mood",
    "blueprint_render": "product design render, studio softbox lighting, white background, brushed metal and matte plastic",
    "none": "",
}


def build_prompt(subject: str, style: str = "photoreal", extra: str = "") -> str:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLES)}")
    parts = [subject.strip(), STYLES[style], extra.strip()]
    return ", ".join(p for p in parts if p)
