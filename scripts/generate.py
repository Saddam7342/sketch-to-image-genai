"""Generate images from a single sketch.

Example
  python scripts/generate.py --sketch sketches/02_robot_character.png \
      --subject "a friendly robot waving" --style clay --control scribble --cond-scale 0.9 --seeds 1 2 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sketch2image.pipeline import GenConfig, SketchGenerator  # noqa: E402
from sketch2image.preprocess import CleanConfig, clean_sketch, make_control  # noqa: E402
from sketch2image.prompts import STYLES, build_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--subject", required=True, help="what the sketch shows")
    ap.add_argument("--style", default="photoreal", choices=sorted(STYLES))
    ap.add_argument("--extra", default="", help="extra prompt text")
    ap.add_argument("--base", default="sd15", choices=["sd15", "sdxl", "tiny"])
    ap.add_argument("--control", default="scribble")
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--control-start", type=float, default=0.0)
    ap.add_argument("--control-end", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--scheduler", default="unipc")
    ap.add_argument("--lcm", action="store_true", help="LCM-LoRA (use ~4-8 steps, guidance ~1.5)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clean = clean_sketch(a.sketch, CleanConfig(size=512))
    control = make_control(clean, a.control) if a.control != "none" else None
    stem = Path(a.sketch).stem
    clean.save(out / f"{stem}__clean.png")
    if control is not None:
        control.save(out / f"{stem}__control_{a.control}.png")

    gen = SketchGenerator()
    for seed in a.seeds:
        cfg = GenConfig(prompt=build_prompt(a.subject, a.style, a.extra), base=a.base, control=a.control,
                        steps=a.steps, guidance_scale=a.guidance, cond_scale=a.cond_scale,
                        control_start=a.control_start, control_end=a.control_end,
                        seed=seed, scheduler=a.scheduler, lcm=a.lcm)
        img, meta = gen.generate(cfg, control)
        path = out / f"{stem}__{a.style}__{a.control}__seed{seed}.png"
        img.save(path)
        (path.with_suffix(".json")).write_text(json.dumps({**cfg.to_dict(), **meta}, indent=2))
        print("saved", path, meta)


if __name__ == "__main__":
    main()
