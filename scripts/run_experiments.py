"""Run the experiment sweeps defined in a YAML config.

Examples
  python scripts/run_experiments.py --config configs/experiments.yaml --dry-run
  python scripts/run_experiments.py --config configs/experiments.yaml --only E1_control_type E2_cond_scale
  python scripts/run_experiments.py --config configs/smoke.yaml --out results_smoke --no-clip   # CPU test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sketch2image.experiments import run_experiments  # noqa: E402
from sketch2image.pipeline import SketchGenerator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--sketches", default="sketches", help="folder with the input sketches")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="experiment names to run (default: all)")
    ap.add_argument("--sketch", nargs="*", help="restrict to these sketch names")
    ap.add_argument("--no-clip", action="store_true", help="skip the CLIP score")
    ap.add_argument("--cpu-offload", action="store_true", help="save GPU memory (slower)")
    ap.add_argument("--no-safety-checker", action="store_true",
                    help="skip the SD1.5 NSFW filter (saves a 600 MB download; prompts here are fixed and benign)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gen = None if a.dry_run else SketchGenerator(cpu_offload=a.cpu_offload, safety_checker=not a.no_safety_checker)
    run_experiments(a.config, a.sketches, a.out, only=a.only, sketches=a.sketch,
                    clip=not a.no_clip, dry_run=a.dry_run, generator=gen)


if __name__ == "__main__":
    main()
