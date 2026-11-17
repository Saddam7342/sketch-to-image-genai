"""Run the SketchAgent on a few sketches and save every attempt plus the decision log.

  python scripts/agent_demo.py --out results/agent --f1-target 0.30
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketch2image.agent import SketchAgent  # noqa: E402
from sketch2image.pipeline import SketchGenerator  # noqa: E402
from sketch2image.visualize import save_grid  # noqa: E402

REQUESTS = [
    ("03_hover_car", "a watercolor painting of a futuristic hover car"),
    ("04_desk_lamp", "a product render of a modern desk lamp"),
    ("05_floating_island", "an epic fantasy floating island with a cabin and a waterfall"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketches", default="sketches")
    ap.add_argument("--out", default="results/agent")
    ap.add_argument("--f1-target", type=float, default=0.30)
    ap.add_argument("--clip-target", type=float, default=26.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = SketchAgent(generator=SketchGenerator(safety_checker=False), fast=True, f1_target=a.f1_target,
                        clip_target=a.clip_target, captioner=None)
    rows, summary = [], []
    for name, text in REQUESTS:
        res = agent.run(Path(a.sketches) / f"{name}.png", user_text=text, seed=1)
        cells = [("input sketch", Image.open(Path(a.sketches) / f"{name}.png"))]
        for i, (im, m, cfg) in enumerate(res.attempts, 1):
            im.save(out / f"{name}__attempt{i}.png")
            cells.append((f"#{i} {cfg.control} {cfg.cond_scale:.1f} F1={m['edge_f1']:.2f}", im))
        rows.append((name[3:], cells))
        summary.append({"sketch": name, "request": text, "log": res.log,
                        "attempts": [{**m, "control": c.control, "cond_scale": c.cond_scale, "prompt": c.prompt}
                                     for _, m, c in res.attempts]})
        print("\n".join(res.log), flush=True)
    save_grid(rows, out / "agent_attempts.jpg", tile=220, row_label_w=120)
    (out / "agent_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
