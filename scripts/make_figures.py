"""Build the report figures from the experiment outputs.

  python scripts/make_figures.py --results results --out report/figures
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketch2image.visualize import save_grid  # noqa: E402

SKETCHES = ["01_futuristic_tower", "02_robot_character", "03_hover_car", "04_desk_lamp", "05_floating_island"]
SHORT = {s: s[3:].replace("_", "\n") for s in SKETCHES}


def img(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sketches", default="sketches")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    R, S, O = Path(a.results), Path(a.sketches), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)

    # Fig 1: main showcase. Sketch, cleaned sketch, control map, result, three styles
    rows = []
    for s in SKETCHES:
        cells = [("input sketch", img(S / f"{s}.png")),
                 ("control (scribble)", img(R / "controls" / f"{s}__scribble.png")),
                 ("photoreal", img(R / "E4_style" / s / "style=photoreal__seed=1.png")),
                 ("concept art", img(R / "E4_style" / s / "style=concept_art__seed=1.png")),
                 ("watercolor", img(R / "E4_style" / s / "style=watercolor__seed=1.png")),
                 ("clay", img(R / "E4_style" / s / "style=clay__seed=1.png")),
                 ("cyberpunk", img(R / "E4_style" / s / "style=cyberpunk__seed=1.png"))]
        rows.append((SHORT[s], cells))
    save_grid(rows, O / "fig_showcase.jpg", tile=220, row_label_w=120)

    # Fig 2: control type
    rows = []
    for s in ["02_robot_character", "05_floating_island"]:
        cells = [("input sketch", img(S / f"{s}.png"))]
        for c in ["none", "scribble", "lineart", "softedge", "canny"]:
            cells.append((c if c != "none" else "none (text only)", img(R / "E1_control_type" / s / f"control={c}__seed=1.png")))
        rows.append((SHORT[s], cells))
    save_grid(rows, O / "fig_control_type.jpg", tile=220, row_label_w=120)

    # Fig 3: conditioning scale
    rows = []
    for s in ["03_hover_car"]:
        cells = [("input sketch", img(S / f"{s}.png"))]
        for v in ["0.3", "0.6", "1.0", "1.3", "1.6"]:
            cells.append((f"cond_scale {v}", img(R / "E2_cond_scale" / s / f"cond_scale={v}__seed=1.png")))
        rows.append((SHORT[s], cells))
    save_grid(rows, O / "fig_cond_scale.jpg", tile=220, row_label_w=120)

    # Fig 4: sampler + preprocessing, one sketch each (the control window is reported as a table)
    rows = []
    s = "01_futuristic_tower"
    rows.append(("sampler", [("input sketch", img(S / f"{s}.png"))] + [
        (f"LCM {n} steps", img(R / "E5_sampler" / s / f"steps={n}__seed=1.png")) for n in [2, 4, 8]] + [
        ("UniPC 20 steps, CFG 7.5", img(R / "E5b_standard_sampler" / s / "seed=1.png"))]))
    s = "05_floating_island"
    rows.append(("pre-\nprocessing", [("input sketch", img(S / f"{s}.png")),
                                      ("raw control", img(R / "controls" / f"{s}__scribble__raw.png")),
                                      ("result from raw", img(R / "E8_preprocess" / s / "preprocess=raw__seed=1.png")),
                                      ("clean control", img(R / "controls" / f"{s}__scribble.png")),
                                      ("result from clean", img(R / "E8_preprocess" / s / "preprocess=clean__seed=1.png"))]))
    save_grid(rows, O / "fig_ablations.jpg", tile=200, row_label_w=120)

    # Fig 5: seeds
    rows = []
    for s in ["02_robot_character", "03_hover_car"]:
        rows.append((SHORT[s], [("input sketch", img(S / f"{s}.png"))] + [
            (f"seed {k}", img(R / "E7_seeds" / s / f"seed={k}.png")) for k in [1, 2, 3]]))
    save_grid(rows, O / "fig_seeds.jpg", tile=200, row_label_w=120)
    print("figures written to", O)


if __name__ == "__main__":
    main()
