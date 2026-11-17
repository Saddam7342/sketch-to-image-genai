"""Trade-off plot: sketch preservation (edge F1) vs. prompt alignment (CLIP) for every configuration.

  python scripts/make_tradeoff.py --results results --out report/figures/fig_tradeoff.png
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT = (0.0, 0.0)
GROUPS = [  # (experiment, column, label prefix, marker)
    ("E1_control_type", "control", "", "o"),
    ("E2_cond_scale", "cond_scale", "scale ", "s"),
    ("E3_control_window", None, "window ", "^"),
    ("E5_sampler", "steps", "LCM ", "D"),
    ("E5b_standard_sampler", None, "UniPC 20 + CFG", "*"),
    ("E8_preprocess", "preprocess", "", "P"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures/fig_tradeoff.png")
    a = ap.parse_args()
    d0 = pd.read_csv(Path(a.results) / "E1_control_type" / "metrics.csv")
    global DEFAULT
    DEFAULT = tuple(d0[d0.control == "scribble"][["edge_f1", "clip_score"]].mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for exp, col, prefix, mk in GROUPS:
        d = pd.read_csv(Path(a.results) / exp / "metrics.csv")
        if exp == "E3_control_window":
            d["key"] = d.control_start.astype(str) + "-" + d.control_end.astype(str)
        elif col is None:
            d["key"] = ""
        else:
            d["key"] = d[col].astype(str)
        g = d.groupby("key")[["edge_f1", "clip_score"]].mean()
        ax.scatter(g.edge_f1, g.clip_score, marker=mk, s=40, label=exp.split("_", 1)[1].replace("_", " "))
        for k, r in g.iterrows():
            if abs(r.edge_f1 - DEFAULT[0]) < 1e-3 and abs(r.clip_score - DEFAULT[1]) < 1e-3:
                continue  # the shared default configuration is labelled once below
            dy = -9 if f"{prefix}{k}" in ("window 0.0-0.5",) else 3
            ax.annotate(f"{prefix}{k}", (r.edge_f1, r.clip_score), fontsize=6.5, xytext=(4, dy),
                        textcoords="offset points")
    d = pd.read_csv(Path(a.results) / "E1_control_type" / "metrics.csv")
    dflt = d[d.control == "scribble"][["edge_f1", "clip_score"]].mean()
    ax.annotate("default: scribble, scale 1.0, LCM 4", (dflt.edge_f1, dflt.clip_score), fontsize=6.5,
                xytext=(-10, -16), textcoords="offset points", ha="right")
    ax.set_xlabel("edge F1 (sketch preservation)  →")
    ax.set_ylabel("CLIP score (prompt alignment)  →")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
