"""Aggregate experiment CSVs into plots and a Markdown summary for the report.

  python scripts/analyze.py --config configs/experiments.yaml --results results

Writes results/analysis/<exp>.png (metric vs. swept setting, mean ± std over
sketches and seeds) and results/analysis/summary.md (tables).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketch2image.visualize import plot_metric_vs_param  # noqa: E402

METRICS = ["edge_recall", "edge_f1", "chamfer_s2g", "clip_score"]


def setting_column(df: pd.DataFrame, swept: list[str]) -> str:
    if len(swept) == 1 and swept[0] in df.columns:
        return swept[0]
    df["setting"] = df["run"].str.rsplit("__seed=", n=1).str[0]
    return "setting"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    res = Path(a.results)
    out = res / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    md = ["# Experiment summary", "",
          "Mean ± std over all sketches and seeds. edge_recall / edge_f1: higher = sketch better preserved; "
          "chamfer_s2g (px at 512): lower = better; clip_score: higher = closer to the prompt.", ""]
    for exp, spec in cfg["experiments"].items():
        csv = res / exp / "metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        metrics = [m for m in METRICS if m in df.columns]
        swept = list(spec.get("sweep", {})) or ["seed"]
        col = setting_column(df, swept)
        df = df.sort_values("order")
        order = list(dict.fromkeys(df[col]))
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)

        plot_metric_vs_param(df, col, metrics, out / f"{exp}.png", title=f"{exp}: {spec.get('description', '')}"[:110])

        agg = df.groupby(col, observed=True)[metrics].agg(["mean", "std"])
        table = pd.DataFrame({m: [f"{mu:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{mu:.3f}"
                                  for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")])] for m in metrics},
                             index=agg.index)
        if "seconds" in df.columns and df["seconds"].notna().any():
            table["sec/img"] = df.groupby(col, observed=True)["seconds"].mean().round(2).astype(str)
        md += [f"## {exp}", "", spec.get("description", ""), "", table.to_markdown(), "",
               f"![{exp}]({exp}.png)", ""]

        per_sketch = df.groupby(["sketch", col], observed=True)["edge_f1"].mean().unstack(col).round(3)
        md += ["Edge F1 per sketch:", "", per_sketch.to_markdown(), ""]

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
