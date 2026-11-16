"""Comparison grids (sketch | outputs) and metric plots for the report."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def comparison_grid(
    rows: list[tuple[str, list[tuple[str, Image.Image]]]],
    tile: int = 256,
    label_h: int = 26,
    row_label_w: int = 150,
) -> Image.Image:
    """rows = [(row_label, [(col_label, image), ...]), ...] -> one labelled grid image."""
    ncol = max(len(r[1]) for r in rows)
    W = row_label_w + ncol * tile
    H = len(rows) * (tile + label_h)
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    f, fr = _font(15), _font(16)
    for ri, (rlabel, cells) in enumerate(rows):
        y = ri * (tile + label_h)
        d.multiline_text((8, y + label_h + tile // 2 - 20), rlabel.replace("_", "\n", 1), fill="black", font=fr)
        for ci, (clabel, im) in enumerate(cells):
            x = row_label_w + ci * tile
            grid.paste(im.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y + label_h))
            d.text((x + 4, y + 5), clabel[:34], fill="black", font=f)
    return grid


def save_grid(rows, path: Path, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_grid(rows, **kw).save(path, quality=92)
    return path


def plot_metric_vs_param(df, param: str, metrics: list[str], path: Path, title: str = "") -> Path:
    """Mean ± std of each metric over sketches and seeds, as a function of a swept parameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = df.groupby(param)[metrics].agg(["mean", "std"])
    xs = list(agg.index)
    numeric = all(isinstance(x, (int, float)) for x in xs)
    pos = xs if numeric else list(range(len(xs)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        mu, sd = agg[(m, "mean")].values, agg[(m, "std")].fillna(0).values
        ax.errorbar(pos, mu, yerr=sd, marker="o", capsize=4)
        ax.set_title(m)
        ax.set_xlabel(param)
        if not numeric:
            ax.set_xticks(pos, [str(x) for x in xs], rotation=30, ha="right")
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
