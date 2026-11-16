"""Config-driven experiment runner.

A YAML file defines default generation settings, the sketches with their
subject prompts, and named experiments. Each experiment sweeps one or more
parameters (cartesian product) over a list of seeds:

    experiments:
      E2_strength:
        description: "ControlNet conditioning scale"
        sweep: {cond_scale: [0.3, 0.6, 0.9, 1.2, 1.5]}
        seeds: [1, 2]
        overrides: {control: scribble}

Outputs (all under --out):
    controls/<sketch>__<mode>.png            control images
    <exp>/<sketch>/<run>.png                 generated images
    <exp>/metrics.csv                        one row per image, all settings + metrics
    <exp>/grid_<sketch>.jpg                  sketch | outputs comparison grid
    env.json                                 library versions and hardware

Already existing images are skipped, so an interrupted Kaggle session can be resumed.
"""
from __future__ import annotations

import csv
import itertools
import json
import platform
from dataclasses import fields
from pathlib import Path

import yaml
from PIL import Image

from .metrics import ClipScorer, preservation_metrics
from .pipeline import GenConfig, SketchGenerator
from .preprocess import CleanConfig, clean_sketch, make_control, sketch_lines_mask
from .prompts import build_prompt
from .visualize import save_grid

GEN_FIELDS = {f.name for f in fields(GenConfig)}


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _apply(params: dict) -> dict:
    """Expand convenience keys (control_window -> control_start/control_end)."""
    p = dict(params)
    if "control_window" in p:
        p["control_start"], p["control_end"] = p.pop("control_window")
    return p


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    return str(v)


def expand_runs(cfg: dict, only: list[str] | None = None, sketches: list[str] | None = None) -> list[dict]:
    runs = []
    defaults = cfg.get("defaults", {})
    all_sketches = cfg["sketches"]
    for exp_name, exp in cfg["experiments"].items():
        if only and exp_name not in only:
            continue
        sweep = exp.get("sweep", {})
        keys = list(sweep)
        combos = list(itertools.product(*[sweep[k] for k in keys])) or [()]
        seeds = exp.get("seeds", cfg.get("seeds", [1]))
        names = exp.get("sketches", list(all_sketches))
        for sk in names:
            if sketches and sk not in sketches:
                continue
            for combo in combos:
                swept = dict(zip(keys, combo))
                for seed in seeds:
                    params = {**defaults, **exp.get("overrides", {}), **swept, "seed": seed}
                    run_name = "__".join([f"{k}={_fmt(v)}" for k, v in swept.items()] + [f"seed={seed}"])
                    runs.append({"exp": exp_name, "sketch": sk, "swept": swept, "params": params,
                                 "run": run_name, "order": len(runs)})
    # group runs that share a model so each base/ControlNet is loaded once
    runs.sort(key=lambda r: (r["params"].get("base", "sd15"), r["params"].get("lcm", False), r["params"].get("control", "")))
    return runs


def make_gen_config(params: dict, subject: str) -> GenConfig:
    p = _apply(params)
    style = p.pop("style", "photoreal")
    prompt = build_prompt(subject, style, p.pop("prompt_extra", ""))
    known = {k: v for k, v in p.items() if k in GEN_FIELDS and k != "prompt"}
    return GenConfig(prompt=prompt, **known)


def write_env(out: Path, gen: SketchGenerator) -> None:
    import importlib.metadata as md

    import torch

    pkgs = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "peft", "opencv-python",
                 "opencv-python-headless", "scikit-image", "numpy", "Pillow", "controlnet-aux"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": gen.device,
        "dtype": str(gen.dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": pkgs,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(info, indent=2))


def run_experiments(
    config_path,
    sketch_dir,
    out_dir,
    only: list[str] | None = None,
    sketches: list[str] | None = None,
    clip: bool = True,
    dry_run: bool = False,
    generator: SketchGenerator | None = None,
) -> list[dict]:
    cfg = load_config(config_path)
    sketch_dir, out = Path(sketch_dir), Path(out_dir)
    runs = expand_runs(cfg, only, sketches)
    print(f"{len(runs)} runs planned")
    if dry_run:
        for r in runs:
            print(r["exp"], r["sketch"], r["run"])
        return runs

    gen = generator or SketchGenerator()
    write_env(out, gen)
    clip_scorer = ClipScorer(device=gen.device) if clip else None
    clean_cfg = CleanConfig(**cfg.get("preprocess", {}))

    clean_cache: dict[str, Image.Image] = {}
    control_cache: dict[tuple[str, str], Image.Image] = {}

    def get_clean(sk: str) -> Image.Image:
        if sk not in clean_cache:
            src = next(sketch_dir.glob(f"{sk}.*"))
            clean_cache[sk] = clean_sketch(src, clean_cfg)
            (out / "controls").mkdir(parents=True, exist_ok=True)
            clean_cache[sk].save(out / "controls" / f"{sk}__clean.png")
        return clean_cache[sk]

    def get_control(sk: str, mode: str, prep: str = "clean") -> Image.Image | None:
        """prep='clean' uses the full cleanup; prep='raw' feeds the unprocessed scan
        (only resized and inverted), to measure what preprocessing contributes."""
        if mode == "none":
            return None
        key = (sk, mode, prep)
        if key not in control_cache:
            if prep == "raw":
                src = Image.open(next(sketch_dir.glob(f"{sk}.*"))).convert("RGB")
                base_img = src.resize((clean_cfg.size, clean_cfg.size), Image.LANCZOS)
            else:
                base_img = get_clean(sk)
            control_cache[key] = make_control(base_img, mode)
            suffix = "" if prep == "clean" else f"__{prep}"
            control_cache[key].save(out / "controls" / f"{sk}__{mode}{suffix}.png")
        return control_cache[key]

    rows = []
    done: dict[str, tuple[Path, dict]] = {}  # config key -> (image path, meta) for reuse
    for i, r in enumerate(runs, 1):
        exp_dir = out / r["exp"]
        img_path = exp_dir / r["sketch"] / f"{r['run']}.png"
        subject = cfg["sketches"][r["sketch"]]
        gcfg = make_gen_config(r["params"], subject)
        clean = get_clean(r["sketch"])
        prep = r["params"].get("preprocess", "clean")
        key = json.dumps([r["sketch"], prep, gcfg.to_dict()], sort_keys=True, default=str)
        if img_path.exists():
            img, meta = Image.open(img_path).convert("RGB"), {"seconds": None, "resumed": True}
        elif key in done:  # identical configuration already generated in another experiment
            img, meta = Image.open(done[key][0]).convert("RGB"), {**done[key][1], "reused": True}
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
        else:
            control = get_control(r["sketch"], gcfg.control, r["params"].get("preprocess", "clean"))
            img, meta = gen.generate(gcfg, control)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
        done.setdefault(key, (img_path, meta))
        m = preservation_metrics(sketch_lines_mask(clean), img)
        if clip_scorer:
            m["clip_score"] = clip_scorer(img, gcfg.prompt)
        row = {"exp": r["exp"], "sketch": r["sketch"], "run": r["run"], "order": r["order"],
               "style": r["params"].get("style", "photoreal"),
               "preprocess": r["params"].get("preprocess", "clean"),
               **{k: _fmt(v) for k, v in r["swept"].items()},
               **{k: v for k, v in gcfg.to_dict().items() if k != "extra"}, **meta, **m, "image": str(img_path)}
        rows.append(row)
        _append_csv(exp_dir / "metrics.csv", row)
        print(f"[{i}/{len(runs)}] {r['exp']} {r['sketch']} {r['run']}  "
              f"recall={m['edge_recall']:.3f} clip={m.get('clip_score', float('nan'))}  t={meta.get('seconds')}")

    _grids(rows, out, sketch_dir)
    return rows


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:  # keep a stable header; new keys are appended as extra columns
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header and set(row) - set(header):
            header = header + [k for k in row if k not in header]
            _rewrite_with_header(path, header)
        fieldnames = header or list(row)
    else:
        fieldnames = list(row)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _rewrite_with_header(path: Path, header: list[str]) -> None:
    with open(path, newline="", encoding="utf-8") as fh:
        old = list(csv.DictReader(fh))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(old)


def _grids(rows: list[dict], out: Path, sketch_dir: Path) -> None:
    """One grid per experiment: a row per sketch (first seed only), original sketch in column 0."""
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        by_exp.setdefault(r["exp"], []).append(r)
    for exp, rs in by_exp.items():
        rs = sorted(rs, key=lambda r: r["order"])  # restore sweep order (runs were grouped by model)
        first_seed = min(int(r["seed"]) for r in rs)
        grid_rows = []
        for sk in dict.fromkeys(r["sketch"] for r in rs):
            cells = [("input sketch", Image.open(next(sketch_dir.glob(f"{sk}.*"))))]
            for r in rs:
                if r["sketch"] == sk and int(r["seed"]) == first_seed:
                    label = r["run"].rsplit("__seed=", 1)[0] or "default"
                    cells.append((label, Image.open(r["image"])))
            grid_rows.append((sk, cells))
        save_grid(grid_rows, out / exp / "grid.jpg")
