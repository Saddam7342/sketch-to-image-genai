# Bringing Sketches to Life: Sketch-to-Image with Stable Diffusion + ControlNet

*Author: Hina Kanwal (Student ID 32520481). Individual project for the course module "Generative AI", Session 2.*

A configurable pipeline that turns rough hand-drawn sketches into realistic or stylised images while
preserving the sketch's structure, together with a **quantitative evaluation** of how well that
structure survives (edge-based preservation metrics + CLIP score) and a systematic set of ablation
experiments.

```
sketch photo/scan ─► preprocessing ─► control map ─► SD 1.5 / SDXL + ControlNet ─► image ─► metrics
                    (lighting fix,    (scribble,     (+ text prompt: subject        (edge recall/F1,
                     threshold,        lineart,        + style preset,               Chamfer, CLIP)
                     denoise, crop)    soft edge,      negative prompt)
                                       canny)
```

## How the sketch steers the image

Stable Diffusion generates an image by iteratively denoising a latent tensor (64×64×4 for 512 px), with
a U-Net conditioned on the CLIP embedding of the text prompt. **ControlNet** adds a trainable copy of the
U-Net encoder that receives the processed sketch. Its feature maps pass through *zero-initialised* 1×1
convolutions and are **added to the skip connections of the frozen U-Net** at every resolution, so the
sketch fixes *where* things are (layout, contours), while the prompt decides *what they look like*
(materials, lighting, style). The main control knobs, all studied in the experiments:

| Parameter | Effect |
|---|---|
| `cond_scale` | weight of the ControlNet residuals: low = free interpretation, high = strict line following |
| `control_start / control_end` | part of the denoising trajectory in which ControlNet is active (early steps = layout, late steps = detail) |
| control representation | scribble / lineart / soft edge / canny: what the ControlNet was trained to read |
| `guidance_scale` | classifier-free guidance strength for the text prompt |
| `steps`, scheduler, LCM-LoRA | quality vs. speed (LCM: 4–8 steps) |

## Repository layout

```
configs/experiments.yaml     all experiment definitions (sweeps, seeds, prompts)
configs/smoke.yaml           fast CPU test with tiny random models
sketches/                    the 5 input sketches (+ provenance)
src/sketch2image/
  preprocess.py              sketch cleanup + control-map extraction
  pipeline.py                model registry, loading/caching, generation
  prompts.py                 style presets, negative prompt
  metrics.py                 edge recall / precision / F1, Chamfer distance, CLIP score
  experiments.py             config-driven sweep runner (resumable, CSV logging, grids)
  visualize.py               comparison grids, metric plots
scripts/
  make_sketches.py           regenerates the input sketches
  generate.py                single-sketch generation CLI
  run_experiments.py         runs the sweeps
  analyze.py                 plots + Markdown summary from the CSVs
notebooks/kaggle_run_all.ipynb   reproduces every result on a free Kaggle GPU
docs/KAGGLE_GUIDE.md         step-by-step guide for Kaggle
results/                     grids, metrics CSVs, analysis plots (after a run)
```

## Installation

**Kaggle / Colab (GPU, recommended):** follow [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) and run
`notebooks/kaggle_run_all.ipynb`.

**Local:**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install torch                  # pick the right build from https://pytorch.org
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: controlnet-aux, gradio
```

A CUDA GPU with ≥ 8 GB VRAM is recommended for SD 1.5 (≥ 12 GB for SDXL). On CPU, use LCM-LoRA
(`--lcm --steps 4 --guidance 1.5`): about 1–3 min per 512 px image.

## Usage

```bash
# one sketch, several seeds
python scripts/generate.py --sketch sketches/03_hover_car.png \
    --subject "a sleek futuristic hover car floating above the road" \
    --style concept_art --control scribble --cond-scale 0.9 --seeds 1 2 3

# list / run the experiment sweeps, then analyse
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --only E1_control_type E2_cond_scale
python scripts/analyze.py

# CPU smoke test of all code paths (tiny random models, ~1 min)
python scripts/run_experiments.py --config configs/smoke.yaml --out results_smoke --no-clip
```

## Experiments

All runs use fixed seeds; every image's full settings and metrics are logged in `results/<exp>/metrics.csv`.

| ID | Question | Sweep |
|---|---|---|
| E1 | Which control representation preserves the sketch best? | none (text only), scribble, lineart, soft edge, canny |
| E1b | Learned vs. hand-crafted control maps | PiDiNet soft edge, learned lineart |
| E2 | Sketch adherence vs. creative freedom | `cond_scale` 0.2 → 1.6 |
| E3 | When does structure get decided? | ControlNet active over [0,1], [0,0.6], [0,0.3], [0.3,1] |
| E4 | Style transfer from the prompt | photoreal, concept art, watercolor, clay, cyberpunk |
| E5 | Text guidance and step count | CFG {3, 7.5, 12} × steps {15, 30, 50} |
| E6 | Few-step sampling | LCM-LoRA, 2 / 4 / 8 steps |
| E7 | Seed variance | 6 seeds |
| E8 | Does preprocessing matter? | raw scan vs. cleaned sketch × {scribble, lineart} |
| E9 | Bigger base model | SDXL + xinsir scribble ControlNet |

### Metrics

- **Edge recall@3px**: share of sketch line pixels (skeleton) that have an edge of the generated image within 3 px. Main *preservation* score.
- **Edge F1@3px**: also penalises edges that do not belong to the sketch, which guards against noisy, texture-heavy images that hit the sketch by chance.
- **Chamfer distance** (sketch → image, px at 512): mean distance from sketch lines to the nearest generated edge.
- **CLIP score**: image–prompt agreement (Hessel et al., 2021).

The metrics were validated on known cases: identical image 1.0, 4 px shift ≈ 1.0, 15 px shift ≈ 0.53,
a different drawing ≈ 0.14, blank image 0.

## Results

*Filled in after the Kaggle run: comparison grids in `results/<exp>/grid.jpg`, plots and tables in
`results/analysis/`.*

## Reproducibility

- Seeds are fixed per run (`torch.Generator("cpu").manual_seed(seed)`); results are reproducible on the same GPU type and library versions (minor differences across GPU models are expected with fp16).
- Library versions and the GPU are recorded in `results/env.json`.
- Main settings: SD 1.5, 512×512, UniPC scheduler, 30 steps, CFG 7.5, `cond_scale` 1.0, fp16 on CUDA.
- Input sketches are regenerated deterministically by `scripts/make_sketches.py`.

## References and acknowledgements

See [REFERENCES.md](REFERENCES.md) for all papers, models (with licenses), libraries and the AI-assistance statement.
