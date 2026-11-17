# How the project works: a one-page study sheet

## 1. The idea in one sentence
A pretrained text-to-image model (Stable Diffusion 1.5) draws the picture. A second network (ControlNet)
reads my sketch and pushes the drawing to follow its lines, while the text prompt decides colours,
materials, light and style.

## 2. The pipeline, step by step
1. **Clean the sketch** (`preprocess.py`). Grayscale. Then divide by a blurred copy to remove uneven
   lighting. Then an adaptive threshold separates ink from paper. Remove tiny specks. Crop to the drawing
   and resize to 512×512.
2. **Make the control image.** ControlNet expects **white lines on black**. *Scribble* = thickened lines.
   *Lineart* = thin lines. *Soft edge* = blurred lines. *Canny* = edge detector applied to the sketch. Canny
   gives double lines, because each pencil stroke has two edges.
3. **Generate** (`pipeline.py`). Stable Diffusion works in a compressed *latent* space (64×64×4 for a
   512 px image). It starts from random noise and removes noise step by step. At every step the
   **U-Net** predicts the noise. It sees the text through **CLIP** text embeddings (cross-attention).
4. **ControlNet** is a trainable copy of the U-Net's encoder. It gets the control image. Its outputs go
   through **zero-initialised 1×1 convolutions** and are **added** to the U-Net's skip connections. Because
   of the zero init, an untrained ControlNet changes nothing. My smoke test with random weights showed
   exactly this: identical images with and without the sketch.
5. **Measure** (`metrics.py`). Detect edges in the generated image. Compare them with the sketch's
   centre lines:
   - **edge recall** = share of sketch lines that reappear (within 3 px)
   - **F1** also punishes edges that are not in the sketch
   - **Chamfer distance** = average distance from sketch lines to the image's edges
   - **CLIP score** = how well the image matches the prompt

## 3. The knobs and what they do
| Knob | Meaning | Low value | High value |
|---|---|---|---|
| `cond_scale` | strength of ControlNet's added signal | model ignores the sketch | strict tracing, may look flat or drawn |
| control window | steps in which ControlNet is active | — | early steps fix layout, late steps fix detail |
| `guidance_scale` (CFG) | how strongly the text is followed: ε = ε_uncond + w·(ε_text − ε_uncond) | weak prompt | over-saturated |
| steps | number of denoising steps | fast, rough | slow, refined |
| LCM-LoRA | small add-on weights (LoRA) distilled so that 2–8 steps suffice | — | — |

## 4. Why LCM-LoRA? (the CPU constraint)
My laptop has no GPU. A normal 20-step image with CFG took about 9.5 min. LCM-LoRA needs only 4 steps and
works without CFG (guidance 1.0), so one image takes about 70 s. LoRA means low-rank update matrices (A·B)
added to the frozen weights. The LCM LoRA was trained by *consistency distillation* to jump to the clean
image in a few steps.

## 5. The agent (bonus)
A rule-based loop (`agent.py`):
1. Pick ControlNet and strength from the line density.
2. Generate.
3. Measure edge F1 and CLIP.
4. If the structure is too weak, raise `cond_scale` or switch to lineart. If the structure is fine but CLIP
   is low, lower `cond_scale`.
5. Repeat at most 3 times and keep the best attempt.

It is explainable: every decision is logged.

## 6. Likely questions
- *Did you train a model?* No. I used pretrained SD 1.5, ControlNet v1.1 and LCM-LoRA, and configured them in my own pipeline. The assignment allows this.
- *Why is recall alone not enough?* A noisy image has edges everywhere and hits the sketch by chance (noise ≈ 0.49 recall). F1 and Chamfer catch that.
- *Why does the text-only baseline score low?* Without ControlNet the model invents its own layout, so the sketch lines are missing.
- *Limitations?* Edge metrics reward tracing, not beauty. Canny edges on photos also catch texture. All of this runs on a CPU, so there are few seeds. Results are from SD 1.5 at 512 px.
