"""Web UI: upload or draw a sketch, describe it (optional), get an image.

  python app/gradio_app.py            # then open http://127.0.0.1:7860

Two modes
* Manual: you choose ControlNet type, conditioning scale, style, steps...
* Agent: the SketchAgent plans, generates, checks sketch preservation and retries automatically.
On a CPU-only machine keep "Fast (LCM)" enabled (~30-60 s per image).
"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketch2image.agent import SketchAgent  # noqa: E402
from sketch2image.metrics import preservation_metrics  # noqa: E402
from sketch2image.pipeline import GenConfig, SketchGenerator  # noqa: E402
from sketch2image.preprocess import CleanConfig, clean_sketch, make_control, sketch_lines_mask  # noqa: E402
from sketch2image.prompts import STYLES, build_prompt  # noqa: E402

GEN = SketchGenerator()
AGENT = None
EXAMPLES = sorted(str(p) for p in (Path(__file__).resolve().parents[1] / "sketches").glob("*.png"))


def _to_pil(sketch) -> Image.Image:
    """Accept an uploaded image or the ImageEditor's dict output."""
    if isinstance(sketch, dict):  # gr.ImageEditor -> {"background", "layers", "composite"}
        sketch = sketch.get("composite")
    if sketch is None:
        raise gr.Error("Please upload or draw a sketch first.")
    img = Image.fromarray(sketch) if isinstance(sketch, np.ndarray) else sketch
    if img.mode == "RGBA":  # drawn strokes on a transparent canvas -> put on white paper
        bg = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def manual(sketch, subject, style, control, cond, start, end, fast, steps, guidance, seed):
    img_in = _to_pil(sketch)
    clean = clean_sketch(img_in, CleanConfig(size=512))
    ctrl = make_control(clean, control)
    cfg = GenConfig(prompt=build_prompt(subject or "an object", style), control=control, cond_scale=cond,
                    control_start=start, control_end=end, lcm=fast, steps=int(steps),
                    guidance_scale=guidance, seed=int(seed))
    out, meta = GEN.generate(cfg, ctrl)
    m = preservation_metrics(sketch_lines_mask(clean), out)
    info = (f"prompt: {cfg.prompt}\n\nedge recall {m['edge_recall']:.3f} | edge F1 {m['edge_f1']:.3f} | "
            f"Chamfer {m['chamfer_s2g']:.2f} px | {meta['seconds']:.1f} s on {meta['device']}")
    return ctrl, out, info


def agent(sketch, text, seed):
    global AGENT
    if AGENT is None:
        AGENT = SketchAgent(generator=GEN, fast=GEN.device == "cpu", use_clip=True)
    res = AGENT.run(_to_pil(sketch), user_text=text or "", seed=int(seed))
    gallery = [(im, f"attempt {i + 1}: F1 {m['edge_f1']:.2f}") for i, (im, m, _) in enumerate(res.attempts)]
    return res.image, gallery, "\n".join(res.log)


with gr.Blocks(title="Sketch to Image") as demo:
    gr.Markdown("# Sketch → Image\nStable Diffusion 1.5 + ControlNet. Upload a sketch or draw one.")
    with gr.Tab("Manual"):
        with gr.Row():
            with gr.Column():
                sk = gr.Image(label="Sketch (upload)", type="pil")
                gr.Examples(EXAMPLES, inputs=sk)
                subject = gr.Textbox(label="What is it?", value="a friendly small robot character waving")
                style = gr.Dropdown(sorted(STYLES), value="photoreal", label="Style")
                control = gr.Dropdown(["scribble", "lineart", "softedge", "canny"], value="scribble", label="ControlNet")
                cond = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="Conditioning scale (sketch strength)")
                with gr.Row():
                    start = gr.Slider(0, 1, 0.0, step=0.05, label="Control start")
                    end = gr.Slider(0, 1, 1.0, step=0.05, label="Control end")
                fast = gr.Checkbox(GEN.device == "cpu", label="Fast (LCM-LoRA, few steps)")
                with gr.Row():
                    steps = gr.Slider(2, 50, 6 if GEN.device == "cpu" else 30, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, 1.5 if GEN.device == "cpu" else 7.5, step=0.5, label="Guidance")
                    seed = gr.Number(1, label="Seed", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                ctrl_out = gr.Image(label="Control map fed to ControlNet")
                img_out = gr.Image(label="Result")
                info = gr.Textbox(label="Details", lines=4)
        btn.click(manual, [sk, subject, style, control, cond, start, end, fast, steps, guidance, seed],
                  [ctrl_out, img_out, info])

    with gr.Tab("Agent"):
        gr.Markdown("Draw or upload a sketch and optionally say what you want. The agent picks the settings, "
                    "checks how well the sketch was preserved and retries if needed.")
        with gr.Row():
            with gr.Column():
                canvas = gr.ImageEditor(label="Draw or upload", type="pil", image_mode="RGBA",
                                        brush=gr.Brush(colors=["#000000"], default_size=4))
                text = gr.Textbox(label="Optional request", placeholder="e.g. a watercolor painting of a lighthouse")
                aseed = gr.Number(1, label="Seed", precision=0)
                abtn = gr.Button("Run agent", variant="primary")
            with gr.Column():
                best = gr.Image(label="Selected result")
                gal = gr.Gallery(label="All attempts", columns=3, height=220)
                log = gr.Textbox(label="Agent log", lines=10)
        abtn.click(agent, [canvas, text, aseed], [best, gal, log])

if __name__ == "__main__":
    demo.queue().launch()
