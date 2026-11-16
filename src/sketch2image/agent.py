"""SketchAgent: automates sketch -> final image with a self-correcting quality loop.

    receive sketch (+ optional text)
      1. clean the sketch, extract control map
      2. understand the request: subject from user text (or a BLIP caption of the sketch),
         style from keywords in the text ("watercolor", "neon", ...)
      3. plan: pick start settings (ControlNet type, conditioning scale)
      4. generate
      5. evaluate: edge F1 (structure) and, if available, CLIP score (prompt match)
      6. reflect: structure too weak  -> raise cond_scale / switch to stricter lineart
                  structure fine but image looks like a line drawing (low CLIP) -> lower cond_scale
      7. stop when the targets are met or the budget is used up; return the best attempt + a log

The policy is rule-based on purpose: every decision is explainable and logged, which makes the
agent's behaviour easy to analyse in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .metrics import ClipScorer, preservation_metrics
from .pipeline import GenConfig, SketchGenerator
from .preprocess import CleanConfig, clean_sketch, make_control, sketch_lines_mask
from .prompts import STYLES, build_prompt

STYLE_KEYWORDS = {
    "watercolor": ["watercolor", "watercolour", "painting", "aquarelle"],
    "clay": ["clay", "plasticine", "toy", "cute"],
    "cyberpunk": ["cyberpunk", "neon", "night", "futuristic city"],
    "concept_art": ["concept", "cinematic", "fantasy", "epic", "game art"],
    "blueprint_render": ["product", "render", "studio", "industrial design"],
    "photoreal": ["photo", "realistic", "real", "photograph"],
}


def detect_style(text: str, default: str = "photoreal") -> str:
    t = text.lower()
    for style, words in STYLE_KEYWORDS.items():
        if any(w in t for w in words):
            return style
    return default


@dataclass
class AgentResult:
    image: Image.Image
    config: GenConfig
    metrics: dict
    log: list[str] = field(default_factory=list)
    attempts: list[tuple[Image.Image, dict, GenConfig]] = field(default_factory=list)


class SketchAgent:
    def __init__(
        self,
        generator: SketchGenerator | None = None,
        fast: bool = True,                 # LCM-LoRA few-step sampling (CPU friendly)
        f1_target: float = 0.30,
        clip_target: float = 26.0,
        max_attempts: int = 3,
        use_clip: bool = True,
        captioner: str | None = "Salesforce/blip-image-captioning-base",
    ):
        self.gen = generator or SketchGenerator()
        self.fast = fast
        self.f1_target = f1_target
        self.clip_target = clip_target
        self.max_attempts = max_attempts
        self.clip = ClipScorer(device=self.gen.device) if use_clip else None
        self.captioner_id = captioner
        self._captioner = None

    # ------------------------------------------------------------ perception
    def caption(self, clean: Image.Image) -> str:
        """Describe the sketch with BLIP when the user gives no subject."""
        if self.captioner_id is None:
            return "an object"
        if self._captioner is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._captioner = (BlipProcessor.from_pretrained(self.captioner_id),
                               BlipForConditionalGeneration.from_pretrained(self.captioner_id).to(self.gen.device))
        proc, model = self._captioner
        inputs = proc(images=clean.convert("RGB"), text="a drawing of", return_tensors="pt").to(self.gen.device)
        out = model.generate(**inputs, max_new_tokens=25)
        text = proc.decode(out[0], skip_special_tokens=True)
        return text.replace("a drawing of", "").replace("a sketch of", "").strip() or "an object"

    # -------------------------------------------------------------- main loop
    def run(self, sketch, user_text: str = "", style: str | None = None, seed: int = 1) -> AgentResult:
        log: list[str] = []
        clean = clean_sketch(sketch, CleanConfig(size=512))
        mask = sketch_lines_mask(clean)
        density = float(mask.mean())

        subject = user_text.strip()
        if not subject:
            subject = self.caption(clean)
            log.append(f"no text given -> BLIP caption: '{subject}'")
        style = style or detect_style(user_text)
        if style not in STYLES:
            style = "photoreal"
        log.append(f"style: {style}")

        # plan: dense, detailed drawings -> lineart (thin precise lines); sparse -> scribble (tolerant)
        mode = "lineart" if density > 0.06 else "scribble"
        cond = 0.9
        log.append(f"line density {density:.3f} -> start with ControlNet '{mode}', cond_scale {cond}")

        steps, cfg_scale = (6, 1.5) if self.fast else (30, 7.5)
        attempts = []
        best = None
        for k in range(1, self.max_attempts + 1):
            cfg = GenConfig(prompt=build_prompt(subject, style), control=mode, cond_scale=cond,
                            steps=steps, guidance_scale=cfg_scale, lcm=self.fast, seed=seed)
            img, meta = self.gen.generate(cfg, make_control(clean, mode))
            m = preservation_metrics(mask, img)
            if self.clip:
                m["clip_score"] = self.clip(img, cfg.prompt)
            m["seconds"] = meta["seconds"]
            attempts.append((img, m, cfg))
            log.append(f"attempt {k}: {mode} cond={cond:.2f} -> edge F1 {m['edge_f1']:.3f}, "
                       f"recall {m['edge_recall']:.3f}, CLIP {m.get('clip_score', 'n/a')}")

            score = m["edge_f1"] + 0.01 * m.get("clip_score", 0)
            if best is None or score > best[0]:
                best = (score, k - 1)

            structure_ok = m["edge_f1"] >= self.f1_target
            clip_ok = m.get("clip_score", self.clip_target) >= self.clip_target
            if structure_ok and clip_ok:
                log.append("targets met -> stop")
                break
            if not structure_ok:
                if cond < 1.3:
                    cond = round(cond + 0.3, 2)
                    log.append("structure too weak -> increase cond_scale")
                else:
                    mode = "lineart" if mode == "scribble" else "scribble"
                    log.append(f"cond_scale already high -> switch ControlNet to '{mode}'")
            else:  # structure fine, semantics weak: give the model more freedom
                cond = round(max(0.4, cond - 0.3), 2)
                log.append("structure fine but weak prompt match -> decrease cond_scale")
            seed += 1
        img, m, cfg = attempts[best[1]]
        log.append(f"selected attempt {best[1] + 1}")
        return AgentResult(image=img, config=cfg, metrics=m, log=log, attempts=attempts)
