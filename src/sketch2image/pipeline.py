"""Sketch-conditioned image generation with Stable Diffusion + ControlNet (diffusers).

How the sketch steers generation
--------------------------------
Stable Diffusion denoises a 64x64x4 latent (for 512 px images) over T steps; at
each step the U-Net predicts the noise, conditioned on the CLIP text embedding.
ControlNet is a trainable copy of the U-Net *encoder* that receives the control
image (our processed sketch). Its outputs pass through zero-initialised 1x1
convolutions and are *added* to the skip connections and middle block of the
frozen U-Net. The sketch therefore injects spatial structure at every
resolution, while the text prompt decides appearance (materials, light, style).

Knobs exposed here:
* cond_scale      – multiplier on the ControlNet residuals (sketch adherence)
* control_start/end – fraction of the denoising trajectory during which ControlNet
                    is active (early steps fix layout, late steps fix detail)
* guidance_scale  – classifier-free guidance weight for the text prompt
* steps, scheduler, seed
* lcm             – LCM-LoRA distillation: 4–8 steps, makes CPU inference viable
"""
from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field

import torch
from PIL import Image

from .prompts import DEFAULT_NEGATIVE

# --------------------------------------------------------------------------- registry
BASES = {
    "sd15": {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "family": "sd15", "res": 512},
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "res": 1024,
        "vae": "madebyollin/sdxl-vae-fp16-fix",
    },
    # tiny random-weight models, only for fast CPU smoke tests of the code paths
    "tiny": {"repo": "hf-internal-testing/tiny-stable-diffusion-torch", "family": "sd15", "res": 64},
}

# control mode -> (family -> ControlNet repo). "*_aux" modes reuse the base model.
CONTROLNETS = {
    "scribble": {"sd15": "lllyasviel/control_v11p_sd15_scribble", "sdxl": "xinsir/controlnet-scribble-sdxl-1.0"},
    "lineart": {"sd15": "lllyasviel/control_v11p_sd15_lineart"},
    "softedge": {"sd15": "lllyasviel/control_v11p_sd15_softedge"},
    "canny": {"sd15": "lllyasviel/control_v11p_sd15_canny", "sdxl": "xinsir/controlnet-canny-sdxl-1.0"},
}
LCM_LORA = {"sd15": "latent-consistency/lcm-lora-sdv1-5", "sdxl": "latent-consistency/lcm-lora-sdxl"}


def controlnet_key(mode: str) -> str | None:
    """Map a preprocessing mode to the ControlNet it feeds ('none' = no ControlNet)."""
    if mode == "none":
        return None
    return mode.removesuffix("_aux")


@dataclass
class GenConfig:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    base: str = "sd15"
    control: str = "scribble"          # scribble | lineart | softedge | canny | *_aux | none
    steps: int = 30
    guidance_scale: float = 7.5
    cond_scale: float = 1.0
    control_start: float = 0.0
    control_end: float = 1.0
    seed: int = 1
    scheduler: str = "unipc"           # unipc | ddim | euler_a | dpmpp
    lcm: bool = False
    resolution: int | None = None      # default: model's native resolution
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _make_scheduler(name: str, config):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        LCMScheduler,
        UniPCMultistepScheduler,
    )

    table = {
        "unipc": UniPCMultistepScheduler,
        "ddim": DDIMScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpmpp": DPMSolverMultistepScheduler,
        "lcm": LCMScheduler,
    }
    if name not in table:
        raise ValueError(f"unknown scheduler {name!r}")
    return table[name].from_config(config)


class SketchGenerator:
    """Loads models lazily and caches them, so a sweep re-uses one base model.

    Only one base model family is kept in memory at a time (SD1.5 and SDXL do not
    both fit on a 16 GB T4 alongside their ControlNets).
    """

    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None, cpu_offload: bool = False,
                 safety_checker: bool = True):
        d, t = pick_device()
        self.device = device or d
        self.dtype = dtype or t
        self.cpu_offload = cpu_offload
        self.safety_checker = safety_checker  # SD1.5 NSFW filter; disable only to save ~600 MB download
        self._base_name: str | None = None
        self._base_pipe = None
        self._controlnets: dict[str, object] = {}
        self._pipes: dict[str | None, object] = {}
        self._lcm_loaded = False

    # ---------------------------------------------------------------- loading
    def _load_base(self, base: str):
        if self._base_name == base:
            return
        self.unload()
        from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline

        spec = BASES[base]
        kw = {"torch_dtype": self.dtype}
        if spec["family"] == "sdxl":
            if "vae" in spec:
                kw["vae"] = AutoencoderKL.from_pretrained(spec["vae"], torch_dtype=self.dtype)
            pipe = StableDiffusionXLPipeline.from_pretrained(spec["repo"], **kw)
        else:
            extra = {} if self.safety_checker else {"safety_checker": None, "requires_safety_checker": False}
            pipe = _from_pretrained_fp16_first(StableDiffusionPipeline, spec["repo"], self.dtype, **extra)
        self._base_pipe = self._place(pipe)
        self._base_name = base

    def _place(self, pipe):
        if self.cpu_offload and self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.device == "cpu":
            pipe.enable_attention_slicing()
        return pipe

    def _controlnet(self, key: str):
        if key in self._controlnets:
            return self._controlnets[key]
        from diffusers import ControlNetModel

        family = BASES[self._base_name]["family"]
        if self._base_name == "tiny":  # random ControlNet built from the tiny U-Net
            # the tiny test VAE downsamples by 2 (not 8), so one stride-2 stage suffices
            cn = ControlNetModel.from_unet(self._base_pipe.unet, conditioning_embedding_out_channels=(16, 32))
        else:
            repo = CONTROLNETS.get(key, {}).get(family)
            if repo is None:
                raise ValueError(f"no ControlNet for mode {key!r} on base family {family!r}")
            cn = _from_pretrained_fp16_first(ControlNetModel, repo, self.dtype)
        cn = cn.to(self.device)  # dtype already set by from_pretrained
        self._controlnets[key] = cn
        return cn

    def get_pipe(self, base: str, mode: str):
        self._load_base(base)
        key = controlnet_key(mode)
        if key in self._pipes:
            return self._pipes[key]
        if key is None:
            pipe = self._base_pipe
        else:
            from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline

            cls = (
                StableDiffusionXLControlNetPipeline
                if BASES[base]["family"] == "sdxl"
                else StableDiffusionControlNetPipeline
            )
            # from_pipe shares the already-loaded VAE / U-Net / text encoders (no extra memory)
            pipe = cls.from_pipe(self._base_pipe, controlnet=self._controlnet(key))
            pipe.set_progress_bar_config(disable=True)
        self._pipes[key] = pipe
        return pipe

    def _set_lcm(self, enabled: bool):
        family = BASES[self._base_name]["family"]
        if enabled and not self._lcm_loaded:
            self._base_pipe.load_lora_weights(LCM_LORA[family], weight_name="pytorch_lora_weights.safetensors",
                                              adapter_name="lcm")
            self._lcm_loaded = True
        if self._lcm_loaded:
            # the LoRA lives in the shared U-Net, so toggling it affects all derived pipes
            (self._base_pipe.enable_lora if enabled else self._base_pipe.disable_lora)()

    def unload(self):
        self._pipes.clear()
        self._controlnets.clear()
        self._base_pipe = None
        self._base_name = None
        self._lcm_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(self, cfg: GenConfig, control_image: Image.Image | None) -> tuple[Image.Image, dict]:
        pipe = self.get_pipe(cfg.base, cfg.control)
        if self._base_name != "tiny":
            self._set_lcm(cfg.lcm)
        pipe.scheduler = _make_scheduler("lcm" if cfg.lcm else cfg.scheduler, pipe.scheduler.config)

        res = cfg.resolution or BASES[cfg.base]["res"]
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        kw = dict(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt or None,
            num_inference_steps=int(cfg.steps),
            guidance_scale=float(cfg.guidance_scale),
            generator=gen,
            height=res,
            width=res,
        )
        if controlnet_key(cfg.control) is not None:
            if control_image is None:
                raise ValueError("this control mode needs a control image")
            kw.update(
                image=control_image.convert("RGB").resize((res, res), Image.BILINEAR),
                controlnet_conditioning_scale=float(cfg.cond_scale),
                control_guidance_start=float(cfg.control_start),
                control_guidance_end=float(cfg.control_end),
            )
        t0 = time.perf_counter()
        out = pipe(**kw)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nsfw = getattr(out, "nsfw_content_detected", None)
        meta = {
            "seconds": round(dt, 3),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "nsfw_flag": bool(nsfw[0]) if nsfw else False,
            "resolution": res,
        }
        return out.images[0], meta


def _from_pretrained_fp16_first(cls, repo: str, dtype, **kw):
    """Download the smaller fp16 weight files when the repo has them (half the download);
    they are cast to `dtype` on load, so fp32 CPU inference works from fp16 files too."""
    try:
        return cls.from_pretrained(repo, torch_dtype=dtype, variant="fp16", **kw)
    except (OSError, ValueError):
        return cls.from_pretrained(repo, torch_dtype=dtype, **kw)
