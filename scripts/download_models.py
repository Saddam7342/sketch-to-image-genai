"""Pre-download exactly the model files the pipeline needs (fp16 weights only).

Useful on slow or unstable connections: downloads are resumable, ordered by priority,
and skip everything unused (fp32 duplicates, .ckpt files, the safety checker if disabled).

  python scripts/download_models.py                 # core set for the CPU study
  python scripts/download_models.py --all           # + soft edge / canny ControlNets
Set HF_HOME to choose the cache location (e.g. a drive with free space).
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet transfer stalls on some networks; plain HTTP is robust

from huggingface_hub import snapshot_download  # noqa: E402

SD15 = ("stable-diffusion-v1-5/stable-diffusion-v1-5", [
    "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
])
SAFETY = ("stable-diffusion-v1-5/stable-diffusion-v1-5", ["safety_checker/config.json", "safety_checker/model.fp16.safetensors"])
CN = ["config.json", "diffusion_pytorch_model.fp16.safetensors"]

CORE = [
    SD15,
    ("lllyasviel/control_v11p_sd15_scribble", CN),
    ("latent-consistency/lcm-lora-sdv1-5", ["*.json", "*.safetensors"]),
    ("lllyasviel/control_v11p_sd15_lineart", CN),
    ("openai/clip-vit-base-patch32", ["*.json", "*.txt", "pytorch_model.bin"]),  # repo has no safetensors
]
EXTRA = [
    ("lllyasviel/control_v11p_sd15_softedge", CN),
    ("lllyasviel/control_v11p_sd15_canny", CN),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also soft edge + canny ControlNets")
    ap.add_argument("--safety-checker", action="store_true", help="also the SD safety checker (~600 MB)")
    a = ap.parse_args()
    todo = CORE + (EXTRA if a.all else []) + ([SAFETY] if a.safety_checker else [])
    for repo, patterns in todo:
        for attempt in range(1, 6):
            try:
                t = time.time()
                path = snapshot_download(repo, allow_patterns=patterns)
                print(f"OK   {repo}  ({time.time() - t:.0f}s)  -> {path}", flush=True)
                break
            except Exception as e:  # network hiccup: retry, the download resumes
                print(f"retry {attempt} {repo}: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * attempt)
        else:
            print(f"FAIL {repo}", flush=True)


if __name__ == "__main__":
    main()
