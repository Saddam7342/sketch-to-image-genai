# References

## Papers

1. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. arXiv:2112.10752
2. L. Zhang, A. Rao, M. Agrawala. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV 2023. arXiv:2302.05543
3. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239
4. J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop. arXiv:2207.12598
5. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502
6. W. Zhao, L. Bai, Y. Rao, J. Zhou, J. Lu. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023. arXiv:2302.04867
7. D. Podell et al. *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* ICLR 2024. arXiv:2307.01952
8. S. Luo et al. *LCM-LoRA: A Universal Stable-Diffusion Acceleration Module.* arXiv:2311.05556, 2023
9. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685
10. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021. arXiv:2103.00020
11. J. Hessel et al. *CLIPScore: A Reference-free Evaluation Metric for Image Captioning.* EMNLP 2021. arXiv:2104.08718
12. S. Xie, Z. Tu. *Holistically-Nested Edge Detection.* ICCV 2015.
13. Z. Su et al. *Pixel Difference Networks for Efficient Edge Detection* (PiDiNet). ICCV 2021.
14. J. Canny. *A Computational Approach to Edge Detection.* IEEE TPAMI, 1986.

## Pretrained models (Hugging Face)

| Use | Model | License |
|---|---|---|
| Base model (main) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | CreativeML OpenRAIL-M |
| ControlNet scribble | [lllyasviel/control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) | OpenRAIL |
| ControlNet lineart | [lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) | OpenRAIL |
| ControlNet soft edge | [lllyasviel/control_v11p_sd15_softedge](https://huggingface.co/lllyasviel/control_v11p_sd15_softedge) | OpenRAIL |
| ControlNet canny | [lllyasviel/control_v11p_sd15_canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) | OpenRAIL |
| SDXL base | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | CreativeML OpenRAIL++-M |
| SDXL ControlNet scribble | [xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) | Apache-2.0 |
| SDXL fp16 VAE fix | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | MIT |
| Few-step sampling | [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | OpenRAIL++ |
| CLIP score | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | MIT |
| Learned annotators (PiDiNet, Lineart) | [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) | see repo |
| CPU smoke test only | [hf-internal-testing/tiny-stable-diffusion-torch](https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-torch) | test weights |

## Software

- PyTorch: <https://pytorch.org>
- 🤗 Diffusers: <https://github.com/huggingface/diffusers> (pipelines, ControlNet, schedulers, LoRA loading)
- 🤗 Transformers: <https://github.com/huggingface/transformers> (CLIP)
- 🤗 Accelerate, PEFT: <https://github.com/huggingface/accelerate>, <https://github.com/huggingface/peft>
- ControlNet v1.1 reference implementation: <https://github.com/lllyasviel/ControlNet-v1-1-nightly>
- controlnet_aux (annotators): <https://github.com/huggingface/controlnet_aux>
- OpenCV, scikit-image, NumPy, Pillow, pandas, matplotlib, PyYAML

## Data

- Input sketches: created for this project (see [sketches/README.md](sketches/README.md)); no external datasets.

## AI assistance

Parts of the code and documentation were developed with the help of an AI coding assistant (Claude, Anthropic).
All design decisions, experiments and conclusions were reviewed, run and verified by the author.
