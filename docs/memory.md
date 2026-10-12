# Where the memory goes at 1024x1024 (and what each optimisation removes)

All numbers below are **analytic estimates** from parameter counts and tensor shapes (1 GiB = 2^30
bytes). Measured peaks for your GPU come from `scripts/benchmark_memory.py`.

Measurements on an NVIDIA A30 (README, `results/memory_*_1024.json`) match the estimates:
- With all SDXL fp16 weights resident (VAE in fp32), the text-encoder stage peaks at 6.73 GiB, against
  the 6.6 GiB estimated below.
- For SD 1.5 at 1024 px, naive attention fails trying to allocate exactly 16.00 GiB in fp32 and 8.00 GiB
  in fp16, the attention-matrix sizes in section 2.

## 1. Weights

| component | parameters | fp32 | fp16 |
|---|---:|---:|---:|
| SDXL UNet | 2.567 B | 9.56 GiB | 4.78 GiB |
| SD 1.5 UNet | 0.860 B | 3.20 GiB | 1.60 GiB |
| CLIP ViT-L/14 text encoder | 0.123 B | 0.46 GiB | 0.23 GiB |
| OpenCLIP ViT-bigG/14 text encoder (SDXL) | 0.695 B | 2.59 GiB | 1.29 GiB |
| VAE (both models) | 0.084 B | 0.31 GiB | 0.16 GiB |

SDXL in fp32 needs ~13 GiB for weights alone; fp16 (with the VAE kept in fp32, see below) needs
**6.6 GiB**. The parameter counts are asserted exactly in `tests/test_models.py`.

* **fp16 weights** halve everything. The SDXL VAE is the exception: its activations overflow fp16
  (NaNs -> black images), so it runs in fp32; tiling keeps that affordable.
* **Model offload** (`offload="model"`): only one component is on the GPU at a time
  (text encoders -> UNet -> VAE), so the weight term becomes max(component) = 4.78 GiB for SDXL.
* **Block-wise offload** (`offload="sequential"`): the UNet is streamed one top-level block at a
  time in every step; the largest SDXL block (a ResBlock + 10-layer transformer at 1280 channels) is
  ~0.4 B parameters, ~0.8 GiB in fp16. Costs a full PCIe copy of the UNet per network evaluation.

## 2. Attention matrices

Naive attention materialises `batch x heads x N x N` scores (N = tokens = latent pixels at that level):

| where | tokens N | heads | batch (CFG) | size per layer (fp16) |
|---|---:|---:|---:|---:|
| SDXL UNet level 1 @ 1024 px | 64x64 = 4,096 | 10 | 2 | 0.63 GiB |
| SDXL UNet level 2 @ 1024 px | 32x32 = 1,024 | 20 | 2 | 0.08 GiB |
| SD 1.5 UNet level 0 @ 512 px | 64x64 = 4,096 | 8 | 2 | 0.50 GiB |
| SD 1.5 UNet level 0 @ 1024 px | 128x128 = 16,384 | 8 | 2 | **8.0 GiB** |
| VAE mid-block @ 1024 px (fp32) | 128x128 = 16,384 | 1 | 1 | 1.0 GiB (x2 with softmax output) |

Attention cost is quadratic in the number of pixels, so it dominates as resolution grows.

* **Sliced attention** (`attention="sliced"`) processes 1,024 queries at a time: the score buffer
  shrinks by N/1024 (16x for SD 1.5 @ 1024) at a small speed cost.
* **SDPA** (`attention="sdpa"`, default) uses PyTorch's fused FlashAttention / memory-efficient
  kernels, which never materialise the matrix (O(N) memory) and are also the fastest.

## 3. Activations and the VAE decoder

* **CFG batching**: classifier-free guidance doubles the UNet batch. `cfg_mode="sequential"` runs the
  conditional and unconditional passes one after the other - half the activation memory, ~same FLOPs.
* **VAE decode** is often the true peak at 1024 px: the last decoder stage keeps several
  128 x 1024 x 1024 fp32 tensors alive (0.5 GiB each) plus the 16k-token mid-block attention.
  **Tiled decoding** (`vae_tiling=True`) decodes 64x64-latent tiles (512 px) with a 16-latent-pixel
  overlap and feather-blends them, so the decoder peak is that of a 512 px image regardless of output
  size. GroupNorm statistics differ slightly per tile; the linear cross-fade hides the seams.

## Presets (`sdscratch.pipeline.PRESETS`)

| preset | offload | VAE tiling | CFG | target GPU (estimate; verify with the benchmark) |
|---|---|---|---|---|
| `fast` | none | auto (on above 768 px) | batched | >= 12 GB GPUs |
| `balanced` | model | on | batched | ~8 GB GPUs |
| `low` | model | on | sequential | 6-8 GB GPUs |
| `minimal` | block-wise | on | sequential | 4 GB GPUs (slow) |

Emulate a smaller card on a big one with `python scripts/benchmark_memory.py --vram-limit-gb 8`.
