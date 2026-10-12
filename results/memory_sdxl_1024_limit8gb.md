**SDXL 1024x1024, 20 DPM-Solver++(2M) steps, CFG on, GPU: NVIDIA A30, allocator capped at 8 GiB**

| Configuration | Peak VRAM (GiB) | UNet s/step | Total s | Status |
|---|---:|---:|---:|---|
| + fp16 weights | - | - | - | OOM |
| + SDPA attention | - | - | - | OOM |
| + tiled VAE decode | 7.62 | 0.22 | 5.2 | ok |
| + model offload | 5.20 | 0.30 | 7.9 | ok |
| + sequential CFG | 5.04 | 0.36 | 8.9 | ok |
| + block-wise UNet offload | 1.54 | 2.11 | 43.9 | ok |
| fp16, sliced attention (no SDPA) | 7.62 | 0.34 | 7.6 | ok |
