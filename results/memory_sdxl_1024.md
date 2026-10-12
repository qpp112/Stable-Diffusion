**SDXL 1024x1024, 20 DPM-Solver++(2M) steps, CFG on, GPU: NVIDIA A30**

| Configuration | Peak VRAM (GiB) | UNet s/step | Total s | Status |
|---|---:|---:|---:|---|
| fp32, naive attention (baseline) | 16.58 | 1.50 | 30.4 | ok |
| + fp16 weights | 10.23 | 0.31 | 6.7 | ok |
| + SDPA attention | 10.23 | 0.21 | 4.7 | ok |
| + tiled VAE decode | 7.62 | 0.21 | 5.2 | ok |
| + model offload | 5.20 | 0.30 | 7.9 | ok |
| + sequential CFG | 5.04 | 0.34 | 8.6 | ok |
| + block-wise UNet offload | 1.54 | 2.20 | 45.7 | ok |
| fp16, sliced attention (no SDPA) | 7.62 | 0.33 | 7.5 | ok |
