**SD15 1024x1024, 20 DPM-Solver++(2M) steps, CFG on, GPU: NVIDIA A30**

| Configuration | Peak VRAM (GiB) | UNet s/step | Total s | Status |
|---|---:|---:|---:|---|
| fp32, naive attention (baseline) | - | - | - | OOM |
| + fp16 weights | - | - | - | OOM |
| + SDPA attention | 4.48 | 0.20 | 4.3 | ok |
| + tiled VAE decode | 2.57 | 0.20 | 4.5 | ok |
| + model offload | 2.15 | 0.22 | 5.2 | ok |
| + sequential CFG | 1.89 | 0.25 | 5.7 | ok |
| + block-wise UNet offload | 0.72 | 0.91 | 18.9 | ok |
| fp16, sliced attention (no SDPA) | 3.31 | 0.48 | 10.1 | ok |
