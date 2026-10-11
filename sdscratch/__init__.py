"""Stable Diffusion (SD 1.5 / SDXL) from scratch in PyTorch.

The sampler code in ``sdscratch.samplers`` and the toy experiments in ``sdscratch.toy``
are NumPy-compatible and do not import torch; everything else does.
"""

__version__ = "0.1.0"


def __getattr__(name):  # lazy: keep `import sdscratch.samplers` torch-free
    if name in ("StableDiffusion", "MemoryOptions", "PRESETS"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
