"""Model + pipeline tests with tiny random-weight configs. CPU-only, ~1 minute, no downloads.

Run with `pytest` or `python tests/test_models.py`.
"""

import os
import sys
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdscratch.configs import SD15, SDXL  # noqa: E402
from sdscratch.loader import Components, build_module, convert_openclip  # noqa: E402
from sdscratch.models import (  # noqa: E402
    CLIP_VIT_BIGG,
    CLIP_VIT_L,
    SD15_UNET,
    SD15_VAE,
    SDXL_UNET,
    AutoencoderKL,
    CLIPTextConfig,
    CLIPTextModel,
    UNetConfig,
    UNetModel,
    VAEConfig,
    set_attention_backend,
)
from sdscratch.models.attention import attention  # noqa: E402
from sdscratch.pipeline import MemoryOptions, StableDiffusion  # noqa: E402
from sdscratch.samplers import SAMPLERS  # noqa: E402

# ----------------------------------------------------------------------------- tiny configs
TINY_TE = CLIPTextConfig(vocab_size=1000, max_len=77, dim=32, layers=2, heads=4, mlp_dim=64, pad_id=999)
TINY_TE_G = replace(TINY_TE, dim=48, heads=4, mlp_dim=96, act="gelu", proj_dim=40, pad_id=0)
TINY_VAE = VAEConfig(ch=32, ch_mult=(1, 2), num_res_blocks=1)
TINY_UNET = UNetConfig(model_channels=32, channel_mult=(1, 2), num_res_blocks=1, attention_resolutions=(1, 2),
                       transformer_depth=(1, 1), num_heads=4, context_dim=32)
TINY_UNET_XL = replace(TINY_UNET, attention_resolutions=(2,), transformer_depth=(1, 2), transformer_depth_middle=2,
                       num_head_channels=16, context_dim=32 + 48, use_linear_in_transformer=True,
                       adm_in_channels=40 + 6 * 256)
TINY_SD15 = replace(SD15, unet=TINY_UNET, vae=TINY_VAE, text_encoders=(TINY_TE,), default_size=64)
TINY_SDXL = replace(SDXL, unet=TINY_UNET_XL, vae=replace(TINY_VAE, scaling_factor=0.13025),
                    text_encoders=(TINY_TE, TINY_TE_G), default_size=64)


class FakeTokenizer:
    def __call__(self, prompts, pad_id=49407):
        out = []
        for p in prompts:
            ids = [998] + [(ord(c) % 900) + 1 for c in p][:75] + [999]   # 999 plays EOS (largest id)
            out.append(ids + [pad_id] * (77 - len(ids)))
        return out


def tiny_pipeline(spec, memory=None):
    torch.manual_seed(0)
    comps = Components(spec, UNetModel(spec.unet).eval(), AutoencoderKL(spec.vae).eval(),
                       [CLIPTextModel(c).eval() for c in spec.text_encoders])
    return StableDiffusion(comps, FakeTokenizer(), "cpu", torch.float32, memory)


# ------------------------------------------------------------------------------------ tests
def test_param_counts_match_official_models():
    """Exact parameter counts of the official checkpoints (as reported by diffusers/transformers)."""
    expected = [(UNetModel, SD15_UNET, 859_520_964), (UNetModel, SDXL_UNET, 2_567_463_684),
                (AutoencoderKL, SD15_VAE, 83_653_863), (CLIPTextModel, CLIP_VIT_L, 123_060_480),
                (CLIPTextModel, CLIP_VIT_BIGG, 694_659_840)]
    for cls, cfg, n in expected:
        with torch.device("meta"):
            m = cls(cfg)
        assert sum(p.numel() for p in m.parameters()) == n, (cls.__name__, cfg)


def test_attention_backends_agree():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 300, 64), torch.randn(2, 77, 64), torch.randn(2, 77, 64)
    outs = []
    for backend in ("naive", "sliced", "sdpa"):
        set_attention_backend(backend, slice_size=64)
        outs.append(attention(q, k, v, heads=4))
    set_attention_backend("sdpa")
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(outs[0], outs[2], atol=1e-5)


def test_unet_forward_shapes():
    torch.manual_seed(0)
    for cfg in (TINY_UNET, TINY_UNET_XL):
        unet = UNetModel(cfg).eval()
        x = torch.randn(2, 4, 16, 24)
        ctx = torch.randn(2, 77, cfg.context_dim)
        y = torch.randn(2, cfg.adm_in_channels) if cfg.adm_in_channels else None
        out = unet(x, torch.tensor([10.0, 500.0]), ctx, y)
        assert out.shape == x.shape


def test_vae_shapes_and_tiling():
    torch.manual_seed(0)
    vae = AutoencoderKL(TINY_VAE).eval()
    f = TINY_VAE.downsample_factor
    img = torch.randn(1, 3, 32, 48)
    z = vae.encode(img)
    assert z.shape == (1, 4, 32 // f, 48 // f)
    with torch.no_grad():
        full = vae.decode(torch.randn(1, 4, 20, 28))
        same = vae.decode_tiled(torch.randn(1, 4, 20, 28) * 0 + 1, tile=64)     # one tile -> plain decode
        assert same.shape == full.shape
        z = torch.randn(1, 4, 20, 28)
        tiled = vae.decode_tiled(z, tile=8, overlap=4)
        assert tiled.shape == (1, 3, 20 * f, 28 * f)
        assert torch.isfinite(tiled).all()
        assert torch.allclose(vae.decode_tiled(z, tile=64), vae.decode(z))


def test_clip_is_causal_and_pools_eos():
    torch.manual_seed(0)
    te = CLIPTextModel(TINY_TE_G).eval()
    a = torch.tensor([[998, 5, 6, 999, 0, 0] + [0] * 71])
    b = a.clone()
    b[0, 5] = 7                      # change a token *after* EOS
    with torch.no_grad():
        oa, ob = te(a), te(b)
    assert torch.allclose(oa.last_hidden_state[:, :5], ob.last_hidden_state[:, :5], atol=1e-6)
    assert torch.allclose(oa.pooled, ob.pooled, atol=1e-6)
    assert oa.pooled.shape == (1, TINY_TE_G.proj_dim)
    assert oa.penultimate.shape == (1, 77, TINY_TE_G.dim)


def test_openclip_conversion_roundtrip():
    """Write a CLIPTextModel in OpenCLIP layout, convert back, and check identical outputs."""
    torch.manual_seed(0)
    ref = CLIPTextModel(TINY_TE_G).eval()
    sd = ref.state_dict()
    oc = {"token_embedding.weight": sd["text_model.embeddings.token_embedding.weight"],
          "positional_embedding": sd["text_model.embeddings.position_embedding.weight"],
          "ln_final.weight": sd["text_model.final_layer_norm.weight"],
          "ln_final.bias": sd["text_model.final_layer_norm.bias"],
          "text_projection": sd["text_projection.weight"].t().contiguous(),
          "logit_scale": torch.tensor(4.6)}
    names = {"ln_1": "layer_norm1", "ln_2": "layer_norm2", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2",
             "attn.out_proj": "self_attn.out_proj"}
    for i in range(TINY_TE_G.layers):
        src, dst = f"text_model.encoder.layers.{i}.", f"transformer.resblocks.{i}."
        for kind in ("weight", "bias"):
            oc[f"{dst}attn.in_proj_{kind}"] = torch.cat(
                [sd[f"{src}self_attn.{p}_proj.{kind}"] for p in "qkv"], dim=0)
            for a, b in names.items():
                oc[f"{dst}{a}.{kind}"] = sd[f"{src}{b}.{kind}"]
    loaded = build_module(lambda: CLIPTextModel(TINY_TE_G), convert_openclip(oc), "te")
    tokens = torch.tensor(FakeTokenizer()(["a photo of a cat"], pad_id=0))
    with torch.no_grad():
        o1, o2 = ref(tokens), loaded(tokens)
    assert torch.allclose(o1.pooled, o2.pooled, atol=1e-6)
    assert torch.allclose(o1.penultimate, o2.penultimate, atol=1e-6)


def test_strict_loading_reports_mismatch():
    sd = UNetModel(TINY_UNET).state_dict()
    sd.pop("out.2.weight")
    with pytest.raises(RuntimeError, match="missing"):
        build_module(lambda: UNetModel(TINY_UNET), sd, "unet")


@pytest.mark.parametrize("spec", [TINY_SD15, TINY_SDXL], ids=["sd15", "sdxl"])
def test_pipeline_all_samplers(spec):
    pipe = tiny_pipeline(spec)
    for name in SAMPLERS:
        res = pipe("a red cube", steps=4, sampler=name, seed=1, progress=False)
        assert res.images[0].size == (64, 64), name
        assert torch.isfinite(res.latents).all(), name


@pytest.mark.parametrize("spec", [TINY_SD15, TINY_SDXL], ids=["sd15", "sdxl"])
def test_memory_options_do_not_change_the_result(spec):
    """Offloading / attention backend / CFG batching / tiling are pure memory-speed trade-offs."""
    pipe = tiny_pipeline(spec)
    ref = pipe("a red cube", negative_prompt="blurry", steps=3, seed=3, progress=False).latents
    variants = [dict(offload="model"), dict(offload="sequential"), dict(attention="naive"),
                dict(attention="sliced", attention_slice_size=7), dict(cfg_mode="sequential"),
                dict(channels_last=True)]
    for v in variants:
        pipe.set_memory_options(MemoryOptions(**v))
        out = pipe("a red cube", negative_prompt="blurry", steps=3, seed=3, progress=False).latents
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        assert err < 1e-4, (v, err)    # identical up to float32 round-off
    pipe.set_memory_options(MemoryOptions())


def test_seed_reproducibility_and_guidance_one():
    pipe = tiny_pipeline(TINY_SDXL)
    a = pipe("x", steps=2, seed=5, progress=False).latents
    b = pipe("x", steps=2, seed=5, progress=False).latents
    c = pipe("x", steps=2, seed=6, progress=False).latents
    assert torch.equal(a, b) and not torch.equal(a, c)
    pipe("x", steps=2, guidance=1.0, progress=False)   # no unconditional branch


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if name in ("test_pipeline_all_samplers", "test_memory_options_do_not_change_the_result"):
                for spec in (TINY_SD15, TINY_SDXL):
                    fn(spec)
            else:
                fn()
            print(f"PASS {name}")
