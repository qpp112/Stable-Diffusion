"""Tokenizer tests: BPE mechanics on a synthetic vocabulary (offline), plus an optional exact
parity check against Hugging Face's CLIPTokenizer when `transformers` and the Hub are available."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdscratch.tokenizer import CLIPTokenizer, bytes_to_unicode  # noqa: E402


def make_tokenizer(tmp_dir):
    merges = ["#version: 0.2", "h e", "l l", "he ll", "o</w>", "hell o</w>", "c a", "ca t</w>"]
    vocab = {"hello</w>": 10, "cat</w>": 11, "!</w>": 12, "a</w>": 13, "<|startoftext|>": 49406,
             "<|endoftext|>": 49407}
    for ch in bytes_to_unicode().values():
        vocab.setdefault(ch, len(vocab) + 100)
        vocab.setdefault(ch + "</w>", len(vocab) + 100)
    vp, mp = os.path.join(tmp_dir, "vocab.json"), os.path.join(tmp_dir, "merges.txt")
    with open(vp, "w") as f:
        json.dump(vocab, f)
    with open(mp, "w") as f:
        f.write("\n".join(merges))
    return CLIPTokenizer(vp, mp), vocab


def test_bpe_merges_and_framing(tmp_path=None):
    import tempfile

    d = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    tok, vocab = make_tokenizer(d)
    assert tok.encode("Hello  cat!") == [10, 11, 12]          # lowercase, whitespace, punctuation split
    assert tok.encode("&amp;") == [vocab["&</w>"]]            # html unescape
    ids = tok(["hello"], pad_id=0)[0]
    assert len(ids) == 77 and ids[:3] == [49406, 10, 49407] and set(ids[3:]) == {0}
    long_ids = tok(["cat " * 200])[0]
    assert len(long_ids) == 77 and long_ids[0] == 49406 and long_ids[-1] == 49407   # truncated


def test_byte_map_is_bijective():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256


def test_parity_with_transformers():
    try:
        from huggingface_hub import hf_hub_download
        from transformers import CLIPTokenizer as HFTokenizer
        repo = "stabilityai/stable-diffusion-xl-base-1.0"
        vocab = hf_hub_download(repo, "tokenizer/vocab.json")
        merges = hf_hub_download(repo, "tokenizer/merges.txt")
        hf = HFTokenizer.from_pretrained(repo, subfolder="tokenizer")
    except Exception as e:  # offline / not installed
        print(f"skip parity test: {type(e).__name__}")
        return
    ours = CLIPTokenizer(vocab, merges)
    prompts = ["a photo of an astronaut riding a horse on mars",
               "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
               "pixel-art, 8k, (masterpiece), 1920s café, naïve résumé!!", ""]
    for p in prompts:
        ref = hf(p, padding="max_length", max_length=77, truncation=True).input_ids
        assert ours([p])[0] == ref, p


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
