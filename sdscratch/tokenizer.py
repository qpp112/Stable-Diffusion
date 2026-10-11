"""CLIP byte-level BPE tokenizer (Radford et al., 2021), written from scratch.

Reads the standard ``vocab.json`` / ``merges.txt`` files shipped with every Stable
Diffusion repo on the Hugging Face Hub. Pure Python - no torch needed.

Pipeline:  clean text -> lowercase -> regex pre-tokenisation -> UTF-8 bytes mapped to
printable unicode -> greedy BPE merges (lowest rank first) -> ids, wrapped as
``<|startoftext|> ... <|endoftext|>`` and padded to 77 tokens.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from typing import Iterable, List, Sequence, Union

try:  # the exact pattern needs unicode classes (\p{L}, \p{N}) from the `regex` package
    import regex as _regex

    _PAT = _regex.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        _regex.IGNORECASE)
except ImportError:  # close stdlib approximation: letters = [^\W\d_], numbers = \d
    _PAT = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+""",
        re.IGNORECASE)

try:
    import ftfy as _ftfy
except ImportError:
    _ftfy = None


@lru_cache()
def bytes_to_unicode() -> dict:
    """Reversible map from the 256 byte values to printable unicode characters (GPT-2 trick)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _pairs(word: Sequence[str]) -> set:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def _clean(text: str) -> str:
    if _ftfy is not None:
        text = _ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class CLIPTokenizer:
    BOS = 49406  # <|startoftext|>
    EOS = 49407  # <|endoftext|>

    def __init__(self, vocab_path: str, merges_path: str, max_len: int = 77):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder: dict = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")[1:49152 - 256 - 2 + 1]   # skip "#version" header
        self.bpe_ranks = {tuple(line.split()): i for i, line in enumerate(lines)}
        self.byte_encoder = bytes_to_unicode()
        self.max_len = max_len
        self._cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

    def bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)   # mark end of word
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            first, second = best
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        out = " ".join(word)
        self._cache[token] = out
        return out

    def encode(self, text: str) -> List[int]:
        """Text -> BPE ids, without BOS/EOS or padding."""
        ids: List[int] = []
        for tok in _PAT.findall(_clean(text).lower()):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self.bpe(tok).split(" "))
        return ids

    def __call__(self, prompts: Union[str, Iterable[str]], pad_id: int = EOS) -> List[List[int]]:
        """Batch of prompts -> fixed-length id lists (truncated to 75 tokens + BOS/EOS)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        out = []
        for p in prompts:
            ids = [self.BOS] + self.encode(p)[: self.max_len - 2] + [self.EOS]
            out.append(ids + [pad_id] * (self.max_len - len(ids)))
        return out
