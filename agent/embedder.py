"""Dependency-free text embeddings via feature hashing.

NOTE: written fresh for this repo. It is NOT a copy of rag-qa's
HashingEmbedder, which wasn't available. It exposes the same idea (hash tokens
into a fixed-size vector), so the original can be dropped in later if
``embed(texts) -> np.ndarray`` is kept.

How it works: text is lowercased and split into word tokens, common stopwords
are dropped, and each unigram and adjacent-word bigram is hashed into one of
``dim`` buckets with a +/-1 sign (signed hashing reduces collision bias).
Counts are log-scaled and each vector is L2-normalised, so a dot product
between two embeddings is their cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in into is it
    its of on or so that the their them then there these this to was what when where
    which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    # blake2b is stable across processes, unlike Python's salted hash().
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    def __init__(self, dim: int = 1024, bigrams: bool = True) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.bigrams = bigrams

    def features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        if self.bigrams:
            feats += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return feats

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for feat in self.features(text):
            idx, sign = _bucket(feat, self.dim)
            vec[idx] += sign
        # Sublinear term frequency: repeated words help, but with diminishing returns.
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed many texts into an (n, dim) array of unit (or zero) vectors."""
        rows = [self.embed_one(t) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim))
