"""Tools available to the agent: search, page fetching, and ranking.

Search and fetch are stubs for now; they return deterministic placeholder
content so the graph can be exercised offline. ``cosine_similarity`` is real
and will rank evidence once embeddings are wired in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

WEB_CACHE_DIR = Path(__file__).resolve().parent.parent / "web_cache"


def cache_path(url: str) -> Path:
    """Stable on-disk location for a fetched URL."""
    return WEB_CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.txt"


def web_search(query: str, k: int = 3) -> list[dict]:
    """STUB: return ``k`` fake results shaped like a real search API."""
    return [
        {"url": f"stub://search/{i}?q={query}", "title": f"Result {i} for {query!r}"}
        for i in range(k)
    ]


def fetch_page(url: str) -> str:
    """STUB: return cached content if present, else placeholder text (not cached)."""
    path = cache_path(url)
    if path.exists():
        return path.read_text()
    return f"[stub content for {url}]"


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector (d,) against a matrix of docs (n, d)."""
    q = np.asarray(query_vec, dtype=float)
    d = np.atleast_2d(np.asarray(doc_vecs, dtype=float))
    denom = np.linalg.norm(d, axis=1) * np.linalg.norm(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = d @ q / denom
    return np.nan_to_num(sims, nan=0.0)
