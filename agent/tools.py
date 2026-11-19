"""Tools available to the agent: local search, page fetching, and ranking.

``web_search`` searches the markdown pages in ``web_cache/``: each page is
split into chunks, embedded with ``HashingEmbedder``, and ranked against the
query by cosine similarity. The index is rebuilt automatically when files in
``web_cache/`` are added, removed or modified.
"""

from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.embedder import HashingEmbedder

WEB_CACHE_DIR = Path(__file__).resolve().parent.parent / "web_cache"

CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

_embedder = HashingEmbedder()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    source: str  # file name within web_cache/, e.g. "ollama.md"
    title: str
    index: int  # position of the chunk within its page
    text: str


def page_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_page(markdown: str, source: str) -> list[Chunk]:
    """Split a page into paragraph chunks.

    Paragraphs (blank-line separated, headings removed) become chunks. A
    paragraph longer than CHUNK_MAX_WORDS is split into word windows that
    overlap by CHUNK_OVERLAP_WORDS. Each chunk's text is prefixed with the
    page title so it keeps its topic when read or embedded on its own.
    """
    title = page_title(markdown, fallback=Path(source).stem.replace("_", " "))
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", markdown)
        if p.strip() and not p.lstrip().startswith("#")
    ]
    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    pieces: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= CHUNK_MAX_WORDS:
            pieces.append(para)
            continue
        for start in range(0, len(words), step):
            pieces.append(" ".join(words[start : start + CHUNK_MAX_WORDS]))
            if start + CHUNK_MAX_WORDS >= len(words):
                break
    return [Chunk(source, title, i, f"{title}: {text}") for i, text in enumerate(pieces)]


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def _corpus_signature(directory: Path) -> tuple[tuple[str, int, int], ...]:
    """Identifies the current set of pages, so edits invalidate the cache."""
    if not directory.is_dir():
        return ()
    return tuple(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(directory.glob("*.md"))
    )


@functools.lru_cache(maxsize=4)
def _build_index(directory: Path, signature: tuple) -> tuple[list[Chunk], np.ndarray]:
    chunks: list[Chunk] = []
    for name, _, _ in signature:
        chunks += chunk_page((directory / name).read_text(encoding="utf-8"), name)
    return chunks, _embedder.embed(c.text for c in chunks)


def load_index(directory: Path | None = None) -> tuple[list[Chunk], np.ndarray]:
    directory = WEB_CACHE_DIR if directory is None else directory
    return _build_index(directory, _corpus_signature(directory))


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def web_search(query: str, k: int = 3) -> list[dict]:
    """Return the top-``k`` chunks from web_cache/*.md for ``query``.

    Each hit is ``{"url", "title", "content", "score"}``, where ``url`` is the
    page's file name and ``content`` is the matching chunk. Results are sorted
    by descending cosine similarity, with ties broken by file name and chunk
    position. Like a vector store, it always returns up to ``k`` chunks; judging
    relevance is left to the reflect node.
    """
    chunks, matrix = load_index()
    if k <= 0 or not chunks:
        return []
    scores = cosine_similarity(_embedder.embed_one(query), matrix)
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].source, chunks[i].index))
    return [
        {
            "url": chunks[i].source,
            "title": chunks[i].title,
            "content": chunks[i].text,
            "score": float(scores[i]),
        }
        for i in order[:k]
    ]


def cache_path(url: str) -> Path:
    """Stable on-disk location for a fetched URL."""
    return WEB_CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.txt"


def fetch_page(url: str) -> str:
    """Return a full page: a web_cache/ file name, or a cached URL. Else a placeholder."""
    for path in (WEB_CACHE_DIR / url, cache_path(url)):
        if path.parent == WEB_CACHE_DIR and path.is_file():
            return path.read_text(encoding="utf-8")
    return f"[no cached content for {url}]"


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector (d,) against a matrix of docs (n, d)."""
    q = np.asarray(query_vec, dtype=float)
    d = np.atleast_2d(np.asarray(doc_vecs, dtype=float))
    denom = np.linalg.norm(d, axis=1) * np.linalg.norm(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = d @ q / denom
    return np.nan_to_num(sims, nan=0.0)
