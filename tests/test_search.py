"""Tests for the web_cache/ search tool and the embedder behind it."""

from __future__ import annotations

import numpy as np
import pytest

from agent import tools
from agent.embedder import HashingEmbedder, tokenize
from agent.graph import build_graph
from agent.nodes import SEARCH
from agent.state import initial_state
from agent.tools import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS, chunk_page, web_search
from agent.trace import node_path

COSINE_QUESTIONS = [
    "What is cosine similarity?",
    "How does cosine similarity compare two vectors?",
    "measure the angle between two vectors",
]


# --------------------------------------------------------------------------- #
# The requirement: cosine questions find the cosine page (real web_cache/)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("question", COSINE_QUESTIONS)
def test_cosine_question_returns_cosine_page_first(question):
    hits = web_search(question, k=3)
    assert hits[0]["url"] == "cosine_similarity.md"
    assert hits[0]["title"] == "Cosine similarity"
    assert "cosine" in hits[0]["content"].lower()


def test_graph_collects_cosine_page_as_evidence():
    """End to end: the unchanged graph puts the cosine page into evidence."""
    result = build_graph().invoke(initial_state("What is cosine similarity?"))
    assert node_path(result)[:2] == ["plan", SEARCH]
    top = max(result["evidence"], key=lambda e: e["score"])
    assert top["source"] == "cosine_similarity.md"
    assert top["score"] > 0


@pytest.mark.parametrize(
    "question, page",
    [
        ("how does HNSW find nearest neighbors", "hnsw_index.md"),
        ("run a language model locally with ollama", "ollama.md"),
        ("WAPE forecast error compared with MAPE", "wape_metric.md"),
        ("how do reducers merge langgraph state updates", "langgraph_state.md"),
        ("chunking documents with overlap", "chunking.md"),
    ],
)
def test_other_topics_route_to_their_page(question, page):
    assert web_search(question, k=1)[0]["url"] == page


# --------------------------------------------------------------------------- #
# web_search contract
# --------------------------------------------------------------------------- #


def test_hit_shape_and_ordering():
    hits = web_search("vectors", k=5)
    assert len(hits) == 5
    for h in hits:
        assert set(h) == {"url", "title", "content", "score"}
        assert h["url"].endswith(".md") and "/" not in h["url"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_returns_chunks_not_whole_pages():
    hit = web_search("What is cosine similarity?", k=1)[0]
    page = (tools.WEB_CACHE_DIR / hit["url"]).read_text()
    assert len(hit["content"]) < len(page)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_nothing(k):
    assert web_search("cosine", k=k) == []


def test_k_larger_than_corpus_returns_every_chunk():
    chunks, _ = tools.load_index()
    assert len(web_search("anything", k=10_000)) == len(chunks)


# --------------------------------------------------------------------------- #
# Isolated corpus (tmp dir)
# --------------------------------------------------------------------------- #


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WEB_CACHE_DIR", tmp_path)
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "notes.txt").write_text("cosine cosine cosine")  # not .md: ignored
    (tmp_path / "apples.md").write_text("# Apples\n\nApples are red fruit.\n\nApple pie is baked.")
    (tmp_path / "rockets.md").write_text("# Rockets\n\nRockets burn fuel to reach orbit.")
    return tmp_path


def test_only_markdown_files_are_indexed(corpus):
    sources = {h["url"] for h in web_search("cosine apples rockets", k=100)}
    assert sources == {"apples.md", "rockets.md"}


def test_index_picks_up_new_and_edited_files(corpus):
    assert web_search("submarine", k=1)[0]["url"] != "subs.md"
    (corpus / "subs.md").write_text("# Submarines\n\nA submarine travels underwater.")
    assert web_search("submarine", k=1)[0]["url"] == "subs.md"
    (corpus / "subs.md").write_text("# Boats\n\nBoats float on the surface of water.")
    assert web_search("submarine underwater", k=1)[0]["url"] != "subs.md"


def test_empty_or_missing_cache_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WEB_CACHE_DIR", tmp_path / "missing")
    assert web_search("anything") == []
    monkeypatch.setattr(tools, "WEB_CACHE_DIR", tmp_path)
    assert web_search("anything") == []


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_chunk_page_paragraphs_with_title_prefix():
    chunks = chunk_page("# My Topic\n\nFirst para.\n\nSecond\npara.", "my_topic.md")
    assert [c.text for c in chunks] == ["My Topic: First para.", "My Topic: Second para."]
    assert [c.index for c in chunks] == [0, 1]
    assert {c.source for c in chunks} == {"my_topic.md"}


def test_chunk_page_title_falls_back_to_file_name():
    assert chunk_page("Just text.", "some_page.md")[0].title == "some page"


def test_long_paragraph_split_with_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_page("# T\n\n" + " ".join(words), "t.md")
    bodies = [c.text.removeprefix("T: ").split() for c in chunks]
    assert all(len(b) <= CHUNK_MAX_WORDS for b in bodies)
    assert bodies[0][-CHUNK_OVERLAP_WORDS:] == bodies[1][:CHUNK_OVERLAP_WORDS]
    assert bodies[-1][-1] == "w249"  # nothing dropped at the end


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #


def test_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a, b = e.embed_one("cosine similarity of vectors"), e.embed_one("cosine similarity of vectors")
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_embedder_empty_text_is_zero_vector():
    assert not HashingEmbedder(dim=64).embed_one("the of and").any()  # stopwords only


def test_embedder_related_text_scores_higher():
    e = HashingEmbedder()
    q = e.embed_one("vector similarity")
    assert q @ e.embed_one("similarity between vectors and vector angles") > q @ e.embed_one("baking apple pie")


def test_embed_batch_shape():
    e = HashingEmbedder(dim=32)
    assert e.embed(["a b", "c d", "e"]).shape == (3, 32)
    assert e.embed([]).shape == (0, 32)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What IS the WAPE metric?") == ["wape", "metric"]
