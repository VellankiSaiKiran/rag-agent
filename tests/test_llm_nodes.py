"""Tests for the LLM-backed nodes, driven by a scripted fake model."""

from __future__ import annotations

import json

import pytest

from agent.graph import build_graph
from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, get_llm
from agent.nodes import (
    ANSWER,
    MAX_FOLLOW_UPS,
    MAX_PLAN_QUERIES,
    PLAN,
    REFLECT,
    SEARCH,
    MAX_EVIDENCE_IN_PROMPT,
    REFLECT_SCHEMA,
    append_sources,
    cited_numbers,
    make_llm_nodes,
    parse_json_object,
    rank_evidence,
)
from agent.state import initial_state
from agent.trace import node_path
from tests.fakes import FakeLLM


def _evidence(query: str, source: str = "stub://x", content: str = "c", score: float = 0.0) -> dict:
    return {"query": query, "source": source, "content": content, "score": score}


# --------------------------------------------------------------------------- #
# Model config
# --------------------------------------------------------------------------- #


def test_default_llm_config(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = get_llm()
    assert (llm.model, llm.base_url) == (DEFAULT_MODEL, DEFAULT_BASE_URL)
    assert (DEFAULT_MODEL, DEFAULT_BASE_URL) == ("llama3.2", "http://localhost:11434")


def test_llm_config_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    llm = get_llm()
    assert (llm.model, llm.base_url) == ("qwen2.5", "http://gpu-box:11434")


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('Sure! Here you go: {"a": 1} hope that helps', {"a": 1}),
        ("not json", None),
        ("[1, 2]", None),
        ("{broken", None),
    ],
)
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #


def test_plan_uses_model_queries():
    llm = FakeLLM(plan='{"queries": ["a", "b"]}')
    out = make_llm_nodes(llm)[PLAN](initial_state("q"))
    assert out["pending_queries"] == ["a", "b"]


def test_plan_caps_and_dedupes_queries():
    queries = ["a", "A", " ", "b", "c", "d", "e"]
    llm = FakeLLM(plan=json.dumps({"queries": queries}))
    out = make_llm_nodes(llm)[PLAN](initial_state("q"))
    assert out["pending_queries"] == ["a", "b", "c"][:MAX_PLAN_QUERIES]


@pytest.mark.parametrize("reply", ["garbage", '{"queries": []}', '{"queries": "a"}'])
def test_plan_falls_back_to_question(reply):
    out = make_llm_nodes(FakeLLM(plan=reply))[PLAN](initial_state("what is RAG?"))
    assert out["pending_queries"] == ["what is RAG?"]


# --------------------------------------------------------------------------- #
# reflect
# --------------------------------------------------------------------------- #


def test_reflect_sufficient_clears_follow_ups():
    llm = FakeLLM(reflect='{"sufficient": true, "follow_up_queries": ["x"]}')
    out = make_llm_nodes(llm)[REFLECT]({"question": "q", "evidence": [_evidence("q")]})
    assert out["sufficient"] is True
    assert out["pending_queries"] == []


def test_reflect_drops_already_run_queries_and_caps():
    llm = FakeLLM(
        reflect=json.dumps(
            {"sufficient": False, "follow_up_queries": ["Q", "new1", "new2", "new3"]}
        )
    )
    out = make_llm_nodes(llm)[REFLECT]({"question": "q", "evidence": [_evidence("q")]})
    assert out["sufficient"] is False
    assert out["pending_queries"] == ["new1", "new2", "new3"][:MAX_FOLLOW_UPS]


def test_reflect_unparseable_stops_research():
    out = make_llm_nodes(FakeLLM(reflect="no idea"))[REFLECT]({"question": "q", "evidence": []})
    assert out == {"sufficient": True, "pending_queries": [], "trace": out["trace"]}


def test_reflect_schema_asks_for_reason_before_verdict():
    assert list(REFLECT_SCHEMA["properties"])[:2] == ["reason", "sufficient"]


def test_reflect_with_reason_field_stops():
    llm = FakeLLM(reflect='{"reason": "covers it", "sufficient": true, "follow_up_queries": []}')
    out = make_llm_nodes(llm)[REFLECT]({"question": "q", "evidence": [_evidence("q")]})
    assert out["sufficient"] is True and out["pending_queries"] == []


def test_reflect_non_boolean_sufficient_is_false():
    llm = FakeLLM(reflect='{"sufficient": "yes", "follow_up_queries": ["more"]}')
    out = make_llm_nodes(llm)[REFLECT]({"question": "q", "evidence": []})
    assert out["sufficient"] is False
    assert out["pending_queries"] == ["more"]


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #


def test_answer_prompt_numbers_evidence_best_first_and_resolves_citations():
    llm = FakeLLM(answer="  The answer [1].  ")
    state = {
        "question": "q",
        "evidence": [_evidence("q", "low.md", score=0.1), _evidence("q", "high.md", score=0.9)],
    }
    out = make_llm_nodes(llm)[ANSWER](state)
    assert out["answer"] == "The answer [1].\n\nSources:\n[1] high.md"
    prompt = llm.calls[0][1][-1].content
    assert "[1] high.md" in prompt and "[2] low.md" in prompt


def test_answer_without_citations_is_left_unchanged():
    llm = FakeLLM(answer="No idea.")
    out = make_llm_nodes(llm)[ANSWER]({"question": "q", "evidence": [_evidence("q")]})
    assert out["answer"] == "No idea."


def test_answer_system_prompt_demands_citations():
    llm = FakeLLM()
    make_llm_nodes(llm)[ANSWER]({"question": "q", "evidence": []})
    system = llm.calls[0][1][0].content
    assert "[1]" in system and "cite" in system.lower()


# --------------------------------------------------------------------------- #
# Evidence ranking and citations
# --------------------------------------------------------------------------- #


def test_rank_evidence_dedupes_keeping_best_score_and_sorts():
    ev = [
        _evidence("q1", "a.md", "same", 0.2),
        _evidence("q2", "b.md", "other", 0.5),
        _evidence("q3", "a.md", "same", 0.7),  # duplicate chunk, better score
    ]
    ranked = rank_evidence(ev)
    assert [(e["source"], e["score"]) for e in ranked] == [("a.md", 0.7), ("b.md", 0.5)]


def test_rank_evidence_caps_items():
    ev = [_evidence("q", f"{i}.md", score=i / 100) for i in range(MAX_EVIDENCE_IN_PROMPT + 5)]
    assert len(rank_evidence(ev)) == MAX_EVIDENCE_IN_PROMPT


@pytest.mark.parametrize(
    "text, n, expected",
    [
        ("A [2]. B [1][2]. C [3].", 3, [2, 1, 3]),
        ("Out of range [0] [4] [99].", 3, []),
        ("No citations.", 3, []),
        ("Mixed [1] and [7].", 2, [1]),
    ],
)
def test_cited_numbers(text, n, expected):
    assert cited_numbers(text, n) == expected


def test_append_sources_lists_only_cited_items_in_order():
    items = [_evidence("q", s) for s in ("a.md", "b.md", "c.md")]
    assert append_sources("X [3]. Y [1].", items) == "X [3]. Y [1].\n\nSources:\n[1] a.md\n[3] c.md"


# --------------------------------------------------------------------------- #
# Full graph with model-driven routing
# --------------------------------------------------------------------------- #


def test_model_driven_follow_up_loop():
    """reflect asks for one follow-up, then is satisfied."""
    verdicts = iter(
        [
            '{"sufficient": false, "follow_up_queries": ["deeper"]}',
            '{"sufficient": true, "follow_up_queries": []}',
        ]
    )
    llm = FakeLLM(reflect=lambda _msgs: next(verdicts))
    result = build_graph(llm=llm).invoke(initial_state("q", max_iterations=5))
    assert node_path(result) == [PLAN, SEARCH, REFLECT, SEARCH, REFLECT, ANSWER]
    assert {e["query"] for e in result["evidence"]} == {"q", "deeper"}
    assert llm.kinds() == ["plan", "reflect", "reflect", "answer"]


def test_repeated_follow_up_does_not_loop():
    """A model that keeps proposing an already-run query ends the loop."""
    llm = FakeLLM(reflect='{"sufficient": false, "follow_up_queries": ["q"]}')
    result = build_graph(llm=llm).invoke(initial_state("q", max_iterations=5))
    assert node_path(result) == [PLAN, SEARCH, REFLECT, ANSWER]
