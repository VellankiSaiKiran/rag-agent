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
    make_llm_nodes,
    parse_json_object,
)
from agent.state import initial_state
from agent.trace import node_path
from tests.fakes import FakeLLM


def _evidence(query: str, source: str = "stub://x") -> dict:
    return {"query": query, "source": source, "content": "c", "score": 0.0}


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


def test_reflect_non_boolean_sufficient_is_false():
    llm = FakeLLM(reflect='{"sufficient": "yes", "follow_up_queries": ["more"]}')
    out = make_llm_nodes(llm)[REFLECT]({"question": "q", "evidence": []})
    assert out["sufficient"] is False
    assert out["pending_queries"] == ["more"]


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #


def test_answer_prompt_numbers_evidence():
    llm = FakeLLM(answer="  The answer [2].  ")
    state = {"question": "q", "evidence": [_evidence("q", "stub://a"), _evidence("q", "stub://b")]}
    out = make_llm_nodes(llm)[ANSWER](state)
    assert out["answer"] == "The answer [2]."
    prompt = llm.calls[0][1][-1].content
    assert "[1] stub://a" in prompt and "[2] stub://b" in prompt


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
