"""Tests for the routing edges: the pure routing functions, and the paths the
compiled graph actually takes with stub (or overridden) nodes."""

from __future__ import annotations

import pytest

from agent.graph import build_graph
from agent.nodes import (
    ANSWER,
    PLAN,
    REFLECT,
    SEARCH,
    route_after_plan,
    route_after_reflect,
)
from agent.state import initial_state
from agent.trace import node_path, traced

# --------------------------------------------------------------------------- #
# route_after_plan
# --------------------------------------------------------------------------- #


def test_plan_with_queries_goes_to_search():
    assert route_after_plan({"pending_queries": ["q"]}) == SEARCH


@pytest.mark.parametrize("state", [{"pending_queries": []}, {}])
def test_plan_without_queries_goes_to_answer(state):
    assert route_after_plan(state) == ANSWER


# --------------------------------------------------------------------------- #
# route_after_reflect
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state, expected",
    [
        # sufficient -> answer, even with follow-ups queued
        ({"sufficient": True, "pending_queries": ["more"], "iteration": 1, "max_iterations": 3}, ANSWER),
        # insufficient + follow-ups + budget left -> search
        ({"sufficient": False, "pending_queries": ["more"], "iteration": 1, "max_iterations": 3}, SEARCH),
        # insufficient but nothing left to search -> answer
        ({"sufficient": False, "pending_queries": [], "iteration": 1, "max_iterations": 3}, ANSWER),
        # budget exactly spent -> answer
        ({"sufficient": False, "pending_queries": ["more"], "iteration": 3, "max_iterations": 3}, ANSWER),
        # budget overspent -> answer
        ({"sufficient": False, "pending_queries": ["more"], "iteration": 5, "max_iterations": 3}, ANSWER),
        # last iteration still available -> search
        ({"sufficient": False, "pending_queries": ["more"], "iteration": 2, "max_iterations": 3}, SEARCH),
    ],
)
def test_route_after_reflect(state, expected):
    assert route_after_reflect(state) == expected


def test_reflect_uses_default_budget_when_missing():
    # DEFAULT_MAX_ITERATIONS is 3
    assert route_after_reflect({"pending_queries": ["q"], "iteration": 2}) == SEARCH
    assert route_after_reflect({"pending_queries": ["q"], "iteration": 3}) == ANSWER


# --------------------------------------------------------------------------- #
# Compiled graph paths
# --------------------------------------------------------------------------- #


def test_happy_path(fake_default_llm):
    result = build_graph().invoke(initial_state("what is RAG?"))
    assert node_path(result) == [PLAN, SEARCH, REFLECT, ANSWER]
    assert result["iteration"] == 1
    assert result["evidence"]
    assert result["answer"] == "Fake answer [1]."
    assert fake_default_llm.kinds() == ["plan", "reflect", "answer"]


def test_empty_plan_skips_search():
    @traced(PLAN)
    def empty_plan(state):
        return {"pending_queries": []}

    result = build_graph({PLAN: empty_plan}).invoke(initial_state("q"))
    assert node_path(result) == [PLAN, ANSWER]
    assert result["evidence"] == []


@traced(REFLECT)
def never_satisfied(state):
    return {"sufficient": False, "pending_queries": [f"follow-up {state['iteration']}"]}


@pytest.mark.parametrize("budget", [1, 2, 4])
def test_reflect_loop_respects_iteration_budget(budget):
    result = build_graph({REFLECT: never_satisfied}).invoke(initial_state("q", max_iterations=budget))
    expected = [PLAN] + [SEARCH, REFLECT] * budget + [ANSWER]
    assert node_path(result) == expected
    assert result["iteration"] == budget


def test_reflect_loop_stops_when_sufficient():
    @traced(REFLECT)
    def satisfied_on_second(state):
        done = state["iteration"] >= 2
        return {"sufficient": done, "pending_queries": [] if done else ["more"]}

    result = build_graph({REFLECT: satisfied_on_second}).invoke(initial_state("q", max_iterations=5))
    assert node_path(result) == [PLAN, SEARCH, REFLECT, SEARCH, REFLECT, ANSWER]


def test_evidence_accumulates_across_loops():
    result = build_graph({REFLECT: never_satisfied}).invoke(initial_state("q", max_iterations=2))
    queries = {e["query"] for e in result["evidence"]}
    assert queries == {"q", "follow-up 1"}


def test_unknown_override_rejected():
    with pytest.raises(ValueError):
        build_graph({"nope": lambda s: {}})


@pytest.mark.parametrize("kwargs", [{"question": "  "}, {"question": "q", "max_iterations": 0}])
def test_initial_state_validation(kwargs):
    with pytest.raises(ValueError):
        initial_state(**kwargs)
