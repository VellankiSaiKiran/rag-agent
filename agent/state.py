"""Graph state for the research agent.

The state is a TypedDict so LangGraph can merge partial updates returned by
nodes. Fields annotated with a reducer (``operator.add``) accumulate across
steps; all other fields are overwritten by the latest update.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

DEFAULT_MAX_ITERATIONS = 3


class Evidence(TypedDict):
    """One piece of retrieved material that may support the answer."""

    query: str
    source: str  # URL or cache key
    content: str
    score: float  # relevance to the question, higher is better


class TraceEvent(TypedDict):
    """One node execution, recorded by ``agent.trace.traced``."""

    node: str
    step: int
    duration_ms: float


class ResearchState(TypedDict, total=False):
    # --- input ---
    question: str
    max_iterations: int

    # --- planning ---
    pending_queries: list[str]  # queries still to run (overwritten each update)

    # --- research loop ---
    evidence: Annotated[list[Evidence], operator.add]
    iteration: int  # number of completed search rounds
    sufficient: bool  # reflect node's verdict: enough evidence to answer?

    # --- output ---
    answer: str

    # --- observability ---
    trace: Annotated[list[TraceEvent], operator.add]


def initial_state(question: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> ResearchState:
    """Build a fully populated starting state for a run."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    return ResearchState(
        question=question,
        max_iterations=max_iterations,
        pending_queries=[],
        evidence=[],
        iteration=0,
        sufficient=False,
        answer="",
        trace=[],
    )
