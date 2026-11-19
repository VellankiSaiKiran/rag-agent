"""Graph nodes and routing functions.

``plan``, ``reflect`` and ``answer`` are LLM-backed and are built by
``make_llm_nodes(llm)``, so the model is injected rather than global.
``search`` still uses the stubbed tools. Routing functions are pure and are
the contract the tests pin down.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agent import tools
from agent.llm import ChatModel
from agent.state import DEFAULT_MAX_ITERATIONS, Evidence, ResearchState
from agent.trace import traced

NodeFn = Callable[[ResearchState], dict]

# Node names (single source of truth for graph.py and tests)
PLAN = "plan"
SEARCH = "search"
REFLECT = "reflect"
ANSWER = "answer"

MAX_PLAN_QUERIES = 3
MAX_FOLLOW_UPS = 2
MAX_EVIDENCE_IN_PROMPT = 10
MAX_CHARS_PER_EVIDENCE = 1500

# JSON schemas passed to Ollama's `format` option to constrain output.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}
REFLECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "follow_up_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "follow_up_queries"],
}

PLAN_SYSTEM = (
    "You are a research planner. Break the user's question into at most "
    f"{MAX_PLAN_QUERIES} concise web search queries that together cover what is "
    'needed to answer it. Respond with JSON: {"queries": ["..."]}.'
)
REFLECT_SYSTEM = (
    "You review research progress. Given a question and the evidence gathered so "
    "far, decide whether the evidence is sufficient to answer the question well. "
    f"If not, propose at most {MAX_FOLLOW_UPS} new search queries that would fill "
    "the gaps; do not repeat queries already run. Respond with JSON: "
    '{"sufficient": true|false, "follow_up_queries": ["..."]}.'
)
ANSWER_SYSTEM = (
    "You are a careful research assistant. Answer the question using only the "
    "numbered evidence provided. Cite sources inline as [n]. If the evidence does "
    "not support an answer, say so plainly."
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating surrounding prose."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def clean_queries(raw: Any, limit: int, exclude: set[str] = frozenset()) -> list[str]:
    """Keep non-empty unique strings not in ``exclude`` (case-insensitive)."""
    if not isinstance(raw, list):
        return []
    seen = {q.lower() for q in exclude}
    out: list[str] = []
    for q in raw:
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
        if len(out) == limit:
            break
    return out


def format_evidence(evidence: list[Evidence]) -> str:
    """Number the most recent evidence for inclusion in a prompt."""
    if not evidence:
        return "(no evidence gathered)"
    items = evidence[-MAX_EVIDENCE_IN_PROMPT:]
    return "\n\n".join(
        f"[{i}] {e['source']}\n{e['content'][:MAX_CHARS_PER_EVIDENCE]}"
        for i, e in enumerate(items, start=1)
    )


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def make_llm_nodes(llm: ChatModel) -> dict[str, NodeFn]:
    """Build the LLM-backed nodes bound to ``llm``."""

    @traced(PLAN)
    def plan(state: ResearchState) -> dict:
        question = state.get("question", "").strip()
        if not question:
            return {"pending_queries": []}
        reply = llm.invoke(
            [SystemMessage(PLAN_SYSTEM), HumanMessage(question)], format=PLAN_SCHEMA
        )
        data = parse_json_object(_text(reply)) or {}
        queries = clean_queries(data.get("queries"), MAX_PLAN_QUERIES)
        # Fall back to searching the question itself rather than giving up.
        return {"pending_queries": queries or [question]}

    @traced(REFLECT)
    def reflect(state: ResearchState) -> dict:
        evidence = state.get("evidence", [])
        already_run = {e["query"] for e in evidence}
        prompt = (
            f"Question: {state.get('question', '')}\n\n"
            f"Queries already run: {json.dumps(sorted(already_run))}\n\n"
            f"Evidence:\n{format_evidence(evidence)}"
        )
        reply = llm.invoke(
            [SystemMessage(REFLECT_SYSTEM), HumanMessage(prompt)], format=REFLECT_SCHEMA
        )
        data = parse_json_object(_text(reply))
        if data is None:
            # Unparseable verdict: stop researching rather than loop blindly.
            return {"sufficient": True, "pending_queries": []}
        sufficient = data.get("sufficient") is True
        follow_ups = [] if sufficient else clean_queries(
            data.get("follow_up_queries"), MAX_FOLLOW_UPS, exclude=already_run
        )
        return {"sufficient": sufficient, "pending_queries": follow_ups}

    @traced(ANSWER)
    def answer(state: ResearchState) -> dict:
        prompt = (
            f"Question: {state.get('question', '')}\n\n"
            f"Evidence:\n{format_evidence(state.get('evidence', []))}"
        )
        reply = llm.invoke([SystemMessage(ANSWER_SYSTEM), HumanMessage(prompt)])
        return {"answer": _text(reply).strip()}

    return {PLAN: plan, REFLECT: reflect, ANSWER: answer}


@traced(SEARCH)
def search(state: ResearchState) -> dict:
    """Run every pending query and collect evidence (tools are still stubs)."""
    new_evidence: list[Evidence] = []
    for query in state.get("pending_queries", []):
        for hit in tools.web_search(query, k=2):
            new_evidence.append(
                Evidence(
                    query=query,
                    source=hit["url"],
                    content=tools.fetch_page(hit["url"]),
                    score=0.0,
                )
            )
    return {
        "evidence": new_evidence,
        "pending_queries": [],
        "iteration": state.get("iteration", 0) + 1,
    }


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def route_after_plan(state: ResearchState) -> Literal["search", "answer"]:
    """Search if the planner produced queries; otherwise answer directly."""
    return SEARCH if state.get("pending_queries") else ANSWER


def route_after_reflect(state: ResearchState) -> Literal["search", "answer"]:
    """Loop back to search only when evidence is insufficient, there are
    follow-up queries to run, and the iteration budget is not spent."""
    budget = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if state.get("iteration", 0) >= budget:
        return ANSWER
    if state.get("sufficient"):
        return ANSWER
    if state.get("pending_queries"):
        return SEARCH
    return ANSWER
