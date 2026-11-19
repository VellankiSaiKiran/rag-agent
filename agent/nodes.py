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
    # "reason" comes first so the model explains itself before giving its verdict.
    "properties": {
        "reason": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "follow_up_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reason", "sufficient", "follow_up_queries"],
}

PLAN_SYSTEM = (
    "You are a research planner. Break the user's question into at most "
    f"{MAX_PLAN_QUERIES} concise web search queries that together cover what is "
    'needed to answer it. Respond with JSON: {"queries": ["..."]}.'
)
REFLECT_SYSTEM = (
    "You decide whether a research assistant should stop searching.\n"
    "Set sufficient to true when the evidence directly answers the main point of "
    "the question, even if some details are missing. The answer does not need to be "
    "exhaustive. Extra searching costs time and rarely helps.\n"
    "Set sufficient to false only when the evidence does not address the main point, "
    f"and then give at most {MAX_FOLLOW_UPS} new, specific search queries for the "
    "missing information. Never repeat a query that was already run.\n"
    "First write a one-sentence reason, then the verdict. Respond with JSON: "
    '{"reason": "...", "sufficient": true|false, "follow_up_queries": ["..."]}.'
)
ANSWER_SYSTEM = (
    "You are a careful research assistant. Answer the question using only the "
    "numbered evidence provided.\n"
    "Rules:\n"
    "- After every sentence that uses the evidence, cite it with its number in square "
    "brackets, like [1] or [2][3]. Use only numbers that appear in the evidence list.\n"
    "- Do not add a sources list or reference section yourself.\n"
    "- If the evidence does not answer the question, say so plainly.\n"
    "Example: Ollama runs models locally [1]. Its server listens on port 11434 [2]."
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


def rank_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """De-duplicate evidence by (source, content) and order it best-first.

    The same chunk is often retrieved by several queries; it should appear once,
    so each citation number maps to one distinct passage. Ties keep the order in
    which evidence was found. Capped at MAX_EVIDENCE_IN_PROMPT.
    """
    best: dict[tuple[str, str], Evidence] = {}
    for e in evidence:
        key = (e["source"], e["content"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    ranked = sorted(best.values(), key=lambda e: -e["score"])  # stable sort
    return ranked[:MAX_EVIDENCE_IN_PROMPT]


def format_evidence(items: list[Evidence]) -> str:
    """Number already-ranked evidence as [1], [2], ... for a prompt."""
    if not items:
        return "(no evidence gathered)"
    return "\n\n".join(
        f"[{i}] {e['source']}\n{e['content'][:MAX_CHARS_PER_EVIDENCE]}"
        for i, e in enumerate(items, start=1)
    )


_CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_numbers(text: str, n_items: int) -> list[int]:
    """Valid citation numbers in ``text``, in order of first appearance."""
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_items and n not in seen:
            seen.append(n)
    return seen


def append_sources(answer: str, items: list[Evidence]) -> str:
    """Add a 'Sources:' list resolving each cited [n] to its file name."""
    cited = cited_numbers(answer, len(items))
    if not cited:
        return answer
    lines = [f"[{n}] {items[n - 1]['source']}" for n in sorted(cited)]
    return answer + "\n\nSources:\n" + "\n".join(lines)


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
            f"Evidence:\n{format_evidence(rank_evidence(evidence))}"
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
        items = rank_evidence(state.get("evidence", []))
        prompt = (
            f"Question: {state.get('question', '')}\n\n"
            f"Evidence:\n{format_evidence(items)}"
        )
        reply = llm.invoke([SystemMessage(ANSWER_SYSTEM), HumanMessage(prompt)])
        return {"answer": append_sources(_text(reply).strip(), items)}

    return {PLAN: plan, REFLECT: reflect, ANSWER: answer}


@traced(SEARCH)
def search(state: ResearchState) -> dict:
    """Run every pending query against web_cache/ and collect the top chunks."""
    new_evidence: list[Evidence] = []
    for query in state.get("pending_queries", []):
        for hit in tools.web_search(query, k=2):
            new_evidence.append(
                Evidence(
                    query=query,
                    source=hit["url"],
                    content=hit["content"],
                    score=hit["score"],
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
