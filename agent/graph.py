"""Graph wiring.

    START -> plan -?-> search -> reflect -?-> answer -> END
               \\                  |   ^
                \\-> answer       \\--/ (back to search)

``build_graph`` accepts an injected chat model and per-node overrides so tests
(and experiments) can swap pieces without touching the edges.
"""

from __future__ import annotations

from typing import Mapping

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.llm import ChatModel, get_llm
from agent.nodes import ANSWER, PLAN, REFLECT, SEARCH, NodeFn
from agent.state import ResearchState, initial_state

NODE_NAMES = (PLAN, SEARCH, REFLECT, ANSWER)


def build_graph(overrides: Mapping[str, NodeFn] | None = None, llm: ChatModel | None = None):
    """Build and compile the research graph.

    ``llm`` defaults to ``get_llm()`` (llama3.2 via Ollama). ``overrides``
    replaces individual nodes by name; edges are unaffected.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"unknown node override(s): {sorted(unknown)}")
    impls = {**nodes.make_llm_nodes(llm or get_llm()), SEARCH: nodes.search, **overrides}

    g = StateGraph(ResearchState)
    for name in NODE_NAMES:
        g.add_node(name, impls[name])

    g.add_edge(START, PLAN)
    g.add_conditional_edges(PLAN, nodes.route_after_plan, {SEARCH: SEARCH, ANSWER: ANSWER})
    g.add_edge(SEARCH, REFLECT)
    g.add_conditional_edges(REFLECT, nodes.route_after_reflect, {SEARCH: SEARCH, ANSWER: ANSWER})
    g.add_edge(ANSWER, END)
    return g.compile()


def run(question: str, max_iterations: int | None = None) -> ResearchState:
    kwargs = {} if max_iterations is None else {"max_iterations": max_iterations}
    return build_graph().invoke(initial_state(question, **kwargs))


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    result = run(q)
    print(result["answer"])
    print("path:", " -> ".join(e["node"] for e in result["trace"]))
