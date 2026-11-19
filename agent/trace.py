"""Lightweight per-node tracing.

Wrap a node with ``traced("name")`` and each call appends a ``TraceEvent`` to
``state["trace"]`` (merged via the list reducer declared in ``state.py``).
"""

from __future__ import annotations

import functools
import time
from typing import Callable

from agent.state import ResearchState, TraceEvent

Node = Callable[[ResearchState], dict]


def traced(name: str) -> Callable[[Node], Node]:
    def decorator(fn: Node) -> Node:
        @functools.wraps(fn)
        def wrapper(state: ResearchState) -> dict:
            start = time.perf_counter()
            update = dict(fn(state) or {})
            event = TraceEvent(
                node=name,
                step=len(state.get("trace", [])),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            update["trace"] = [*update.get("trace", []), event]
            return update

        return wrapper

    return decorator


def node_path(state: ResearchState) -> list[str]:
    """The sequence of node names executed so far."""
    return [e["node"] for e in state.get("trace", [])]
