"""A scripted stand-in for ChatOllama.

Dispatches on the ``format`` schema each node passes, so one fake can serve
plan, reflect and answer. Each responder is a string or a callable taking the
message list. Every call is recorded in ``calls`` for assertions.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage

from agent.nodes import PLAN_SCHEMA, REFLECT_SCHEMA

Responder = str | Callable[[list[BaseMessage]], str]


def echo_question_plan(messages: list[BaseMessage]) -> str:
    """Plan a single query: the question itself (the plan node's human message)."""
    return json.dumps({"queries": [messages[-1].content]})


class FakeLLM:
    def __init__(
        self,
        plan: Responder = echo_question_plan,
        reflect: Responder = '{"sufficient": true, "follow_up_queries": []}',
        answer: Responder = "Fake answer [1].",
    ) -> None:
        self.responders = {"plan": plan, "reflect": reflect, "answer": answer}
        self.calls: list[tuple[str, list[BaseMessage]]] = []

    def invoke(self, input: list[BaseMessage], **kwargs: Any) -> AIMessage:
        fmt = kwargs.get("format")
        kind = "plan" if fmt == PLAN_SCHEMA else "reflect" if fmt == REFLECT_SCHEMA else "answer"
        self.calls.append((kind, input))
        responder = self.responders[kind]
        return AIMessage(responder(input) if callable(responder) else responder)

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]
