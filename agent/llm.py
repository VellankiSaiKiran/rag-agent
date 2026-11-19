"""Chat model construction.

Defaults to llama3.2 on a local Ollama server. Override with the
``OLLAMA_MODEL`` / ``OLLAMA_BASE_URL`` environment variables.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"


class ChatModel(Protocol):
    """The slice of the chat-model interface the nodes rely on."""

    def invoke(self, input: list[BaseMessage], **kwargs: Any) -> BaseMessage: ...


def get_llm() -> ChatOllama:
    """Build the default chat model. Construction does not contact the server."""
    return ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        temperature=0,
    )
