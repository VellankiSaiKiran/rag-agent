from __future__ import annotations

import pytest

from tests.fakes import FakeLLM


@pytest.fixture(autouse=True)
def fake_default_llm(monkeypatch) -> FakeLLM:
    """Never reach a real Ollama server from tests: ``build_graph()`` without
    an explicit ``llm`` gets this fake instead of ChatOllama."""
    fake = FakeLLM()
    monkeypatch.setattr("agent.graph.get_llm", lambda: fake)
    return fake
