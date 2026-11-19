# rag-agent

A LangGraph research agent that plans search queries, gathers evidence, decides whether it has enough, and writes an answer. It's meant to run on local models through Ollama.

**Status:** `plan`, `reflect` and `answer` call llama3.2 through Ollama. `search` retrieves the most relevant chunks from the markdown pages in `web_cache/`.

## Graph

```
START → plan ─┬─(has queries)──→ search → reflect ─┬─(sufficient / no follow-ups / budget spent)─→ answer → END
              │                     ▲              │
              │                     └──(follow-ups)┘
              └─(no queries)───────────────────────────────────────────────────────────────────→ answer
```

The `reflect → search` loop is capped by `max_iterations` (default 3), so a run always finishes.

## Layout

| Path | Purpose |
|---|---|
| `agent/state.py` | `ResearchState` TypedDict. `evidence` and `trace` accumulate across steps; all other fields are overwritten. |
| `agent/llm.py` | `get_llm()` builds `ChatOllama` (llama3.2 at `localhost:11434`, temperature 0). Override with `OLLAMA_MODEL` and `OLLAMA_BASE_URL`. |
| `agent/nodes.py` | `make_llm_nodes(llm)` builds the LLM-backed `plan`, `reflect` and `answer` nodes. Also holds `search` and the pure routing functions. |
| `agent/graph.py` | `build_graph(overrides=None, llm=None)` wires and compiles the graph. `run(question)` is a convenience entry point. |
| `agent/tools.py` | `web_search(query, k)` splits `web_cache/*.md` into chunks, embeds them, and returns the top-k chunks ranked by `cosine_similarity`. Each hit's `url` is the page's file name. The index rebuilds when pages change. |
| `agent/embedder.py` | `HashingEmbedder`, a feature-hashing embedder that hashes words and word pairs into a fixed-size unit vector. It has no dependencies and doesn't call a model. |
| `agent/trace.py` | The `@traced(name)` decorator, which records each node's execution in `state["trace"]`. |
| `web_cache/` | The markdown pages the agent searches. Each page is split into paragraph chunks of at most 120 words, with a 20-word overlap when a paragraph is split, and each chunk is prefixed with the page title. |
| `eval/` | `questions.jsonl` and `run_eval.py`, which scores answers by keyword recall. |
| `tests/` | Tests for routing, the LLM nodes, and search and embedding, plus `FakeLLM`. They run offline. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
ollama pull llama3.2      # one-time model download
ollama serve              # unless the Ollama app is already running
```

The tests don't need Ollama.

## Usage

```bash
.venv/bin/pytest                                  # run the tests
.venv/bin/python -m agent.graph "your question"   # run the agent once
.venv/bin/python -m eval.run_eval                 # run the eval set
```

## How the LLM nodes behave

- **plan** asks for up to 3 search queries. It uses Ollama's `format` option with a JSON schema to constrain the output. If the reply is unusable, it searches the question itself.
- **reflect** gives a one-sentence reason and then a verdict. The evidence counts as sufficient once it answers the main point of the question, even if some details are missing. Otherwise it proposes up to 2 follow-up queries. Queries that have already been run are dropped, so a model that repeats itself can't loop. If the reply can't be parsed, research stops and the agent answers with what it has.
- **answer** writes a final answer from up to 10 numbered evidence items. Duplicates are removed, the best match is numbered `[1]`, and each item is truncated to 1,500 characters. The model must cite every sentence that uses evidence as `[n]`. The code then adds a `Sources:` list that maps each cited number to its file in `web_cache/`. Numbers that don't match any evidence item are ignored.

## Testing approach

The routing functions are pure, so they are tested directly. `tests/fakes.py` provides `FakeLLM`, which returns scripted replies for `plan`, `reflect` and `answer`, telling them apart by the JSON schema each node passes. An autouse fixture in `tests/conftest.py` replaces the default model with this fake, so the tests never contact Ollama. Graph-path tests either pass node overrides or script the model's verdicts, then check the exact node sequence using `agent.trace.node_path`.

## Next steps

- Optionally swap `HashingEmbedder` for `OllamaEmbeddings` to match by meaning rather than exact words ("chunks" currently doesn't match "chunking").
- Add a live web search that saves fetched pages into `web_cache/`.
