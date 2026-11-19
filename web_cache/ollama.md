# Ollama

Ollama is a tool for running large language models on your own computer. Instead of sending prompts to a cloud service, you download a model once and run it locally, which keeps your data on your machine and avoids per-request costs.

Getting started takes a couple of commands. You pull a model by name, for example `ollama pull llama3.2`, and Ollama downloads it and stores it for reuse. You can then chat with it in the terminal, or let other programs talk to it.

Ollama runs a small server in the background, by default at `http://localhost:11434`. Applications send requests to this address and get the model's replies back. Libraries such as LangChain have ready-made connectors for it, so code written for a hosted model can often be switched to a local one with little change.

Local models are usually smaller than the largest hosted ones, and they are often compressed to fit in ordinary memory. That makes them fast and cheap, but sometimes less capable on hard reasoning tasks. Ollama can also ask a model to reply in strict JSON, which helps smaller models produce output that a program can parse reliably.
