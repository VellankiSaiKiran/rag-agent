"""Run the agent over eval/questions.jsonl and report keyword recall.

Usage: python -m eval.run_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.graph import run

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


def main() -> None:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    scores = []
    for row in rows:
        result = run(row["question"])
        score = keyword_recall(result["answer"], row.get("expected_keywords", []))
        scores.append(score)
        print(f"{row['id']}: recall={score:.2f} steps={len(result['trace'])}")
    if scores:
        print(f"mean recall: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
