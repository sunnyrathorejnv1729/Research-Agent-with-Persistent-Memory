"""
Evaluation pipeline for the ResearchAgent.

Metrics measured per query:
  - memory_hit_rate      : fraction of queries answered purely from memory
  - keyword_coverage     : fraction of expected keywords found in the answer
  - latency_seconds      : wall-clock time for the full research turn
  - tool_efficiency      : 1.0 if memory hit, else 0.0
  - web_searches         : number of web_search tool calls
  - url_fetches          : number of fetch_url tool calls
  - total_tool_calls     : total tool calls made

Run:  python evaluation/eval_pipeline.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ResearchAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark dataset
# ---------------------------------------------------------------------------

BENCHMARK = [
    {
        "query": "What is the capital of France?",
        "keywords": ["paris", "france", "capital"],
    },
    {
        "query": "Explain how transformer neural networks work",
        "keywords": ["attention", "encoder", "decoder", "transformer"],
    },
    {
        "query": "What are the main benefits of using Python for data science?",
        "keywords": ["python", "data", "libraries", "pandas", "numpy"],
    },
    {
        "query": "How does the TCP/IP protocol stack work?",
        "keywords": ["tcp", "ip", "network", "protocol", "layers"],
    },
    {
        "query": "What is quantum entanglement?",
        "keywords": ["quantum", "entanglement", "particles", "correlation"],
    },
    # Repeated queries — should hit memory after first run
    {
        "query": "What is the capital of France?",
        "keywords": ["paris", "france"],
        "note": "repeat — expect memory hit",
    },
    {
        "query": "Explain how transformer neural networks work",
        "keywords": ["attention", "transformer"],
        "note": "repeat — expect memory hit",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def keyword_coverage(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


def run_evaluation(save_path: str = "evaluation/eval_results.json") -> dict:
    agent = ResearchAgent()
    records = []

    for i, item in enumerate(BENCHMARK):
        query    = item["query"]
        keywords = item["keywords"]
        note     = item.get("note", "")

        logger.info("─── Query %d/%d: %s %s", i + 1, len(BENCHMARK), query, f"({note})" if note else "")

        trace = agent.research(query)

        record = {
            "query_id": i + 1,
            "query": query,
            "note": note,
            "answer_preview": trace.answer[:200],
            "memory_hit": trace.memory_hit,
            "web_searches": trace.web_searches,
            "url_fetches": trace.url_fetches,
            "total_tool_calls": len(trace.tool_calls),
            "tool_efficiency": trace.tool_efficiency,
            "latency_seconds": trace.latency,
            "keyword_coverage": keyword_coverage(trace.answer, keywords),
            "error": trace.error,
        }
        records.append(record)
        logger.info(
            "  ✓ memory_hit=%s | kw_coverage=%.0f%% | latency=%.2fs",
            record["memory_hit"],
            record["keyword_coverage"] * 100,
            record["latency_seconds"],
        )

    # Aggregate
    summary = _aggregate(records)

    output = {"summary": summary, "records": records}
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(output, fh, indent=2)

    logger.info("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)
    logger.info("Results saved → %s", save_path)

    return output


def _aggregate(records: list[dict]) -> dict:
    n = len(records)
    hits    = sum(1 for r in records if r["memory_hit"])
    repeats = [r for r in records if r.get("note", "").startswith("repeat")]
    repeat_hits = sum(1 for r in repeats if r["memory_hit"])

    return {
        "total_queries": n,
        "memory_hit_rate": round(hits / n, 3),
        "repeat_memory_hit_rate": round(repeat_hits / len(repeats), 3) if repeats else None,
        "avg_keyword_coverage": round(sum(r["keyword_coverage"] for r in records) / n, 3),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in records) / n, 3),
        "avg_tool_calls": round(sum(r["total_tool_calls"] for r in records) / n, 2),
        "avg_web_searches": round(sum(r["web_searches"] for r in records) / n, 2),
        "avg_url_fetches": round(sum(r["url_fetches"] for r in records) / n, 2),
        "total_errors": sum(1 for r in records if r["error"]),
    }


if __name__ == "__main__":
    run_evaluation()
