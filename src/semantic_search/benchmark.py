from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .service import SearchService


def run_benchmark(messages: int, queries: int, dimension: int) -> dict[str, Any]:
    topics = [
        "distributed systems",
        "database indexing",
        "machine learning",
        "incident response",
        "network security",
        "api reliability",
        "mobile performance",
        "cloud deployment",
        "concurrent programming",
        "information retrieval",
    ]
    now = datetime.now(UTC)
    payloads = []
    for message_id in range(messages):
        topic_id = message_id % 100
        topic = topics[topic_id % len(topics)]
        payloads.append(
            {
                "content": (
                    f"topic{topic_id} {topic} engineering note {message_id}. "
                    "Includes implementation details, validation checks, and measurable outcomes."
                ),
                "created_at": (now - timedelta(minutes=message_id % 43_200)).isoformat(),
                "metadata": {"topic": topic_id, "source": "synthetic-benchmark"},
            }
        )

    with tempfile.TemporaryDirectory(prefix="semantic-search-benchmark-") as temporary:
        service = SearchService(Path(temporary) / "benchmark.db", dimension=dimension)
        index_started = time.perf_counter()
        for start in range(0, len(payloads), 1_000):
            service.add_messages(payloads[start : start + 1_000])
        indexing_seconds = time.perf_counter() - index_started
        latencies_ms: list[float] = []
        reciprocal_ranks: list[float] = []
        recall_hits = 0
        for query_id in range(queries):
            topic_id = query_id % 100
            started = time.perf_counter_ns()
            results = service.search(
                f"topic{topic_id} implementation validation", limit=10, now=now
            )
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            relevant_positions = [
                position
                for position, result in enumerate(results, start=1)
                if result.message.metadata.get("topic") == topic_id
            ]
            if relevant_positions:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / relevant_positions[0])
            else:
                reciprocal_ranks.append(0.0)

    ordered = sorted(latencies_ms)
    return {
        "messages": messages,
        "queries": queries,
        "dimension": dimension,
        "backend": service.backend,
        "indexing_seconds": round(indexing_seconds, 3),
        "indexing_messages_per_second": round(messages / indexing_seconds, 1),
        "recall_at_10": round(recall_hits / queries, 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "p99_ms": round(_percentile(ordered, 0.99), 3),
        "query_throughput_per_second": round(1_000 / statistics.mean(latencies_ms), 1),
    }


def _percentile(values: list[float], percentile: float) -> float:
    index = min(len(values) - 1, max(0, int(len(values) * percentile + 0.999999) - 1))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic semantic-search benchmark.")
    parser.add_argument("--messages", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.messages < 1 or args.queries < 1:
        parser.error("messages and queries must be positive")
    results = run_benchmark(args.messages, args.queries, args.dimension)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
