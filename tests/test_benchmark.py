from semantic_search.benchmark import run_benchmark


def test_benchmark_uses_standard_metric_names():
    report = run_benchmark(messages=200, queries=10, dimension=64)
    assert report["messages"] == 200
    assert report["queries"] == 10
    assert 0 <= report["hit_rate_at_10"] <= 1
    assert 0 <= report["precision_at_10"] <= 1
    assert 0 <= report["mrr_at_10"] <= 1
    assert "recall_at_10" not in report
