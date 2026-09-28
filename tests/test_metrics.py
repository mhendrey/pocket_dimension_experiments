import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import metrics


def test_evaluate_retrieval_delegates_and_combines_metric_groups(monkeypatch):
    qrels = {"query-1": {"doc-1": 1}}
    retrieval_results = {"query-1": {"doc-1": 0.9}}
    k_values = [1, 3]
    metric_groups = (
        {"NDCG@1": 1.0},
        {"MAP@1": 1.0},
        {"Recall@1": 1.0},
        {"P@1": 1.0},
    )
    calls = []

    class FakeEvaluator:
        def evaluate(self, received_qrels, received_results, received_k_values):
            calls.append((received_qrels, received_results, received_k_values))
            return metric_groups

    monkeypatch.setattr(metrics, "EvaluateRetrieval", FakeEvaluator)

    result = metrics.evaluate_retrieval(retrieval_results, qrels, k_values)

    assert calls == [(qrels, retrieval_results, k_values)]
    assert result == {
        "NDCG@1": 1.0,
        "MAP@1": 1.0,
        "Recall@1": 1.0,
        "P@1": 1.0,
    }