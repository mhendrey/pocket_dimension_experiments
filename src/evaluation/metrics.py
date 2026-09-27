"""
Evaluation metrics calculation

This module provides functions for calculating standard information 
retrieval evaluation metrics.
"""

from __future__ import annotations

from beir.retrieval.evaluation import EvaluateRetrieval


def evaluate_retrieval(
    results: dict, qrels: dict[str, dict[str, int]], k_values: list[int]
) -> dict[str, float]:
    """
    Evaluate retrieval performance using standard IR metrics.

    Args:
        results: Retrieval results in BEIR format
        qrels: Ground truth relevance judgments
        k_values: List of k values to evaluate at (e.g., [1, 3, 5, 10])

    Returns:
        Dictionary of metrics (NDCG, MAP, Recall, Precision) for each k value
    """
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

    metrics = {}
    metrics.update(ndcg)
    metrics.update(_map)
    metrics.update(recall)
    metrics.update(precision)

    return metrics
