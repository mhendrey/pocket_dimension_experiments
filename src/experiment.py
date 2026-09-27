"""
Experiment runner

This module orchestrates the complete experiment workflow, including
loading datasets, running retrieval methods, and evaluating results.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.loader import (
    convert_to_beir,
    load_quora_dataset,
    make_synthetic_beir_dataset,
)
from src.retrieval.bm25 import run_bm25_baseline
from src.retrieval.pocket_dimension import run_pocket_dimension_pipeline
from src.evaluation.metrics import evaluate_retrieval


def run_experiment(
    output_dir: str | Path | None = None, d: int = 128, testing: bool = False
) -> dict:
    """
    Run a complete experiment comparing BM25 and Pocket Dimension retrieval.

    Args:
        output_dir: Directory to save artifacts (default: "artifacts")
        d: Dimensionality of dense vectors for Pocket Dimension

    Returns:
        Dictionary containing summary of results and metrics
    """
    output_dir = Path(output_dir) if output_dir is not None else Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Quora dataset
    print("Loading Quora dataset...")
    corpus_ds, queries_ds, qrels_ds = (
        load_quora_dataset() if not testing else make_synthetic_beir_dataset()
    )
    corpus, queries, qrels = convert_to_beir(corpus_ds, queries_ds, qrels_ds)

    print("Starting BM25 baseline...")
    bm25 = run_bm25_baseline(corpus, queries, top_k=10, artifact_dir=output_dir)

    print("Starting Pocket Dimension pipeline...")
    pocket = run_pocket_dimension_pipeline(
        corpus, queries, top_k=10, artifact_dir=output_dir, d=d
    )

    print("Evaluating BM25 results...")
    bm25_metrics = evaluate_retrieval(bm25["results"], qrels, [1, 3, 5, 10])

    print("Evaluating Pocket Dimension results...")
    pocket_metrics = evaluate_retrieval(pocket["results"], qrels, [1, 3, 5, 10])

    summary = {
        "artifacts_dir": str(output_dir),
        "bm25": {
            "results_path": str(bm25["results_path"]),
            "index_path": str(bm25["index_path"]),
            "index_size_bytes": bm25["index_size_bytes"],
            "initialization_time_seconds": bm25["initialization_time_seconds"],
            "query_time_seconds": bm25["query_time_seconds"],
            "index_save_time_seconds": bm25["index_save_time_seconds"],
            "metrics": bm25_metrics,
        },
        "pocket_dimension": {
            "results_path": str(pocket["results_path"]),
            "index_path": str(pocket["index_path"]),
            "index_size_bytes": pocket["index_size_bytes"],
            "initialization_time_seconds": pocket["initialization_time_seconds"],
            "query_time_seconds": pocket["query_time_seconds"],
            "index_save_time_seconds": pocket["index_save_time_seconds"],
            "metrics": pocket_metrics,
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary
