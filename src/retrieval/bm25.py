"""
BM25 baseline implementation

This module provides the traditional BM25 lexical retrieval method as a baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import BM25


def run_bm25_baseline(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    top_k: int,
    artifact_dir: Path,
) -> dict:
    """
    Run BM25 baseline retrieval on the given corpus and queries.

    Args:
        corpus: Dictionary of documents in BEIR format
        queries: Dictionary of queries
        top_k: Number of results to return per query
        artifact_dir: Directory to save artifacts (results, index)

    Returns:
        Dictionary containing results and performance metrics
    """
    # Create ordered mapping from corpus IDs to texts
    doc_ids = list(corpus.keys())  # Keep original IDs in order
    doc_texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

    # Create index → original_id mapping
    idx_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}

    init_start = time.perf_counter()
    retriever = BM25.index(doc_texts)
    init_elapsed = time.perf_counter() - init_start

    query_order = list(queries.keys())
    query_texts = [queries[qid] for qid in query_order]
    search_start = time.perf_counter()
    raw_results = retriever.search(query_texts, k=top_k)
    search_elapsed = time.perf_counter() - search_start

    index_dir = artifact_dir / "bm25_index"
    index_save_start = time.perf_counter()
    retriever.retriever.save(str(index_dir), corpus=doc_texts)
    index_save_elapsed = time.perf_counter() - index_save_start

    results = {}
    for qid, hits in zip(query_order, raw_results):
        # Map BM25's indices back to original corpus IDs
        results[qid] = {idx_to_doc_id[hit["id"]]: float(hit["score"]) for hit in hits}

    results_path = artifact_dir / "bm25_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    return {
        "results": results,
        "results_path": results_path,
        "index_path": index_dir,
        "index_size_bytes": sum(
            path.stat().st_size for path in index_dir.rglob("*") if path.is_file()
        ),
        "initialization_time_seconds": init_elapsed,
        "query_time_seconds": search_elapsed,
        "index_save_time_seconds": index_save_elapsed,
        "method": "bm25",
    }
