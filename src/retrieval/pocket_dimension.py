"""
Pocket Dimension sparse-to-dense projection implementation

This module implements the Pocket Dimension method for projecting 
sparse BM25-style term weight vectors into dense embeddings.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from BM25 import Tokenizer
from sketchnu.countmin import CountMin
from sklearn.feature_extraction.text import HashingVectorizer

from pocket_dimension.vectorizer import BM25Vectorizer
from src.retrieval.index_profile import get_index_profile

STOPWORDS = set(Tokenizer().stopwords)  # Matches BM25's default stopwords

TEXT_ANALYZER = HashingVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    analyzer="word",
    stop_words=STOPWORDS,
).build_analyzer()


def run_pocket_dimension_pipeline(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    top_k: int,
    artifact_dir: Path,
    d: int = 128,
    index_profile: str = "flat",
) -> dict:
    """
    Run Pocket Dimension retrieval pipeline on the given corpus and queries.

    Args:
        corpus: Dictionary of documents in BEIR format
        queries: Dictionary of queries
        top_k: Number of results to return per query
        artifact_dir: Directory to save artifacts (results, index)
        d: Dimensionality of the dense vectors
        index_profile: Named FAISS index profile to use

    Returns:
        Dictionary containing results and performance metrics
    """
    init_start = time.perf_counter()
    cms, records = build_cms_and_records(corpus, cms_batch_size=100_000)
    print(
        f"Built CountMin sketch with {cms.n_records():,} in {time.perf_counter() - init_start:.2f} seconds."
    )
    vectorizer = BM25Vectorizer(
        d=d,
        cms_file=cms,
        k1=1.2,
        b=0.75,
        temperature=1.0,
    )
    start_vectorization = time.perf_counter()
    doc_embeddings, doc_ids = vectorizer(records)
    print(
        f"Vectorized documents in {time.perf_counter() - start_vectorization:.2f} seconds."
    )

    # Define, train, and add vectors to the FAISS index
    profile = get_index_profile(index_profile)
    resolved_profile = profile.resolve(len(doc_embeddings), top_k)
    faiss_index = faiss.index_factory(
        d, resolved_profile.factory_description, faiss.METRIC_INNER_PRODUCT
    )
    training_sample_count = 0
    if not faiss_index.is_trained:
        train_size = resolved_profile.training_sample_size
        if train_size is None:
            raise ValueError(
                f"Index profile {profile.key!r} requires training but has no "
                "training sample policy."
            )
        sample_indices = np.random.choice(
            len(doc_embeddings), size=train_size, replace=False
        )
        sample_embeddings = doc_embeddings[sample_indices]
        faiss_index.train(sample_embeddings)
        training_sample_count = train_size

    # Add the document embeddings to the FAISS index
    faiss_index.add(doc_embeddings)
    _apply_search_parameters(faiss_index, resolved_profile.search_parameters)
    initialization_elapsed = time.perf_counter() - init_start

    # Save the FAISS index to disk
    index_path = artifact_dir / "faiss_index.bin"
    index_write_start = time.perf_counter()
    faiss.write_index(faiss_index, str(index_path))
    index_write_elapsed = time.perf_counter() - index_write_start

    # ==========================================
    # OPTIMIZED BATCH QUERY PROCESSING
    # ==========================================
    qid_order = list(queries.keys())

    # 1. Prep all text records
    query_start = time.perf_counter()
    all_query_records = [record_from_text(qid, queries[qid]) for qid in qid_order]

    print("Vectorizing valid queries in batch...")
    # 2. Batch vectorization (returns only the valid embeddings and matching qids)
    all_q_embs, all_q_ids = vectorizer(all_query_records)

    # Initialize empty dictionaries for all query IDs upfront (handles empty/filtered strings)
    query_results = {qid: {} for qid in qid_order}

    # 3. Only query FAISS if we actually have valid embeddings
    if all_q_embs.size > 0:
        # Query FAISS in one highly-optimized batch operation
        all_scores, all_neighbor_indices = faiss_index.search(all_q_embs, k=top_k)

        # 4. Map results back using the returned valid ID tracker
        for idx, qid in enumerate(all_q_ids):
            # Safe string lookup/conversion to match your original qid types
            str_qid = str(qid)
            if str_qid not in query_results:
                continue

            hits = {}
            for rank, candidate_idx in enumerate(all_neighbor_indices[idx]):
                doc_id = str(doc_ids[candidate_idx])
                score = float(all_scores[idx][rank])
                if doc_id and score > 0.25:
                    hits[doc_id] = score
            query_results[str_qid] = hits
    else:
        print("No valid query embeddings generated.")
    query_elapsed = time.perf_counter() - query_start

    results_path = artifact_dir / "pocket_dimension_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(query_results, f, indent=2, sort_keys=True)

    return {
        "results": query_results,
        "results_path": results_path,
        "index_path": index_path,
        "index_size_bytes": index_path.stat().st_size,
        "index_profile": resolved_profile.metadata(training_sample_count),
        "initialization_time_seconds": initialization_elapsed,
        "query_time_seconds": query_elapsed,
        "index_save_time_seconds": index_write_elapsed,
        "method": "pocket_dimension",
    }


def _apply_search_parameters(faiss_index, search_parameters: dict[str, int]) -> None:
    if "hnsw.efSearch" in search_parameters:
        faiss_index.hnsw.efSearch = search_parameters["hnsw.efSearch"]

    if "ivf.nprobe" in search_parameters:
        ivf_index = faiss.extract_index_ivf(faiss_index)
        ivf_index.nprobe = search_parameters["ivf.nprobe"]

    if "ivf.quantizer.hnsw.efSearch" in search_parameters:
        ivf_index = faiss.extract_index_ivf(faiss_index)
        quantizer = faiss.downcast_index(ivf_index.quantizer)
        quantizer.hnsw.efSearch = search_parameters[
            "ivf.quantizer.hnsw.efSearch"
        ]

    if "k_factor" in search_parameters:
        faiss_index.k_factor = search_parameters["k_factor"]


def record_from_text(doc_id: str, text: str) -> dict:
    counts = Counter(TEXT_ANALYZER(text))
    return {
        "id": doc_id,
        "features": [token.encode("utf-8") for token in counts.keys()],
        "counts": list(counts.values()),
    }


def build_cms_and_records(
    corpus: dict[str, dict[str, str]],
    cms_batch_size: int = 10_000,
) -> tuple[CountMin, list[dict]]:
    """
    Build CMS document frequency estimator and records from corpus.

    Optimizes CMS building by batching feature aggregation across multiple
    documents before calling cms.update(). Each feature contributes at most
    once per document (document frequency), so we aggregate across documents.

    Args:
        corpus: Dictionary of documents in BEIR format
        cms_batch_size: Number of documents to aggregate before flushing to CMS

    Returns:
        Tuple of (CountMin object, list of records)
    """
    # Build all records first (needed for BM25Vectorizer later)
    records = [record_from_text(doc_id, doc["text"]) for doc_id, doc in corpus.items()]

    # Initialize CMS
    cms = CountMin("linear", width=int(1.6 * 157_930), depth=4)

    # Batch aggregate features across documents before updating CMS
    feature_counts = Counter()
    batch_record_count = 0

    for rec in records:
        if not rec["features"]:
            continue  # Skip empty records

        # Each feature from this record contributes 1 to document frequency
        for feature in rec["features"]:
            feature_counts[feature] += 1

        batch_record_count += 1

        # Flush batch when it reaches size limit
        if batch_record_count >= cms_batch_size:
            cms.update(feature_counts)
            cms.n_added_records[1] += batch_record_count
            feature_counts.clear()
            batch_record_count = 0

    # Flush final batch
    if batch_record_count > 0:
        cms.update(feature_counts)
        cms.n_added_records[1] += batch_record_count

    return cms, records
