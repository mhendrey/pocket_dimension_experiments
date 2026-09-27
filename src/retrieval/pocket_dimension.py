"""
Pocket Dimension sparse-to-dense projection implementation

This module implements the Pocket Dimension method for projecting 
sparse BM25-style term weight vectors into dense embeddings.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from sketchnu.countmin import CountMin

from pocket_dimension.vectorizer import BM25Vectorizer


def run_pocket_dimension_pipeline(
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    top_k: int, 
    artifact_dir: Path,
    d: int = 128
) -> dict:
    """
    Run Pocket Dimension retrieval pipeline on the given corpus and queries.
    
    Args:
        corpus: Dictionary of documents in BEIR format
        queries: Dictionary of queries
        top_k: Number of results to return per query
        artifact_dir: Directory to save artifacts (results, index)
        d: Dimensionality of the dense vectors
        
    Returns:
        Dictionary containing results and performance metrics
    """
    def _tokenize(text: str) -> list[str]:
        import re
        return [token.lower() for token in re.findall(r"\b[\w'-]+\b", text or "")]

    def record_from_text(doc_id: str, text: str) -> dict:
        counts = Counter(_tokenize(text))
        return {
            "id": doc_id,
            "features": [token.encode("utf-8") for token in counts.keys()],
            "counts": list(counts.values()),
        }

    init_start = time.perf_counter()
    records = [record_from_text(doc_id, doc["text"]) for doc_id, doc in corpus.items()]

    # HLL build on the corpus only, to estimate the number of unique features for CountMin sizing
    cms = CountMin("linear", width=int(1.6 * 157_930), depth=4)
    for rec in records:
        for feature in rec["features"]:
            cms.add(feature)  # CMS is a document frequency estimator, so we add each feature once per document
        cms.n_added_records[1] += 1

    vectorizer = BM25Vectorizer(
        d=d,
        cms_file=cms,
        k1=1.2,
        b=0.75,
        temperature=1.0,
    )
    doc_embeddings, doc_ids = vectorizer(records)

    # Define, train, and add vectors to the FAISS index
    nlist = max(1, int(1.5 * np.sqrt(len(doc_embeddings))))
    if len(doc_embeddings) < 1000:
        faiss_index = faiss.index_factory(d, f"RR,RaBitQfs2,Refine(SQ8)", faiss.METRIC_INNER_PRODUCT)
    else:
        faiss_index = faiss.index_factory(
            d, f"RR,IVF{nlist}_HNSW,RaBitQfs2,Refine(SQ8)", faiss.METRIC_INNER_PRODUCT
        )
    if not faiss_index.is_trained:
        train_size = min(40 * nlist, len(doc_embeddings))
        sample_indices = np.random.choice(len(doc_embeddings), size=train_size, replace=False)
        sample_embeddings = doc_embeddings[sample_indices]
        faiss_index.train(sample_embeddings)
        try:
            ivf_index = faiss.extract_index_ivf(faiss_index)
            ivf_index.nprobe = min(10, int(0.05 * nlist))  # Search 5% of the clusters
        except Exception:
            pass  # Not an IVF index, so we skip setting nprobe
        faiss_index.k_factor = 15  # Reranking factor for Refine step

    # Add the document embeddings to the FAISS index
    faiss_index.add(doc_embeddings.astype(np.float32))
    initialization_elapsed = time.perf_counter() - init_start

    # Save the FAISS index to disk
    index_path = artifact_dir / "faiss_index.bin"
    index_write_start = time.perf_counter()
    faiss.write_index(faiss_index, str(index_path))
    index_write_elapsed = time.perf_counter() - index_write_start

    # Run queries against the FAISS index and collect results
    qid_order = list(queries.keys())
    query_results = {}
    query_start = time.perf_counter()
    for qid in qid_order:
        start_query = time.perf_counter()
        query_record = record_from_text(qid, queries[qid])
        q_emb, _ = vectorizer([query_record])
        if q_emb.size == 0:
            query_results[qid] = {}
            continue
        q_emb = q_emb.astype(np.float32)
        scores, neighbor_indices = faiss_index.search(q_emb, k=top_k)
        hits = {}
        for rank, candidate_idx in enumerate(neighbor_indices[0]):
            doc_id = str(doc_ids[candidate_idx])
            score = float(scores[0][rank])
            if doc_id and score > -1e30:
                hits[doc_id] = score
        query_results[qid] = hits
    query_elapsed = time.perf_counter() - query_start

    results_path = artifact_dir / "pocket_dimension_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(query_results, f, indent=2, sort_keys=True)

    return {
        "results": query_results,
        "results_path": results_path,
        "index_path": index_path,
        "index_size_bytes": index_path.stat().st_size,
        "initialization_time_seconds": initialization_elapsed,
        "query_time_seconds": query_elapsed,
        "index_save_time_seconds": index_write_elapsed,
        "method": "pocket_dimension",
    }
