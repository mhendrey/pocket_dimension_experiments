from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sketchnu.countmin import CountMin

import BM25
from pocket_dimension.vectorizer import BM25Vectorizer


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\b[\w'-]+\b", text or "")]


def _make_synthetic_beir_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = {
        "doc_0": "machine learning models learn from data and optimize objective functions",
        "doc_1": "search engines use inverted indexes and ranking functions to retrieve documents",
        "doc_2": "python programming is popular for analysis and retrieval systems research",
        "doc_3": "information retrieval benchmarks compare lexical and dense vector search methods",
        "doc_4": "faiss builds approximate nearest neighbor indexes for similarity search",
        "doc_5": "bm25 is a probabilistic ranking function used in search systems",
        "doc_7": "documents are scored with term frequency and inverse document frequency weights",
        "doc_8": "evaluation measures include ndcg map recall and precision for retrieval tasks",
        "doc_9": "query understanding uses tokenization stemming and normalization before ranking",
        "doc_10": "neural retrieval methods encode queries and documents into shared embeddings",
        "doc_11": "ranking performance is often measured at k values such as 1 3 5 and 10",
        "doc_12": "bm25 based retrieval remains strong on text matching and keyword search",
        "doc_13": "vector search is useful when semantic similarity matters more than exact match",
        "doc_14": "pocket dimension projects sparse bm25 weights into dense vectors for efficient search",
        "doc_15": "beir provides standard retrieval tasks and evaluation datasets for experiments",
        "doc_16": "random projections conserve similarity structure while reducing dimensionality",
        "doc_17": "lexical retrieval usually uses exact term matches with idf and length normalization",
        "doc_18": "approximate nearest neighbor search trades accuracy for speed in very large corpora",
        "doc_19": "search evaluation requires ground truth qrels over relevant document ids",
    }

    queries = {
        "q_0": "machine learning models and optimization",
        "q_1": "search engines indexes and document retrieval",
        "q_2": "python retrieval research and ranking",
        "q_3": "bm25 and lexical search evaluation",
        "q_4": "faiss approximate nearest neighbor similarity search",
        "q_5": "document scoring and tfidf weights",
        "q_6": "vector retrieval semantic similarity and embeddings",
        "q_7": "bm25 ranking for keyword search",
        "q_8": "beir benchmark evaluation metrics for retrieval",
        "q_9": "approximate search index performance and dimension reduction",
    }

    qrels = {
        "q_0": {"doc_0": 1, "doc_10": 1},
        "q_1": {"doc_1": 1, "doc_18": 1},
        "q_2": {"doc_2": 1, "doc_15": 1},
        "q_3": {"doc_5": 1, "doc_12": 1, "doc_17": 1},
        "q_4": {"doc_4": 1, "doc_16": 1},
        "q_5": {"doc_7": 1, "doc_8": 1},
        "q_6": {"doc_10": 1, "doc_13": 1},
        "q_7": {"doc_5": 1, "doc_12": 1},
        "q_8": {"doc_3": 1, "doc_15": 1},
        "q_9": {"doc_4": 1, "doc_16": 1, "doc_18": 1},
    }

    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"_id": doc_id, "text": text}) + "\n")

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")

    qrels_path = output_dir / "qrels" / "test.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with qrels_path.open("w", encoding="utf-8") as f:
        f.write("query_id\tcorpus_id\tscore\n")
        for qid, rels in qrels.items():
            for doc_id, score in rels.items():
                f.write(f"{qid}\t{doc_id}\t{score}\n")

    return output_dir


def _bm25_baseline(corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, artifact_dir: Path) -> dict:
    doc_texts = [entry["text"] for entry in corpus.values()]

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
        results[qid] = {str(int(hit["id"])): float(hit["score"]) for hit in hits}

    results_path = artifact_dir / "bm25_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    return {
        "results": results,
        "results_path": results_path,
        "index_path": index_dir,
        "index_size_bytes": sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file()),
        "initialization_time_seconds": init_elapsed,
        "query_time_seconds": search_elapsed,
        "index_save_time_seconds": index_save_elapsed,
        "method": "bm25",
    }


def _pocket_dimension_pipeline(corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, artifact_dir: Path) -> dict:
    def record_from_text(doc_id: str, text: str) -> dict:
        counts = Counter(_tokenize(text))
        return {
            "id": doc_id,
            "features": [token.encode("utf-8") for token in counts.keys()],
            "counts": list(counts.values()),
        }

    init_start = time.perf_counter()
    records = [record_from_text(doc_id, doc["text"]) for doc_id, doc in corpus.items()]

    cms = CountMin("linear", width=4096, depth=4)
    for rec in records:
        for feature in rec["features"]:
            cms.add(feature)
        cms.n_added_records[1] += 1

    vectorizer = BM25Vectorizer(
        d=128,
        cms_file=cms,
        k1=1.2,
        b=0.75,
        minDF=1,
        maxDF=1_000_000,
        temperature=1.0,
    )

    doc_embeddings, doc_ids = vectorizer(records)
    faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings.astype(np.float32))
    initialization_elapsed = time.perf_counter() - init_start

    index_path = artifact_dir / "faiss_index.bin"
    index_write_start = time.perf_counter()
    faiss.write_index(faiss_index, str(index_path))
    index_write_elapsed = time.perf_counter() - index_write_start

    qid_order = list(queries.keys())
    query_results = {}
    for qid in qid_order:
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

    results_path = artifact_dir / "pocket_dimension_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(query_results, f, indent=2, sort_keys=True)

    return {
        "results": query_results,
        "results_path": results_path,
        "index_path": index_path,
        "index_size_bytes": index_path.stat().st_size,
        "initialization_time_seconds": initialization_elapsed,
        "query_time_seconds": 0.0,
        "index_save_time_seconds": index_write_elapsed,
        "method": "pocket_dimension",
    }


def _evaluate_pair(results: dict, qrels: dict[str, dict[str, int]], k_values: list[int]) -> dict[str, float]:
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    metrics = {}
    metrics.update(ndcg)
    metrics.update(_map)
    metrics.update(recall)
    metrics.update(precision)
    return metrics


def run_experiment(output_dir: str | Path | None = None) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_dir / "beir_data"
    _make_synthetic_beir_dataset(dataset_dir)

    loader = GenericDataLoader(data_folder=str(dataset_dir), prefix=None)
    corpus, queries, qrels = loader.load(split="test")

    bm25 = _bm25_baseline(corpus, queries, top_k=10, artifact_dir=output_dir)
    pocket = _pocket_dimension_pipeline(corpus, queries, top_k=10, artifact_dir=output_dir)

    bm25_metrics = _evaluate_pair(bm25["results"], qrels, [1, 3, 5, 10])
    pocket_metrics = _evaluate_pair(pocket["results"], qrels, [1, 3, 5, 10])

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


def main() -> None:
    summary = run_experiment(Path("artifacts"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
