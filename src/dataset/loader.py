"""
Dataset loading utilities

This module provides functions for loading datasets from various sources and converting them to BeIR format for evaluation.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset


def convert_to_beir(
    corpus_ds: Dataset, queries_ds: Dataset, qrels_ds: Dataset
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[int]]]:
    """
    Convert Hugging Face Datasets to BEIR format.

    Parameters
    ----------
    corpus_ds : Dataset
        Dataset containing corpus documents with '_id', 'title', and 'text' fields.
    queries_ds : Dataset
        Dataset containing query documents with '_id' and 'text' fields.
    qrels_ds : Dataset
        Dataset containing query relevance information with 'query-id',
        'corpus-id', and 'score' fields.

    Returns
    -------
    tuple[dict[str, dict], dict[str, str], dict[str, dict[int]]]
        A tuple of three dictionaries:
        - corpus: Mapping from document IDs to {'title': str, 'text': str}
        - queries: Mapping from query IDs to query text
        - qrels: Mapping from query ID to {doc ID: score}
    """
    # 1. Transform Corpus: List[dict] -> Dict[id, {title, text}]
    corpus = {
        row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in corpus_ds
    }

    # 2. Transform Queries: List[dict] -> Dict[id, text]
    queries = {row["_id"]: row.get("text", "") for row in queries_ds}

    # 3. Transform Qrels: List[dict] -> Dict[query_id, Dict[doc_id, score]]
    qrels = {}
    for row in qrels_ds:
        q_id = str(row["query-id"])
        d_id = str(row["corpus-id"])
        score = int(row["score"])

        if q_id not in qrels:
            qrels[q_id] = {}
        qrels[q_id][d_id] = score

    return corpus, queries, qrels


def make_synthetic_beir_dataset() -> tuple[Dataset, Dataset, Dataset]:
    """
    Create a synthetic BEIR dataset for testing and development.

    Returns:
        Tuple of three datasets: corpus, queries, and qrels
    """

    corpus = [
        {
            "_id": "doc_0",
            "title": "",
            "text": "machine learning models learn from data and optimize objective functions",
        },
        {
            "_id": "doc_1",
            "title": "",
            "text": "search engines use inverted indexes and ranking functions to retrieve documents",
        },
        {
            "_id": "doc_3",
            "title": "",
            "text": "information retrieval benchmarks compare lexical and dense vector search methods",
        },
        {
            "_id": "doc_4",
            "title": "",
            "text": "faiss builds approximate nearest neighbor indexes for similarity search",
        },
        {
            "_id": "doc_5",
            "title": "",
            "text": "bm25 is a probabilistic ranking function used in search systems",
        },
        {
            "_id": "doc_7",
            "title": "",
            "text": "documents are scored with term frequency and inverse document frequency weights",
        },
        {
            "_id": "doc_8",
            "title": "",
            "text": "evaluation measures include ndcg map recall and precision for retrieval tasks",
        },
        {
            "_id": "doc_9",
            "title": "",
            "text": "query understanding uses tokenization stemming and normalization before ranking",
        },
        {
            "_id": "doc_10",
            "title": "",
            "text": "neural retrieval methods encode queries and documents into shared embeddings",
        },
        {
            "_id": "doc_11",
            "title": "",
            "text": "ranking performance is often measured at k values such as 1 3 5 and 10",
        },
        {
            "_id": "doc_12",
            "title": "",
            "text": "bm25 based retrieval remains strong on text matching and keyword search",
        },
        {
            "_id": "doc_13",
            "title": "",
            "text": "vector search is useful when semantic similarity matters more than exact match",
        },
        {
            "_id": "doc_14",
            "title": "",
            "text": "pocket dimension projects sparse bm25 weights into dense vectors for efficient search",
        },
        {
            "_id": "doc_15",
            "title": "",
            "text": "beir provides standard retrieval tasks and evaluation datasets for experiments",
        },
        {
            "_id": "doc_16",
            "title": "",
            "text": "random projections conserve similarity structure while reducing dimensionality",
        },
        {
            "_id": "doc_17",
            "title": "",
            "text": "lexical retrieval usually uses exact term matches with idf and length normalization",
        },
        {
            "_id": "doc_18",
            "title": "",
            "text": "approximate nearest neighbor search trades accuracy for speed in very large corpora",
        },
        {
            "_id": "doc_19",
            "title": "",
            "text": "search evaluation requires ground truth qrels over relevant document ids",
        },
    ]
    queries = [
        {"_id": "q_0", "title": "", "text": "machine learning models and optimization"},
        {
            "_id": "q_1",
            "title": "",
            "text": "search engines indexes and document retrieval",
        },
        {"_id": "q_2", "title": "", "text": "python retrieval research and ranking"},
        {"_id": "q_3", "title": "", "text": "bm25 and lexical search evaluation"},
        {
            "_id": "q_4",
            "title": "",
            "text": "faiss approximate nearest neighbor similarity search",
        },
        {"_id": "q_5", "title": "", "text": "document scoring and tfidf weights"},
        {
            "_id": "q_6",
            "title": "",
            "text": "vector retrieval semantic similarity and embeddings",
        },
        {"_id": "q_7", "title": "", "text": "bm25 ranking for keyword search"},
        {
            "_id": "q_8",
            "title": "",
            "text": "beir benchmark evaluation metrics for retrieval",
        },
        {
            "_id": "q_9",
            "title": "",
            "text": "approximate search index performance and dimension reduction",
        },
    ]

    qrels = [
        {"query-id": "q_0", "corpus-id": "doc_0", "score": 1},
        {"query-id": "q_0", "corpus-id": "doc_10", "score": 0},
        {"query-id": "q_1", "corpus-id": "doc_1", "score": 1},
        {"query-id": "q_1", "corpus-id": "doc_18", "score": 1},
        {"query-id": "q_2", "corpus-id": "doc_2", "score": 1},
        {"query-id": "q_2", "corpus-id": "doc_15", "score": 1},
        {"query-id": "q_3", "corpus-id": "doc_5", "score": 1},
        {"query-id": "q_3", "corpus-id": "doc_12", "score": 1},
        {"query-id": "q_3", "corpus-id": "doc_17", "score": 1},
        {"query-id": "q_4", "corpus-id": "doc_4", "score": 1},
        {"query-id": "q_4", "corpus-id": "doc_16", "score": 1},
        {"query-id": "q_5", "corpus-id": "doc_7", "score": 1},
        {"query-id": "q_5", "corpus-id": "doc_8", "score": 1},
        {"query-id": "q_6", "corpus-id": "doc_10", "score": 1},
        {"query-id": "q_6", "corpus-id": "doc_13", "score": 1},
        {"query-id": "q_7", "corpus-id": "doc_5", "score": 1},
        {"query-id": "q_7", "corpus-id": "doc_12", "score": 1},
        {"query-id": "q_8", "corpus-id": "doc_3", "score": 1},
        {"query-id": "q_8", "corpus-id": "doc_15", "score": 1},
        {"query-id": "q_9", "corpus-id": "doc_4", "score": 1},
        {"query-id": "q_9", "corpus-id": "doc_16", "score": 1},
    ]

    return (
        Dataset.from_list(corpus),
        Dataset.from_list(queries),
        Dataset.from_list(qrels),
    )


def load_quora_dataset() -> tuple[Dataset, Dataset, Dataset]:
    """
    Load the Quora dataset from Hugging Face Datasets.

    Returns:
        Tuple of (corpus, queries, qrels) dictionaries in BEIR format
    """
    print("Loading Quora dataset from Hugging Face...")

    # Configure dataset loading to use cache and avoid unnecessary downloads
    from datasets import DownloadConfig
    
    download_config = DownloadConfig(
        cache_dir=str(Path.home() / ".cache" / "huggingface" / "datasets"),
        force_download=False,
        resume_download=True,
    )

    # Load the pre-built BEIR Quora dataset with explicit configuration
    corpus_ds = load_dataset("BeIR/quora", "corpus", split="corpus", download_config=download_config)
    queries_ds = load_dataset("BeIR/quora", "queries", split="queries", download_config=download_config)
    qrels_ds = load_dataset("BeIR/quora-qrels", split="test", download_config=download_config)

    print(
        f"Loaded {len(corpus_ds)} documents, {len(queries_ds)} queries, {len(qrels_ds)} query-doc relations"
    )

    return corpus_ds, queries_ds, qrels_ds
