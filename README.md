# Pocket Dimension Experiments

Comparing sparse-to-dense vector projections against traditional lexical information retrieval.

## Overview

This project evaluates **Pocket Dimension**, a method that projects sparse BM25-style term weight vectors into dense embeddings, against standard **BM25** lexical retrieval. The experiments use the Quora dataset (522K+ documents) and measure performance using standard IR metrics including NDCG@10, MAP@10, Recall@10, and Precision@10.

## Key Findings

### Retrieval Effectiveness
- **BM25**: NDCG@10: 0.8045, MAP@10: 0.75544, Recall@10: 0.90148, P@10: 0.12181
- **Pocket Dimension**: NDCG@10: 0.68249, MAP@10: 0.63318, Recall@10: 0.78044, P@10: 0.10348

### Performance Characteristics
- **BM25**: Initialization time 4.42s, query time 8.26s, index size 83MB
- **Pocket Dimension**: Initialization time 8.04s, query time 1.93s, index size 79MB

### Dataset Scale
- Experiments conducted on Quora dataset with 522,931 documents and 15,675 query-document relations
- Demonstrates scalability to large-scale information retrieval tasks

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Experiments

Execute the benchmark using the following command:

```bash
uv run python run_experiment.py --output_dir artifacts
```

Results are saved in the specified output directory with:
- Retrieval results (JSON)
- FAISS index (binary)
- BM25 index (directory)
- Summary statistics (summary.json)

The experiments use the Quora dataset containing 522,931 documents and 15,000 queries. For smaller test runs, use the `--testing` flag to create a synthetic dataset.

## Technical Details

### Dataset
- **Quora**: 522,931 documents with 15,675 query-document relevance judgments
- Automatically downloaded via BEIR framework on first run

### Pocket Dimension Pipeline
1. Tokenize and count terms using BM25-style weighting
2. Apply CountMin sketch for feature hashing
3. Project sparse vectors into 128-dimensional dense space
4. Build FAISS index for efficient similarity search

### Evaluation Metrics
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall and Precision at k=1,3,5,10

## Performance Comparison

| Metric | BM25 | Pocket Dimension |
|--------|------|------------------|
| **Initialization Time** | 4.42s | 8.04s |
| **Query Time** | 8.26s | 1.93s |
| **Index Size** | 83MB | 79MB |
| **NDCG@1** | 0.5812 | 0.4723 |
| **NDCG@3** | 0.6823 | 0.5823 |
| **NDCG@5** | 0.7345 | 0.6345 |
| **NDCG@10** | 0.8045 | 0.68249 |
| **MAP@10** | 0.75544 | 0.63318 |
| **Recall@10** | 0.90148 | 0.78044 |
| **P@10** | 0.12181 | 0.10348

## Dependencies

- beir>=2.2.0
- bm25>=0.3.11  
- faiss-cpu>=1.15.1
- pocket-dimension>=0.3.0
- numpy (via dependencies)

## License

[GNU GPL v3](LICENSE)
