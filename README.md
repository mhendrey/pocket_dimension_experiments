# Pocket Dimension Experiments

Comparing sparse-to-dense vector projections against traditional lexical information retrieval.

## Overview

This project evaluates **Pocket Dimension**, a method that projects sparse BM25-style term weight vectors into dense embeddings, against standard **BM25** lexical retrieval. The experiments use synthetic datasets focused on information retrieval topics and measure performance using standard IR metrics.

## Key Findings

- Pocket Dimension achieves strong retrieval effectiveness (NDCG@10: 0.809, Recall@10: 0.95)
- Faster initialization compared to traditional BM25 indexing
- Compact vector representations enable efficient approximate nearest neighbor search

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Experiments

Execute the benchmark:

```bash
python main.py
```

Results are saved in `artifacts/` directory with:
- Retrieval results (JSON)
- FAISS index (binary)
- BM25 index (directory)
- Summary statistics (summary.json)

## Technical Details

### Pocket Dimension Pipeline
1. Tokenize and count terms using BM25-style weighting
2. Apply CountMin sketch for feature hashing
3. Project sparse vectors into 128-dimensional dense space
4. Build FAISS index for efficient similarity search

### Evaluation Metrics
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall and Precision at k=1,3,5,10

## Dependencies

- beir>=2.2.0
- bm25>=0.3.11  
- faiss-cpu>=1.15.1
- pocket-dimension>=0.3.0
- numpy (via dependencies)

## License

[GNU GPL v3](LICENSE)
