# Pocket Dimension Experiments

Comparing sparse-to-dense vector projections against traditional lexical information retrieval.

## Overview

This project evaluates **Pocket Dimension**, a method that projects sparse BM25-style term weight vectors into dense embeddings, against standard **BM25** lexical retrieval. The experiments use the Quora dataset (522K+ documents) and measure performance using standard IR metrics including NDCG@10, MAP@10, Recall@10, and Precision@10.

## Key Findings

The figures below are from the repository's previously recorded experiment configuration. They are not measurements of the new default `flat` profile. New runs default to Flat to provide an exact dense-search baseline; select an ANN profile with `--index-profile` to explore latency, index size, and effectiveness trade-offs. The chosen profile and resolved settings are recorded in each run's `summary.json`.

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

To compare ANN index configurations, run separate experiments with `--index-profile flat`, `sq8`, `rabitq-refine-sq8`, `hnsw32`, or `ivf-hnsw-rabitq-refine-sq8`. Keep the corpus, projection dimension, and evaluation settings the same when comparing profile results.

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

`N` is the corpus document count and `k` is the requested result count. Search settings are resolved at runtime and recorded in each `summary.json`.

- **Flat**: Exhaustive exact dense-vector search.
- **SQ8**: 8-bit scalar-quantized vectors, trained on up to 5,000 samples.
- **RaBitQ-Refine-SQ8**: RaBitQ with SQ8 refinement, trained on up to 5,000 samples.
- **HNSW32**: HNSW graph with `M=32`; `hnsw.efSearch = 3k`.
- **IVF-RaBitQ-HNSW-Refine-SQ8**: IVF with an HNSW quantizer, RaBitQ, and SQ8 refinement. `nlist = max(1, floor(1.5 * sqrt(N)))`; training uses up to `40 * nlist` samples. Search uses `nprobe = max(10, floor(0.05 * nlist))`, quantizer `efSearch = 2 * nprobe`, and `k_factor = 15`.

Flat, SQ8, and RaBitQ-Refine-SQ8 do not set profile-specific search parameters; they are brute force comparisions.



<table>
	<thead>
		<tr>
			<th rowspan="2" scope="col">Metric</th>
			<th rowspan="2" scope="col">BM25</th>
			<th colspan="5" scope="colgroup">Pocket Dimension</th>
		</tr>
		<tr>
			<th scope="col">Flat</th>
			<th scope="col">SQ8</th>
			<th scope="col">RaBitQ-Refine-SQ8</th>
			<th scope="col">HNSW32</th>
			<th scope="col">IVF-RaBitQ-HNSW-Refine-SQ8</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th scope="row">Index Size (MB)</th>
			<td>79.58</td>
			<td>255.32</td>
			<td>63.83</td>
			<td>75.86</td>
			<td>391.03</td>
			<td>81.07</td>
		</tr>
		<tr>
			<th scope="row">Initialization Time (seconds)</th>
			<td>4.422</td>
			<td>7.066</td>
			<td>7.025</td>
			<td>8.104</td>
			<td>15.858</td>
			<td>11.575</td>
		</tr>
		<tr>
			<th scope="row">Query Time (seconds)</th>
			<td>8.195</td>
			<td>13.777</td>
			<td>10.198</td>
			<td>1.792</td>
			<td>0.429</td>
			<td>0.529</td>
		</tr>
		<tr>
			<th scope="row">MAP@1</th>
			<td>0.62555</td>
			<td>0.52339</td>
			<td>0.52324</td>
			<td>0.52170</td>
			<td>0.43158</td>
			<td>0.48286</td>
		</tr>
		<tr>
			<th scope="row">MAP@3</th>
			<td>0.72487</td>
			<td>0.60587</td>
			<td>0.60570</td>
			<td>0.60152</td>
			<td>0.49657</td>
			<td>0.55111</td>
		</tr>
		<tr>
			<th scope="row">MAP@5</th>
			<td>0.74329</td>
			<td>0.62276</td>
			<td>0.62256</td>
			<td>0.61606</td>
			<td>0.50985</td>
			<td>0.56284</td>
		</tr>
		<tr>
			<th scope="row">MAP@10</th>
			<td>0.75544</td>
			<td>0.63353</td>
			<td>0.63347</td>
			<td>0.62387</td>
			<td>0.51836</td>
			<td>0.57071</td>
		</tr>
		<tr>
			<th scope="row">NDCG@1</th>
			<td>0.71630</td>
			<td>0.60180</td>
			<td>0.60160</td>
			<td>0.60000</td>
			<td>0.50190</td>
			<td>0.55760</td>
		</tr>
		<tr>
			<th scope="row">NDCG@3</th>
			<td>0.76589</td>
			<td>0.64378</td>
			<td>0.64364</td>
			<td>0.63851</td>
			<td>0.52918</td>
			<td>0.58567</td>
		</tr>
		<tr>
			<th scope="row">NDCG@5</th>
			<td>0.78611</td>
			<td>0.66418</td>
			<td>0.66394</td>
			<td>0.65496</td>
			<td>0.54421</td>
			<td>0.59842</td>
		</tr>
		<tr>
			<th scope="row">NDCG@10</th>
			<td>0.80450</td>
			<td>0.68323</td>
			<td>0.68325</td>
			<td>0.66762</td>
			<td>0.55853</td>
			<td>0.61183</td>
		</tr>
		<tr>
			<th scope="row">P@1</th>
			<td>0.71630</td>
			<td>0.60180</td>
			<td>0.60160</td>
			<td>0.60000</td>
			<td>0.50190</td>
			<td>0.55760</td>
		</tr>
		<tr>
			<th scope="row">P@3</th>
			<td>0.33053</td>
			<td>0.27813</td>
			<td>0.27807</td>
			<td>0.27527</td>
			<td>0.23070</td>
			<td>0.25163</td>
		</tr>
		<tr>
			<th scope="row">P@5</th>
			<td>0.21954</td>
			<td>0.18604</td>
			<td>0.18590</td>
			<td>0.18162</td>
			<td>0.15400</td>
			<td>0.16514</td>
		</tr>
		<tr>
			<th scope="row">P@10</th>
			<td>0.12181</td>
			<td>0.10374</td>
			<td>0.10380</td>
			<td>0.09856</td>
			<td>0.08580</td>
			<td>0.09076</td>
		</tr>
		<tr>
			<th scope="row">Recall@1</th>
			<td>0.62555</td>
			<td>0.52339</td>
			<td>0.52324</td>
			<td>0.52170</td>
			<td>0.43158</td>
			<td>0.48286</td>
		</tr>
		<tr>
			<th scope="row">Recall@3</th>
			<td>0.79462</td>
			<td>0.66975</td>
			<td>0.66968</td>
			<td>0.66219</td>
			<td>0.54517</td>
			<td>0.60362</td>
		</tr>
		<tr>
			<th scope="row">Recall@5</th>
			<td>0.84859</td>
			<td>0.72471</td>
			<td>0.72435</td>
			<td>0.70843</td>
			<td>0.58744</td>
			<td>0.64055</td>
		</tr>
		<tr>
			<th scope="row">Recall@10</th>
			<td>0.90148</td>
			<td>0.78241</td>
			<td>0.78277</td>
			<td>0.74756</td>
			<td>0.63121</td>
			<td>0.68182</td>
		</tr>
	</tbody>
</table>

## Dependencies

- beir>=2.2.0
- bm25>=0.3.11  
- faiss-cpu>=1.15.1
- pocket-dimension>=0.3.0
- numpy (via dependencies)

## License

[GNU GPL v3](LICENSE)
