# Running Experiments

## Python Environment

This project uses `uv` for dependency management. To run the experiments, use:

```bash
# From the project root directory
uv run python run_experiment.py
```

Alternatively, you can activate the virtual environment directly:

```bash
source .venv/bin/activate
python run_experiment.py
```

## Running Experiments

The `run_experiment.py` script provides a command-line interface for running experiments. It supports several parameters:

### Basic Usage

```bash
uv run python run_experiment.py
```

This runs the full experiment using BEIR datasets with default settings.

### Testing Mode (Small Synthetic Dataset)

To run a quick test with synthetic data (useful for verification):

```bash
uv run python run_experiment.py --testing
```

This creates a small synthetic dataset and runs the experiment quickly to verify everything works.

### Custom Parameters

You can customize the dimensionality of dense vectors and output directory:

```bash
uv run python run_experiment.py --d 256 --output_dir artifacts_custom
```

Choose a FAISS index profile with `--index-profile`:

```bash
uv run python run_experiment.py --testing --index-profile flat
uv run python run_experiment.py --index-profile hnsw32 --output_dir artifacts_hnsw32
```

- `--d`: Dimensionality of dense vectors for Pocket Dimension (default: 128, minimum: 64)
- `--output_dir`: Directory to save artifacts (default: `artifacts_full`)
- `--testing`: Run in testing mode with synthetic data
- `--index-profile`: FAISS index profile (default: `flat`)

Available profiles:
- `flat`: exact dense search; no training or search tuning
- `sq8`: scalar quantization with a training sample capped at 5,000 documents
- `rabitq-refine-sq8`: RaBitQ with SQ8 refinement and a training sample capped at 5,000 documents
- `hnsw32`: HNSW with `efSearch` set to three times the requested result count
- `ivf-hnsw-rabitq-refine-sq8`: IVF-HNSW with RaBitQ and SQ8 refinement; training is capped at `40 * nlist`

The selected profile and its resolved factory, training sample count, and search parameters are recorded in `summary.json`.

### Compare Profile Results

After running each profile, print a Markdown table with BM25 and all profile results:

```bash
uv run python summarize_profiles.py
```

The script discovers `artifacts_*/summary.json` in the current directory. Use `--artifacts-root PATH` to read summaries from another directory.

### Available BEIR Datasets

The experiments use datasets from the BEIR framework. The default dataset is automatically downloaded when needed. You can also download specific datasets using:

```bash
python download_dataset.py
```

This script downloads the 'nfcorpus' dataset (~130K documents) by default and lists other available options.

## Dataset Information

The experiments use BEIR (Benchmarking IR) datasets:
- Datasets are downloaded on-demand from BEIR's servers
- Default dataset: nfcorpus (~130,000 news articles)
- Other available datasets include scidocs, fiqa, dbpedia-entity, etc.
- Use `download_dataset.py` to download specific datasets or see available options

## Output

Results are saved in the specified output directory (default: `artifacts_full`):
- JSON files with retrieval results and metrics
- FAISS index files for Pocket Dimension vectors
- BM25 index directories
- Summary statistics in `summary.json`
- Each run creates its own artifact directory
