# Running Experiments

## Python Environment

This project uses `uv` for dependency management. To run the experiments, use:

```bash
# From the project root directory
uv run python main.py
```

Alternatively, you can activate the virtual environment directly:

```bash
source .venv/bin/activate
python main.py
```

## Running Experiments

The `main.py` script supports different experiment sizes via the `sample_size` parameter:

### Small Test Run (1000 documents)
```bash
uv run python main.py
```

This runs a quick test with 1000 documents to verify everything works.

### Full Experiment (500K+ documents)
To run the full experiment on all available Quora data (~500,000 documents), uncomment the full experiment section in `main.py`:

```python
def main() -> None:
    # Test with a small sample first
    print("Running test with 1000 documents...")
    summary = run_experiment(Path("artifacts_test"), sample_size=1000)
    print(json.dumps(summary, indent=2, sort_keys=True))
    
    # Uncomment to run full experiment on all 500K documents
    # print("\nRunning full experiment with all 500K documents...")
    # summary = run_experiment(Path("artifacts_full"))
    # print(json.dumps(summary, indent=2, sort_keys=True))
```

Then run:
```bash
uv run python main.py
```

## Dataset Information

The experiments use the Quora dataset from Hugging Face Datasets (`BeIR/quora`):
- Full dataset: ~500,000 documents and queries
- The dataset is loaded on-demand from Hugging Face Hub
- No local download required (streamed directly)

## Output

Results are saved in the `artifacts_*` directories:
- `artifacts_test/`: Small test run results
- `artifacts_full/`: Full experiment results (when uncommented)
- Each directory contains JSON files with metrics and index files
