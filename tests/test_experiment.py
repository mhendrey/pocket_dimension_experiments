import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import experiment


def test_run_experiment_orchestrates_pipeline_and_writes_summary(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "artifacts"
    dataset_parts = (object(), object(), object())
    corpus = {"doc-1": {"text": "sample document"}}
    queries = {"query-1": "sample query"}
    qrels = {"query-1": {"doc-1": 1}}
    bm25_results = {"query-1": {"doc-1": 0.8}}
    pocket_results = {"query-1": {"doc-1": 0.7}}
    bm25_output = {
        "results": bm25_results,
        "results_path": output_dir / "bm25_results.json",
        "index_path": output_dir / "bm25_index",
        "index_size_bytes": 100,
        "initialization_time_seconds": 0.1,
        "query_time_seconds": 0.2,
        "index_save_time_seconds": 0.3,
    }
    pocket_output = {
        "results": pocket_results,
        "results_path": output_dir / "pocket_dimension_results.json",
        "index_path": output_dir / "faiss_index.bin",
        "index_size_bytes": 200,
        "initialization_time_seconds": 0.4,
        "query_time_seconds": 0.5,
        "index_save_time_seconds": 0.6,
        "index_profile": {
            "name": "flat",
            "factory": "Flat",
            "nlist": None,
            "training_sample_count": 0,
            "search_parameters": {},
        },
    }
    calls = {}
    metric_calls = []

    monkeypatch.setattr(
        experiment, "make_synthetic_beir_dataset", lambda: dataset_parts
    )

    def unexpected_quora_load():
        raise AssertionError("testing=True should use the synthetic dataset")

    monkeypatch.setattr(experiment, "load_quora_dataset", unexpected_quora_load)

    def fake_convert_to_beir(*datasets):
        calls["datasets"] = datasets
        return corpus, queries, qrels

    def fake_bm25(received_corpus, received_queries, top_k, artifact_dir):
        calls["bm25"] = (received_corpus, received_queries, top_k, artifact_dir)
        return bm25_output

    def fake_pocket_dimension(
        received_corpus, received_queries, top_k, artifact_dir, d, index_profile
    ):
        calls["pocket_dimension"] = (
            received_corpus,
            received_queries,
            top_k,
            artifact_dir,
            d,
            index_profile,
        )
        return pocket_output

    def fake_evaluate(results, received_qrels, k_values):
        metric_calls.append((results, received_qrels, k_values))
        return {"NDCG@10": 0.75 if results is bm25_results else 0.5}

    monkeypatch.setattr(experiment, "convert_to_beir", fake_convert_to_beir)
    monkeypatch.setattr(experiment, "run_bm25_baseline", fake_bm25)
    monkeypatch.setattr(
        experiment, "run_pocket_dimension_pipeline", fake_pocket_dimension
    )
    monkeypatch.setattr(experiment, "evaluate_retrieval", fake_evaluate)

    summary = experiment.run_experiment(output_dir, d=32, testing=True)

    assert calls["datasets"] == dataset_parts
    assert calls["bm25"] == (corpus, queries, 10, output_dir)
    assert calls["pocket_dimension"] == (corpus, queries, 10, output_dir, 32, "flat")
    assert metric_calls == [
        (bm25_results, qrels, [1, 3, 5, 10]),
        (pocket_results, qrels, [1, 3, 5, 10]),
    ]
    assert summary["artifacts_dir"] == str(output_dir)
    assert summary["bm25"]["metrics"] == {"NDCG@10": 0.75}
    assert summary["pocket_dimension"]["metrics"] == {"NDCG@10": 0.5}
    assert summary["pocket_dimension"]["index_profile"] == pocket_output["index_profile"]
    assert json.loads((output_dir / "summary.json").read_text()) == summary
