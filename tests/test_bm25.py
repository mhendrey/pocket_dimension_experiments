import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import bm25


def test_run_bm25_baseline_maps_hits_and_writes_artifacts(tmp_path, monkeypatch):
    corpus = {
        "doc-a": {"text": "amber quartz"},
        "doc-b": {"text": "jade quartz"},
    }
    queries = {"query-b": "jade", "query-a": "amber"}
    calls = {}

    class FakeStorage:
        def save(self, path, corpus):
            calls["saved_path"] = Path(path)
            calls["saved_corpus"] = corpus
            calls["saved_path"].mkdir(parents=True)
            (calls["saved_path"] / "index.bin").write_bytes(b"fake-index")

    class FakeRetriever:
        retriever = FakeStorage()

        def search(self, query_texts, k):
            calls["query_texts"] = query_texts
            calls["top_k"] = k
            return [
                [{"id": 1, "score": 0.875}, {"id": 0, "score": 0.25}],
                [{"id": 0, "score": 1}],
            ]

    fake_retriever = FakeRetriever()

    def fake_index(document_texts):
        calls["document_texts"] = document_texts
        return fake_retriever

    monkeypatch.setattr(bm25.BM25, "index", fake_index)

    result = bm25.run_bm25_baseline(corpus, queries, top_k=2, artifact_dir=tmp_path)

    expected_results = {
        "query-b": {"doc-b": 0.875, "doc-a": 0.25},
        "query-a": {"doc-a": 1.0},
    }
    assert calls["document_texts"] == ["amber quartz", "jade quartz"]
    assert calls["query_texts"] == ["jade", "amber"]
    assert calls["top_k"] == 2
    assert calls["saved_corpus"] == ["amber quartz", "jade quartz"]
    assert calls["saved_path"] == tmp_path / "bm25_index"
    assert result["results"] == expected_results
    assert json.loads(result["results_path"].read_text()) == expected_results
    assert result["index_path"] == tmp_path / "bm25_index"
    assert result["index_size_bytes"] == len(b"fake-index")
    assert result["method"] == "bm25"