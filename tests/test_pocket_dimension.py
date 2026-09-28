import json
import sys
from pathlib import Path

import numpy as np
from pytest import approx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import pocket_dimension


def test_record_from_text_normalizes_and_counts_features():
    record = pocket_dimension.record_from_text("doc-1", "Quartz quartz, amber!")

    assert record == {
        "id": "doc-1",
        "features": [b"quartz", b"amber"],
        "counts": [2, 1],
    }


def test_build_cms_and_records_batches_document_frequencies(monkeypatch):
    corpus = {
        "doc-1": {"text": "amber quartz"},
        "doc-2": {"text": "amber jade"},
        "doc-3": {"text": "quartz topaz"},
    }

    class FakeCountMin:
        def __init__(self, *args, **kwargs):
            self.updates = []
            self.n_added_records = [0, 0]

        def update(self, feature_counts):
            self.updates.append(dict(feature_counts))

    monkeypatch.setattr(pocket_dimension, "CountMin", FakeCountMin)

    cms, records = pocket_dimension.build_cms_and_records(corpus, cms_batch_size=2)

    assert [record["id"] for record in records] == ["doc-1", "doc-2", "doc-3"]
    assert cms.updates == [
        {b"amber": 2, b"quartz": 1, b"jade": 1},
        {b"quartz": 1, b"topaz": 1},
    ]
    assert cms.n_added_records[1] == 3


def test_run_pipeline_maps_and_filters_search_results(tmp_path, monkeypatch):
    corpus = {
        "doc-a": {"text": "amber quartz"},
        "doc-b": {"text": "jade quartz"},
    }
    queries = {"query-a": "amber", "query-b": "jade"}
    calls = {}

    class FakeCMS:
        def n_records(self):
            return 2

    def fake_build_cms_and_records(received_corpus, cms_batch_size):
        calls["corpus"] = received_corpus
        calls["cms_batch_size"] = cms_batch_size
        return FakeCMS(), [{"id": "doc-a"}, {"id": "doc-b"}]

    class FakeVectorizer:
        def __init__(self, **kwargs):
            calls["vectorizer_kwargs"] = kwargs
            self.calls = 0

        def __call__(self, records):
            self.calls += 1
            calls[f"records_{self.calls}"] = records
            if self.calls == 1:
                return np.ones((2, 4), dtype=np.float32), ["doc-a", "doc-b"]
            return np.ones((2, 4), dtype=np.float32), ["query-a", "unknown-query"]

    class FakeFaissIndex:
        is_trained = True

        def add(self, embeddings):
            calls["indexed_embeddings"] = embeddings

        def search(self, query_embeddings, k):
            calls["searched_embeddings"] = query_embeddings
            calls["top_k"] = k
            scores = np.array([[0.9, 0.25], [0.8, 0.7]], dtype=np.float32)
            indices = np.array([[1, 0], [0, 1]], dtype=np.int64)
            return scores, indices

    def fake_index_factory(dimension, description, metric):
        calls["index_factory"] = (dimension, description, metric)
        return FakeFaissIndex()

    def fake_write_index(index, path):
        Path(path).write_bytes(b"fake-faiss-index")

    monkeypatch.setattr(
        pocket_dimension, "build_cms_and_records", fake_build_cms_and_records
    )
    monkeypatch.setattr(pocket_dimension, "BM25Vectorizer", FakeVectorizer)
    monkeypatch.setattr(pocket_dimension.faiss, "index_factory", fake_index_factory)
    monkeypatch.setattr(pocket_dimension.faiss, "write_index", fake_write_index)

    result = pocket_dimension.run_pocket_dimension_pipeline(
        corpus, queries, top_k=2, artifact_dir=tmp_path, d=4
    )

    expected_results = {"query-a": {"doc-b": approx(0.9)}, "query-b": {}}
    assert calls["corpus"] == corpus
    assert calls["cms_batch_size"] == 100_000
    assert calls["vectorizer_kwargs"]["d"] == 4
    assert calls["records_1"] == [{"id": "doc-a"}, {"id": "doc-b"}]
    assert [record["id"] for record in calls["records_2"]] == ["query-a", "query-b"]
    assert calls["top_k"] == 2
    assert calls["index_factory"][1] == "Flat"
    assert result["results"] == expected_results
    assert json.loads(result["results_path"].read_text()) == expected_results
    assert result["index_path"] == tmp_path / "faiss_index.bin"
    assert result["index_size_bytes"] == len(b"fake-faiss-index")
    assert result["method"] == "pocket_dimension"
    assert result["index_profile"] == {
        "name": "flat",
        "factory": "Flat",
        "nlist": None,
        "training_sample_count": 0,
        "search_parameters": {},
    }


def test_apply_search_parameters_targets_configured_faiss_components(monkeypatch):
    class FakeHNSW:
        efSearch = 0

    class FakeQuantizer:
        hnsw = FakeHNSW()

    class FakeIVF:
        quantizer = FakeQuantizer()
        nprobe = 0

    class FakeFaissIndex:
        hnsw = FakeHNSW()
        k_factor = 0

    faiss_index = FakeFaissIndex()
    ivf_index = FakeIVF()
    monkeypatch.setattr(
        pocket_dimension.faiss, "extract_index_ivf", lambda index: ivf_index
    )
    monkeypatch.setattr(
        pocket_dimension.faiss, "downcast_index", lambda index: index
    )

    pocket_dimension._apply_search_parameters(
        faiss_index,
        {
            "hnsw.efSearch": 30,
            "ivf.nprobe": 15,
            "ivf.quantizer.hnsw.efSearch": 30,
            "k_factor": 15,
        },
    )

    assert faiss_index.hnsw.efSearch == 30
    assert ivf_index.nprobe == 15
    assert ivf_index.quantizer.hnsw.efSearch == 30
    assert faiss_index.k_factor == 15