from pathlib import Path
import sys

from datasets import DownloadConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import loader


def test_load_quora_dataset_uses_expected_hugging_face_splits(monkeypatch):
    expected_datasets = (["doc"], ["query"], ["qrel"])
    calls = []

    def fake_load_dataset(dataset_id, *args, **kwargs):
        calls.append((dataset_id, args, kwargs))
        return expected_datasets[len(calls) - 1]

    monkeypatch.setattr(loader, "load_dataset", fake_load_dataset)

    result = loader.load_quora_dataset()

    assert result == expected_datasets
    assert [
        (dataset_id, args, kwargs["split"])
        for dataset_id, args, kwargs in calls
    ] == [
        ("BeIR/quora", ("corpus",), "corpus"),
        ("BeIR/quora", ("queries",), "queries"),
        ("BeIR/quora-qrels", (), "test"),
    ]

    download_configs = [kwargs["download_config"] for _, _, kwargs in calls]
    assert all(isinstance(config, DownloadConfig) for config in download_configs)
    assert download_configs[0] is download_configs[1] is download_configs[2]
    assert download_configs[0].cache_dir == str(
        Path.home() / ".cache" / "huggingface" / "datasets"
    )