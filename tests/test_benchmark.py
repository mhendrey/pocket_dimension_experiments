from pathlib import Path

from main import run_experiment


def test_run_experiment(tmp_path):
    result = run_experiment(tmp_path)

    assert "bm25" in result
    assert "pocket_dimension" in result
    assert "artifacts_dir" in result
    assert result["artifacts_dir"].exists()
    assert result["bm25"]["results_path"].exists()
    assert result["pocket_dimension"]["results_path"].exists()
    assert result["bm25"]["index_path"].exists()
    assert result["pocket_dimension"]["index_path"].exists()
