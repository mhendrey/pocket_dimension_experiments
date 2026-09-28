import sys
from pathlib import Path

# Add project root to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import run_experiment


def test_run_experiment(tmp_path):
    result = run_experiment(tmp_path, testing=True)

    assert "bm25" in result
    assert "pocket_dimension" in result
    assert "artifacts_dir" in result
    assert Path(result["artifacts_dir"]) == tmp_path
    assert Path(result["bm25"]["results_path"]).is_file()
    assert Path(result["pocket_dimension"]["results_path"]).is_file()
    assert Path(result["bm25"]["index_path"]).is_dir()
    assert Path(result["pocket_dimension"]["index_path"]).is_file()
    assert (tmp_path / "summary.json").is_file()
