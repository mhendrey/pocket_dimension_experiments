import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_experiment import parse_args


def test_cli_defaults_to_flat_profile(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment.py"])

    args = parse_args()

    assert args.index_profile == "flat"


@pytest.mark.parametrize(
    "profile",
    [
        "flat",
        "sq8",
        "rabitq-refine-sq8",
        "hnsw32",
        "ivf-hnsw-rabitq-refine-sq8",
    ],
)
def test_cli_accepts_named_index_profiles(monkeypatch, profile):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--index-profile", profile],
    )

    assert parse_args().index_profile == profile


def test_cli_rejects_unknown_index_profile(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_experiment.py", "--index-profile", "unknown"]
    )

    with pytest.raises(SystemExit):
        parse_args()