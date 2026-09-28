#!/usr/bin/env python3
"""
CLI script for running experiments

This script provides a simple command-line interface for running
the retrieval experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment import run_experiment
from src.retrieval.index_profile import INDEX_PROFILES


def main(output_dir: Path, d: int, testing: bool, index_profile: str) -> None:
    """Main entry point for the CLI script."""
    if testing:
        print("\nRunning experiment in testing mode with a small synthetic dataset...")
    else:
        print("\nRunning full experiment with all 500K documents...")
    summary = run_experiment(
        output_dir, d=d, testing=testing, index_profile=index_profile
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run retrieval experiments.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts_full",
        help="Directory to save artifacts (default: artifacts_full)",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=128,
        help="Dimensionality of dense vectors for Pocket Dimension (default: 128)",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Run in testing mode with a smaller synthetic dataset (default: False)",
    )
    parser.add_argument(
        "--index-profile",
        choices=tuple(INDEX_PROFILES),
        default="flat",
        help="FAISS index profile (default: flat)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        raise ValueError(
            f"Output directory {output_dir} does not exist. Please create it first."
        )
    if not isinstance(args.d, int) or args.d <= 63:
        raise ValueError(
            f"Dimensionality {args.d} is not valid. Please provide a positive integer greater than 63."
        )
    if not isinstance(args.testing, bool):
        raise ValueError(
            f"Testing flag {args.testing} is not valid. Please provide a boolean value."
        )

    main(output_dir, args.d, args.testing, args.index_profile)
