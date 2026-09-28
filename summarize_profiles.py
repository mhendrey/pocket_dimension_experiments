#!/usr/bin/env python3
"""Print an HTML comparison table of BM25 and index profile summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_LABELS = {
    "flat": "Flat",
    "sq8": "SQ8",
    "rabitq-refine-sq8": "RaBitQ-Refine-SQ8",
    "hnsw32": "HNSW32",
    "ivf-hnsw-rabitq-refine-sq8": "IVF-RaBitQ-HNSW-Refine-SQ8",
}
METRICS = (
    "MAP@1",
    "MAP@3",
    "MAP@5",
    "MAP@10",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "NDCG@10",
    "P@1",
    "P@3",
    "P@5",
    "P@10",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Recall@10",
)


def load_summaries(artifacts_root: Path) -> list[dict[str, Any]]:
    """Load all summary.json files in artifacts_* directories."""
    summary_paths = sorted(artifacts_root.glob("artifacts_*/summary.json"))
    if not summary_paths:
        raise ValueError(f"No artifacts_*/summary.json files found in {artifacts_root}")

    summaries = []
    for path in summary_paths:
        with path.open(encoding="utf-8") as summary_file:
            summaries.append(json.load(summary_file))
    return summaries


def render_comparison_table(summaries: list[dict[str, Any]]) -> str:
    """Render a grouped HTML table of BM25 and per-profile metrics."""
    profiles: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        profile_name = summary["pocket_dimension"]["index_profile"]["name"]
        if profile_name in PROFILE_LABELS:
            if profile_name in profiles:
                raise ValueError(f"Multiple summaries found for profile {profile_name!r}")
            profiles[profile_name] = summary["pocket_dimension"]

    missing = [name for name in PROFILE_LABELS if name not in profiles]
    if missing:
        labels = ", ".join(PROFILE_LABELS[name] for name in missing)
        raise ValueError(f"Missing required index profile summaries: {labels}")
    if not summaries:
        raise ValueError("At least one summary is required to report BM25 results")

    bm25 = summaries[0]["bm25"]
    columns = [("BM25", bm25)] + [
        (PROFILE_LABELS[name], profiles[name]) for name in PROFILE_LABELS
    ]
    rows = [
        ("Index Size (MB)", "index_size_MB", None, 2),
        ("Initialization Time (seconds)", "initialization_time_seconds", None, 3),
        ("Query Time (seconds)", "query_time_seconds", None, 3),
    ]
    rows.extend((metric, None, metric, 5) for metric in METRICS)

    table = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2" scope="col">Metric</th>',
        '      <th rowspan="2" scope="col">BM25</th>',
        '      <th colspan="5" scope="colgroup">Pocket Dimension</th>',
        "    </tr>",
        "    <tr>",
    ]
    table.extend(
        f'      <th scope="col">{label}</th>' for label, _ in columns[1:]
    )
    table.extend(("    </tr>", "  </thead>", "  <tbody>"))
    for label, field, metric, precision in rows:
        values = []
        for _, result in columns:
            value = result["metrics"][metric] if metric else result[field]
            values.append(f"      <td>{value:.{precision}f}</td>")
        table.extend(("    <tr>", f"      <th scope=\"row\">{label}</th>"))
        table.extend(values)
        table.append("    </tr>")
    table.extend(("  </tbody>", "</table>"))
    return "\n".join(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print an HTML table comparing BM25 and index profile results."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing artifacts_*/summary.json (default: current directory)",
    )
    args = parser.parse_args()

    try:
        print(render_comparison_table(load_summaries(args.artifacts_root)))
    except (KeyError, json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()