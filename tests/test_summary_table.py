import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarize_profiles import METRICS, PROFILE_LABELS, render_comparison_table


def make_summary(profile_name, value):
    metrics = {metric: value for metric in METRICS}
    return {
        "bm25": {
            "index_size_MB": value,
            "initialization_time_seconds": value,
            "query_time_seconds": value,
            "metrics": metrics,
        },
        "pocket_dimension": {
            "index_profile": {"name": profile_name},
            "index_size_MB": value,
            "initialization_time_seconds": value,
            "query_time_seconds": value,
            "metrics": metrics,
        },
    }


def test_render_comparison_table_includes_grouped_columns_rows_and_values():
    summaries = [
        make_summary(profile_name, value)
        for value, profile_name in enumerate(PROFILE_LABELS, start=1)
    ]

    table = render_comparison_table(summaries)
    lines = table.splitlines()

    assert '      <th rowspan="2" scope="col">BM25</th>' in lines
    assert '      <th colspan="5" scope="colgroup">Pocket Dimension</th>' in lines
    assert [line for line in lines if 'scope="col">' in line][-5:] == [
        f'      <th scope="col">{label}</th>' for label in PROFILE_LABELS.values()
    ]
    assert '      <th scope="row">Index Size (MB)</th>' in lines
    assert '      <td>1.00</td>' in lines
    assert '      <th scope="row">MAP@1</th>' in lines
    assert '      <td>5.00000</td>' in lines
    assert '      <th scope="row">Recall@10</th>' in lines
    assert lines.count("    <tr>") == 2 + 3 + len(METRICS)
    assert lines[-1] == "</table>"


def test_render_markdown_table_reports_missing_profiles():
    summaries = [make_summary("flat", 1)]

    with pytest.raises(ValueError, match="Missing required index profile summaries: SQ8"):
        render_comparison_table(summaries)


def test_render_markdown_table_rejects_duplicate_profile_summaries():
    summaries = [make_summary("flat", 1), make_summary("flat", 2)]

    with pytest.raises(ValueError, match="Multiple summaries found for profile 'flat'"):
        render_comparison_table(summaries)