from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from generate_strategy_report import fusion_factors, factor_winner  # noqa: E402


def test_fusion_factor_parser_covers_independent_choices():
    assert fusion_factors(
        "su2-hea", "lookup-forward_split-backward"
    ) == ("lookup", "split")
    assert fusion_factors("qaoa", "fused-backward") == ("split", "fused")
    assert fusion_factors("ra-hea", "real-fused-both") is None


def test_factor_winner_medians_out_orthogonal_choice():
    rows = [
        {
            "variant": "split-forward_split-backward",
            "forward_ms": "10",
            "backward_ms": "8",
        },
        {
            "variant": "split-forward_fused-backward",
            "forward_ms": "12",
            "backward_ms": "7",
        },
        {
            "variant": "fused-forward_split-backward",
            "forward_ms": "7",
            "backward_ms": "9",
        },
        {
            "variant": "fused-forward_fused-backward",
            "forward_ms": "9",
            "backward_ms": "6",
        },
    ]
    assert factor_winner(rows, "rzz-hea", "forward")[0] == "fused"
    assert factor_winner(rows, "rzz-hea", "backward")[0] == "fused"


def test_generated_report_is_chinese():
    report = (
        BENCHMARK.parent / "docs" / "research" / "EXECUTION_STRATEGY_REPORT.md"
    ).read_text(encoding="utf-8")
    assert report.startswith("# 执行策略搜索\n")
    assert "## 硬件与搜索覆盖范围" in report
    assert "## 可移植调优流程" in report
    assert "# Execution Strategy Search" not in report
