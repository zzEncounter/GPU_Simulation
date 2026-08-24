from __future__ import annotations

import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
RESULTS = BENCHMARK / "results"
sys.path.insert(0, str(BENCHMARK))

from benchmark_parameter_selection_stages import (  # noqa: E402
    ALL_FIXED_FLAGS,
    POLICIES,
    repetitions_for_qubits,
)
from generate_parameter_selection_report import (  # noqa: E402
    CIRCUITS,
    QUBITS,
    read_csv,
    write_report,
)


def test_all_fixed_policy_explicitly_covers_search_axes():
    definitions = {
        flag.split("=", 1)[0].removeprefix("-D") for flag in ALL_FIXED_FLAGS
    }
    assert len(definitions) == len(ALL_FIXED_FLAGS)
    assert {
        "SAD_FORWARD_BLOCK_THREADS",
        "SAD_FORWARD_REGISTER_BITS",
        "SAD_BLOCK_THREADS",
        "SAD_REGISTER_BITS",
        "SAD_MAILBOX_CHUNKS",
        "SAD_DIAGONAL_LOOKUP_BITS",
        "SAD_CNOT_FORWARD_SCATTER",
        "SAD_REAL_AMPLITUDE",
        "SAD_RA_FORWARD_FUSED",
        "SAD_RA_BACKWARD_FUSED",
        "SAD_SU2_FORWARD_STRATEGY",
        "SAD_SU2_BACKWARD_STRATEGY",
        "SAD_RZZ_FORWARD_FUSED",
        "SAD_RZZ_BACKWARD_STRATEGY",
        "SAD_QAOA_INITIAL_STRATEGY",
        "SAD_QAOA_FUSE_COST_RX",
        "SAD_QAOA_COMPACT_LOOKUP",
        "SAD_QAOA_FUSED_BACKWARD",
        "SAD_XXZ_CROSS_MATCHING",
    } <= definitions


def test_three_stage_raw_data_is_complete_and_paired():
    raw = read_csv(RESULTS / "parameter_selection_stages_raw.csv")
    summary = {
        (row["circuit"], int(row["qubits"])): row
        for row in read_csv(RESULTS / "parameter_selection_stages.csv")
    }
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[(row["circuit"], int(row["qubits"]), int(row["repetition"]))].append(row)
    assert len(grouped) == sum(
        repetitions_for_qubits(q) for _ in CIRCUITS for q in QUBITS
    )
    for (circuit, qubits, _), rows in grouped.items():
        assert circuit in CIRCUITS
        assert Counter(row["policy"] for row in rows) == Counter(POLICIES)
        assert len({row["order"] for row in rows}) == 1
        assert all(row["correct"] == "1" for row in rows)
        assert len(rows) == len(POLICIES)
        assert qubits in QUBITS
    by_scenario: dict[tuple[str, int], list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (circuit, qubits, _), rows in grouped.items():
        by_scenario[(circuit, qubits)].append({row["policy"]: row for row in rows})
    for scenario, repetitions in by_scenario.items():
        structure = []
        execution = []
        total = []
        for policies in repetitions:
            fixed_ms = float(policies["all_fixed"]["median_ms"])
            structure_ms = float(policies["structure_selected"]["median_ms"])
            best_ms = float(policies["fully_selected"]["median_ms"])
            structure.append(fixed_ms / structure_ms)
            execution.append(structure_ms / best_ms)
            total.append(fixed_ms / best_ms)
        row = summary[scenario]
        assert math.isclose(
            statistics.median(structure),
            float(row["structure_speedup_vs_all_fixed"]),
            rel_tol=1e-12,
        )
        assert math.isclose(
            statistics.median(execution),
            float(row["execution_speedup_vs_structure_selected"]),
            rel_tol=1e-12,
        )
        assert math.isclose(
            statistics.median(total),
            float(row["fully_selected_speedup_vs_all_fixed"]),
            rel_tol=1e-12,
        )


def test_three_stage_summary_and_single_report(tmp_path):
    stages = read_csv(RESULTS / "parameter_selection_stages.csv")
    assert len(stages) == len(CIRCUITS) * len(QUBITS) == 65
    assert {(row["circuit"], int(row["qubits"])) for row in stages} == {
        (circuit, q) for circuit in CIRCUITS for q in QUBITS
    }
    assert all(row["correct"] == "1" for row in stages)
    assert all(float(row["fully_selected_speedup_vs_all_fixed"]) > 0 for row in stages)

    report = tmp_path / "参数选择.md"
    write_report(
        report,
        stages,
        read_csv(RESULTS / "parameter_search_experiment.csv"),
        read_csv(RESULTS / "structure_strategy_experiment.csv"),
        read_csv(RESULTS / "rotation_position_group_summary.csv"),
        read_csv(RESULTS / "heterogeneous_phase_paired_raw.csv"),
        read_csv(RESULTS / "mailbox_phase_refinement_summary.csv"),
    )
    text = report.read_text(encoding="utf-8")
    assert "全固定 → 部分选择 → 最佳" in text
    assert "## 五、L2 cache" in text
    assert all(f"### {circuit}" in text for circuit in CIRCUITS)

    main_text = (ROOT / "docs" / "experiments" / "主实验.md").read_text(
        encoding="utf-8"
    )
    assert "[参数选择](参数选择.md)" in main_text
    assert "## 三、分层参数选择与执行参数 A/B" not in main_text
    assert not (ROOT / "docs" / "experiments" / "RX_RY方向参数选择.md").exists()
