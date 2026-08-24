from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from generate_experiment_tables import (  # noqa: E402
    CIRCUITS,
    QUBITS,
    build_tables,
    cross_matching_phase_count,
    directional_shape_index,
    execution_descriptor,
    parameter_descriptor,
    phase_target_counts,
    strategy_descriptor,
    tile_bits,
)


FIELDS = (
    "timestamp_utc,status,execution_mode,kernel_variant,circuit,qubits,layers,"
    "precision,warmup_steps,steps,energy,grad_json,time_median_s,forward_mean_s,"
    "hamiltonian_mean_s,backward_mean_s,error\n"
)


def write_fixture(path: Path, *, fixed: bool, lightning: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(FIELDS)
        writer = csv.writer(stream, lineterminator="\n")
        for circuit in CIRCUITS:
            for qubits in QUBITS:
                variant = "f128r2_b128r2" if fixed else (
                    "f64r4_b64r4" if qubits >= 20 else "f128r2_b128r2"
                )
                time = 3.0 if lightning else (1.2 if fixed else 1.0)
                writer.writerow(
                    [
                        "2026-08-16T00:00:00+00:00", "ok", "optimized", variant,
                        circuit, qubits, 8, "float64", 1, 2, 0.5,
                        json.dumps([0.1, 0.2]), time, 0.2, 0.1, 0.7, "",
                    ]
                )


def write_paired_fixture(path: Path) -> None:
    fields = (
        "repetition,circuit,qubits,layers,selected_variant,default_variant,"
        "selected_median_ms,default_median_ms,default_over_selected,"
        "default_over_selected_forward,default_over_selected_hamiltonian,"
        "default_over_selected_backward,energy_abs_error,"
        "gradient_max_abs_error\n"
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(fields)
        writer = csv.writer(stream, lineterminator="\n")
        for circuit in CIRCUITS:
            for qubits in QUBITS:
                if qubits < 20:
                    continue
                repetitions = 5 if qubits <= 24 else 3
                for repetition in range(repetitions):
                    writer.writerow(
                        [
                            repetition, circuit, qubits, 8, "f64r4_b64r4",
                            "f128r2_b128r2", 1.0, 1.2, 1.2, 1.1, 1.0, 1.3,
                            0.0, 0.0,
                        ]
                    )


def test_parameter_descriptors_match_production_phase_rules():
    assert tile_bits(64, 4) == 10
    assert phase_target_counts(24, 10, fixed_low_5=False) == (10, 10, 4)
    assert phase_target_counts(24, 10, fixed_low_5=True) == (10, 5, 5, 4)
    assert cross_matching_phase_count(24, 10) == 4
    assert execution_descriptor("su2-hea", 24, "f128r3_b64r4") == (
        "F:t128r3m1-C[10+10+4]/B:t64r4m1-X[10+5+5+4]"
    )
    assert "B:t128r2m1-C[9+7]" in execution_descriptor(
        "su2-hea", 16, "f128r2_b128r2"
    )
    assert "B:t128r2m1-X[9+4+4+1]" in execution_descriptor(
        "su2-hea", 18, "f128r2_b128r2"
    )
    assert execution_descriptor("xxz-hva", 24, "f128r3_b32r3") == (
        "F:t128r3m1-Zx4/B:t32r3m1-Zx4"
    )
    assert "separate-matching" in strategy_descriptor("xxz-hva", 6)
    assert execution_descriptor(
        "qaoa",
        10,
        "f32r2_b32r2",
        "compact:L2R2W0-L4R2W0",
        "compact:L4R2W0-L2R2W0",
    ) == (
        "F:t32r2m1-C[L2R2W0-L4R2W0]/"
        "B:t32r2m1-C[L4R2W0-L2R2W0]"
    )
    assert "B=split" in strategy_descriptor("su2-hea", 18)
    assert "B=split" in strategy_descriptor("su2-hea", 20)
    assert "B=split" in parameter_descriptor("rzz-hea", 24, "f64r3_b64r4")
    assert "B=RX+RZ+RZZ-fused" in parameter_descriptor(
        "rzz-hea", 18, "f128r2_b128r2"
    )
    assert "F=cost+RX-fused;B=cost+RX-fused" in strategy_descriptor("qaoa", 20)
    assert "init+cost=combined" in strategy_descriptor("qaoa", 22)
    assert "D=RZ/RZZ-k6-t64" in strategy_descriptor("rzz-hea", 10)
    assert "D=RZ/RZZ-k10-t64" in strategy_descriptor("rzz-hea", 20)
    assert execution_descriptor("rzz-hea", 20, "f128r2_b128r2_d10").startswith(
        "F:t128r2"
    )


def test_directional_shape_index_keeps_forward_and_backward_separate(tmp_path):
    path = tmp_path / "directional.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "circuit",
                "qubits",
                "direction",
                "candidate",
                "baseline_over_candidate",
                "correct",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "circuit": "ra-hea",
                    "qubits": 22,
                    "direction": "forward",
                    "candidate": "joint",
                    "baseline_over_candidate": 0.9,
                    "correct": 1,
                },
                {
                    "circuit": "ra-hea",
                    "qubits": 22,
                    "direction": "backward",
                    "candidate": "joint",
                    "baseline_over_candidate": 1.1,
                    "correct": 1,
                },
            )
        )
    assert directional_shape_index(path) == {
        ("ra-hea", 22, "forward"): 0.9,
        ("ra-hea", 22, "backward"): 1.1,
    }


def test_build_tables_uses_native_and_uniform_fixed_policy(tmp_path):
    optimized = tmp_path / "optimized.csv"
    fixed = tmp_path / "fixed.csv"
    lightning = tmp_path / "lightning.csv"
    paired = tmp_path / "paired.csv"
    write_fixture(optimized, fixed=False)
    write_fixture(fixed, fixed=True)
    write_fixture(lightning, fixed=True, lightning=True)
    write_paired_fixture(paired)
    main, parameters = build_tables(optimized, fixed, lightning, paired)
    assert len(main) == len(CIRCUITS) * len(QUBITS) == 65
    assert len(parameters) == 65
    assert main[0]["speedup_vs_lightning_native"] == 3.0
    assert parameters[0]["speedup_vs_default"] == 1.0
    assert main[0]["sad_execution_parameters"].startswith("F:t128r2")
    assert main[0]["sad_structural_parameters"].startswith("RA-real")
    assert parameters[0]["experiment_group"] == (
        "execution_parameter_dispatch_experiment"
    )
    assert parameters[0]["default_policy"] == (
        "uniform_conservative_execution_same_structure"
    )
    assert parameters[0]["selected_policy"] == "circuit_size_execution_dispatch"
    assert (
        parameters[0]["shared_structural_parameters"]
        == main[0]["sad_structural_parameters"]
    )
    assert {row["default_variant"] for row in parameters} == {"f128r2_b128r2"}
    assert all(row["correct"] == 1 for row in parameters)
