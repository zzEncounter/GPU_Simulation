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
    parameter_descriptor,
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


def test_parameter_descriptor_is_compact_and_scenario_specific():
    assert parameter_descriptor("su2-hea", 24, "f128r3_b64r4") == (
        "F128r3/B64r4;compact;m1;F:lookup-fused/B:split"
    )
    assert "B:split" in parameter_descriptor(
        "rzz-hea", 24, "f64r3_b64r4"
    )
    assert "B:fused" in parameter_descriptor(
        "rzz-hea", 18, "f128r2_b128r2"
    )


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
    assert {row["default_variant"] for row in parameters} == {"f128r2_b128r2"}
    assert all(row["correct"] == 1 for row in parameters)
