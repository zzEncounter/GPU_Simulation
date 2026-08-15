"""Paired A/B benchmark for selected versus uniform SAD kernel parameters."""

from __future__ import annotations

import csv
import gc
import importlib
import os
import random
from pathlib import Path

import numpy as np

import benchmark_sad


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark" / "results" / "parameter_policy_paired_raw.csv"
SAD_PYTHON = ROOT / "sad" / "python"
import sys

sys.path.insert(0, str(SAD_PYTHON))
sad_baseline = importlib.import_module("sad_baseline")
sad_runner = importlib.import_module("sad_baseline.runner")

FIELDS = (
    "repetition",
    "order",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "steps",
    "warmup_steps",
    "selected_variant",
    "default_variant",
    "selected_median_ms",
    "default_median_ms",
    "default_over_selected",
    "selected_forward_mean_ms",
    "default_forward_mean_ms",
    "default_over_selected_forward",
    "selected_hamiltonian_mean_ms",
    "default_hamiltonian_mean_ms",
    "default_over_selected_hamiltonian",
    "selected_backward_mean_ms",
    "default_backward_mean_ms",
    "default_over_selected_backward",
    "energy_abs_error",
    "gradient_max_abs_error",
)


def repetitions_for_qubits(qubits: int) -> int:
    return 5 if qubits <= 24 else 3


def run(circuit: str, qubits: int, *, fixed: bool):
    if fixed:
        os.environ["SAD_DISABLE_VARIANT_DISPATCH"] = "1"
    else:
        os.environ.pop("SAD_DISABLE_VARIANT_DISPATCH", None)
    return sad_baseline.energy_and_grad(
        circuit=circuit,
        random_seed=benchmark_sad.RANDOM_SEED,
        scalability=(qubits, benchmark_sad.LAYERS),
        batches=benchmark_sad.BATCHES,
        precision=benchmark_sad.PRECISION,
        steps=benchmark_sad.steps_for_qubits(qubits),
        warmup_steps=1,
        device_name=benchmark_sad.DEVICE_NAME,
    )


def differing_scenarios() -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for circuit_id, circuit in enumerate(benchmark_sad.CIRCUITS):
        for qubits in benchmark_sad.QUBITS:
            selected, _ = sad_runner._select_library(circuit_id, qubits, "optimized")
            if selected != "f128r2_b128r2":
                result.append((circuit, qubits))
    return result


def main() -> None:
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    scenarios = differing_scenarios()
    total = sum(repetitions_for_qubits(qubits) for _, qubits in scenarios)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        completed = 0
        max_repetitions = max(repetitions_for_qubits(q) for _, q in scenarios)
        for repetition in range(max_repetitions):
            eligible = [
                scenario for scenario in scenarios
                if repetition < repetitions_for_qubits(scenario[1])
            ]
            random.Random(20260816 + repetition).shuffle(eligible)
            for scenario_index, (circuit, qubits) in enumerate(eligible):
                order = ("selected", "default")
                if (repetition + scenario_index) % 2:
                    order = tuple(reversed(order))
                measured = {}
                for policy in order:
                    measured[policy] = run(
                        circuit, qubits, fixed=policy == "default"
                    )
                selected = measured["selected"]
                default = measured["default"]
                selected_ms = 1000 * selected.median_step_time_s
                default_ms = 1000 * default.median_step_time_s
                selected_forward = 1000 * float(np.mean(selected.forward_times_s))
                default_forward = 1000 * float(np.mean(default.forward_times_s))
                selected_hamiltonian = 1000 * float(
                    np.mean(selected.hamiltonian_times_s)
                )
                default_hamiltonian = 1000 * float(
                    np.mean(default.hamiltonian_times_s)
                )
                selected_backward = 1000 * float(np.mean(selected.backward_times_s))
                default_backward = 1000 * float(np.mean(default.backward_times_s))
                row = {
                    "repetition": repetition,
                    "order": "-".join(order),
                    "circuit": circuit,
                    "qubits": qubits,
                    "layers": benchmark_sad.LAYERS,
                    "precision": benchmark_sad.PRECISION,
                    "steps": benchmark_sad.steps_for_qubits(qubits),
                    "warmup_steps": 1,
                    "selected_variant": selected.kernel_variant,
                    "default_variant": default.kernel_variant,
                    "selected_median_ms": selected_ms,
                    "default_median_ms": default_ms,
                    "default_over_selected": default_ms / selected_ms,
                    "selected_forward_mean_ms": selected_forward,
                    "default_forward_mean_ms": default_forward,
                    "default_over_selected_forward": default_forward / selected_forward,
                    "selected_hamiltonian_mean_ms": selected_hamiltonian,
                    "default_hamiltonian_mean_ms": default_hamiltonian,
                    "default_over_selected_hamiltonian": default_hamiltonian
                    / selected_hamiltonian,
                    "selected_backward_mean_ms": selected_backward,
                    "default_backward_mean_ms": default_backward,
                    "default_over_selected_backward": default_backward
                    / selected_backward,
                    "energy_abs_error": abs(selected.energy - default.energy),
                    "gradient_max_abs_error": float(
                        np.max(np.abs(selected.grad - default.grad))
                    ),
                }
                writer.writerow(row)
                stream.flush()
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] {circuit} {qubits}q "
                    f"pair {repetition + 1}: {default_ms / selected_ms:.4f}x "
                    f"({order[0]} first)",
                    flush=True,
                )
                gc.collect()
    os.environ.pop("SAD_DISABLE_VARIANT_DISPATCH", None)
    print(f"CSV written to {OUTPUT}")


if __name__ == "__main__":
    main()
