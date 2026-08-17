"""Paired E2E validation of independently selected forward/backward shapes.

The current joint policy has a small set of cases where a backward win pays
for a slower forward shape.  This experiment keeps the selected backward
geometry but restores the independently faster conservative forward geometry.
It reports each direction separately and only then combines them as wall time.
Rows are append/resume safe.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark"))
from sad_baseline import energy_and_grad  # noqa: E402
from sad_baseline import runner as sad_runner  # noqa: E402
import benchmark_sad  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "directional_policy_paired_raw.csv"
BASELINE = "f128r2_b128r2"


@dataclass(frozen=True)
class Scenario:
    circuit: str
    qubits: int
    joint: str
    directional: str


SCENARIOS = (
    Scenario("ra-hea", 22, "f64r4_b64r4", "f128r2_b64r4"),
    Scenario("su2-hea", 22, "f128r3_b64r4", "f128r2_b64r4"),
    Scenario("su2-hea", 24, "f128r3_b64r4", "f128r2_b64r4"),
    Scenario("rzz-hea", 24, "f64r3_b64r4", "f128r2_b64r4"),
    Scenario("rzz-hea", 26, "f64r4_b128r3", "f128r2_b128r3"),
    Scenario("qaoa", 18, "f32r2_b32r2", "f128r2_b32r2"),
    Scenario("qaoa", 26, "f64r4_b128r3", "f128r2_b128r3"),
)

FIELDS = (
    "repetition",
    "order",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "steps",
    "policy",
    "variant",
    "baseline_variant",
    "joint_variant",
    "directional_variant",
    "median_ms",
    "forward_mean_ms",
    "hamiltonian_mean_ms",
    "backward_mean_ms",
    "baseline_over_policy",
    "baseline_over_policy_forward",
    "baseline_over_policy_hamiltonian",
    "baseline_over_policy_backward",
    "joint_over_policy",
    "joint_over_policy_forward",
    "joint_over_policy_backward",
    "energy_abs_error",
    "gradient_max_abs_error",
    "correct",
)


def repetitions_for_qubits(qubits: int) -> int:
    return 5 if qubits <= 24 else 3


def _library(variant: str) -> Path:
    if variant == BASELINE:
        path = sad_runner._DEFAULT_LIBRARY
    else:
        path = sad_runner._VARIANT_LIBRARIES[variant]
    sad_runner._build_library(path)
    return path


def _run(scenario: Scenario, library: Path):
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    return energy_and_grad(
        circuit=scenario.circuit,
        random_seed=benchmark_sad.RANDOM_SEED,
        scalability=(scenario.qubits, benchmark_sad.LAYERS),
        batches=benchmark_sad.BATCHES,
        precision=benchmark_sad.PRECISION,
        steps=benchmark_sad.steps_for_qubits(scenario.qubits),
        warmup_steps=1,
        device_name=benchmark_sad.DEVICE_NAME,
        forward_phase_plan="",
        backward_phase_plan="",
    )


def _completed(path: Path) -> set[tuple[int, str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["repetition"]), row["circuit"], int(row["qubits"]), row["policy"])
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def _times(result) -> tuple[float, float, float, float]:
    return (
        1e3 * result.median_step_time_s,
        1e3 * statistics.fmean(result.forward_times_s),
        1e3 * statistics.fmean(result.hamiltonian_times_s),
        1e3 * statistics.fmean(result.backward_times_s),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")

    variants = {BASELINE}
    for scenario in SCENARIOS:
        variants.update((scenario.joint, scenario.directional))
    libraries = {variant: _library(variant) for variant in sorted(variants)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    failures: list[str] = []
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        max_repetitions = max(repetitions_for_qubits(s.qubits) for s in SCENARIOS)
        for repetition in range(min(args.rounds, max_repetitions)):
            eligible = [
                scenario
                for scenario in SCENARIOS
                if repetition < repetitions_for_qubits(scenario.qubits)
            ]
            random.Random(20260817 + repetition).shuffle(eligible)
            for scenario in eligible:
                policies = {
                    "baseline": BASELINE,
                    "joint": scenario.joint,
                    "directional": scenario.directional,
                }
                pending = {
                    name: variant
                    for name, variant in policies.items()
                    if (repetition, scenario.circuit, scenario.qubits, name) not in done
                }
                if not pending:
                    continue
                order = list(pending)
                random.Random(
                    f"directional-{repetition}-{scenario.circuit}-{scenario.qubits}"
                ).shuffle(order)
                measured = {
                    name: _run(scenario, libraries[pending[name]]) for name in order
                }
                # Resume safety normally leaves all three pending together.  If a
                # partial row exists, remeasure its reference in-memory only.
                for reference in ("baseline", "joint"):
                    if reference not in measured:
                        measured[reference] = _run(
                            scenario, libraries[policies[reference]]
                        )
                baseline_times = _times(measured["baseline"])
                joint_times = _times(measured["joint"])
                for name in order:
                    result = measured[name]
                    values = _times(result)
                    energy_error = abs(result.energy - measured["baseline"].energy)
                    gradient_error = float(
                        np.max(np.abs(result.grad - measured["baseline"].grad))
                    )
                    correct = energy_error <= 1e-10 and gradient_error <= 1e-9
                    writer.writerow(
                        {
                            "repetition": repetition,
                            "order": "-".join(order),
                            "circuit": scenario.circuit,
                            "qubits": scenario.qubits,
                            "layers": benchmark_sad.LAYERS,
                            "precision": benchmark_sad.PRECISION,
                            "steps": benchmark_sad.steps_for_qubits(scenario.qubits),
                            "policy": name,
                            "variant": pending[name],
                            "baseline_variant": BASELINE,
                            "joint_variant": scenario.joint,
                            "directional_variant": scenario.directional,
                            "median_ms": values[0],
                            "forward_mean_ms": values[1],
                            "hamiltonian_mean_ms": values[2],
                            "backward_mean_ms": values[3],
                            "baseline_over_policy": baseline_times[0] / values[0],
                            "baseline_over_policy_forward": baseline_times[1] / values[1],
                            "baseline_over_policy_hamiltonian": baseline_times[2] / values[2],
                            "baseline_over_policy_backward": baseline_times[3] / values[3],
                            "joint_over_policy": joint_times[0] / values[0],
                            "joint_over_policy_forward": joint_times[1] / values[1],
                            "joint_over_policy_backward": joint_times[3] / values[3],
                            "energy_abs_error": energy_error,
                            "gradient_max_abs_error": gradient_error,
                            "correct": int(correct),
                        }
                    )
                    stream.flush()
                    done.add((repetition, scenario.circuit, scenario.qubits, name))
                    print(
                        f"{scenario.circuit:8s} {scenario.qubits}q {name:11s} "
                        f"F={baseline_times[1] / values[1]:.3f}x "
                        f"B={baseline_times[3] / values[3]:.3f}x "
                        f"wall={baseline_times[0] / values[0]:.3f}x",
                        flush=True,
                    )
                    if not correct:
                        failures.append(
                            f"{scenario.circuit} {scenario.qubits}q {name}"
                        )
                gc.collect()
    os.environ.pop("SAD_LIBRARY_PATH", None)
    if failures:
        raise RuntimeError("correctness failures: " + ", ".join(failures))
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
