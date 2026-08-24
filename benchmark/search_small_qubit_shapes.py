"""Paired end-to-end check of small CUDA tiles at small qubit counts.

The historical execution search started at 20 qubits, so a production policy
that retained the conservative F128r2/B128r2 binary below that boundary was
not evidence that small tiles were inferior.  This focused experiment tests
the two microbenchmark finalists without crossing them with phase, mailbox,
diagonal, or fusion choices.  Rows are append/resume safe.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402
from sad_baseline import runner as sad_runner  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "small_qubit_shape_paired_raw.csv"
)
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
BASELINE = "f128r2_b128r2"
CANDIDATES = ("f64r2_b64r2", "f32r2_b32r2")
FIELDS = (
    "repetition",
    "order",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "steps",
    "candidate_variant",
    "baseline_variant",
    "candidate_median_ms",
    "baseline_median_ms",
    "baseline_over_candidate",
    "candidate_forward_mean_ms",
    "baseline_forward_mean_ms",
    "baseline_over_candidate_forward",
    "candidate_hamiltonian_mean_ms",
    "baseline_hamiltonian_mean_ms",
    "baseline_over_candidate_hamiltonian",
    "candidate_backward_mean_ms",
    "baseline_backward_mean_ms",
    "baseline_over_candidate_backward",
    "energy_abs_error",
    "gradient_max_abs_error",
    "correct",
)


def _parse_qubits(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not result or any(item < 4 or item > 18 for item in result):
        raise argparse.ArgumentTypeError("qubits must be comma-separated 4..18")
    return result


def _library(variant: str) -> Path:
    if variant == BASELINE:
        path = sad_runner._DEFAULT_LIBRARY
    else:
        path = sad_runner._VARIANT_LIBRARIES[variant]
    sad_runner._build_library(path)
    return path


def _run(
    circuit: str,
    qubits: int,
    layers: int,
    steps: int,
    library: Path,
):
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    return energy_and_grad(
        circuit=circuit,
        random_seed=42,
        scalability=(qubits, layers),
        precision="float64",
        steps=steps,
        warmup_steps=5,
        forward_phase_plan="",
        backward_phase_plan="",
    )


def _completed(path: Path) -> set[tuple[int, str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (
                int(row["repetition"]),
                row["circuit"],
                int(row["qubits"]),
                row["candidate_variant"],
            )
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=_parse_qubits, default=(8, 12, 16))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.layers, args.steps, args.rounds) < 1:
        parser.error("layers, steps, and rounds must be positive")

    libraries = {
        variant: _library(variant) for variant in (BASELINE, *CANDIDATES)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    failures: list[str] = []
    scenarios = tuple(
        (circuit, qubits)
        for circuit in CIRCUITS
        for qubits in args.qubits
        if circuit not in {"rzz-hea", "qaoa", "xxz-hva"} or qubits % 2 == 0
    )
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            shuffled = list(scenarios)
            random.Random(20260816 + repetition).shuffle(shuffled)
            for scenario_index, (circuit, qubits) in enumerate(shuffled):
                pending = [
                    candidate
                    for candidate in CANDIDATES
                    if (repetition, circuit, qubits, candidate) not in done
                ]
                if not pending:
                    continue
                order = [BASELINE, *pending]
                random.Random(
                    f"small-shape-{repetition}-{scenario_index}"
                ).shuffle(order)
                measured = {
                    variant: _run(
                        circuit,
                        qubits,
                        args.layers,
                        args.steps,
                        libraries[variant],
                    )
                    for variant in order
                }
                baseline = measured[BASELINE]
                for candidate in pending:
                    result = measured[candidate]
                    candidate_ms = 1e3 * result.median_step_time_s
                    baseline_ms = 1e3 * baseline.median_step_time_s
                    candidate_forward = 1e3 * statistics.fmean(
                        result.forward_times_s
                    )
                    baseline_forward = 1e3 * statistics.fmean(
                        baseline.forward_times_s
                    )
                    candidate_hamiltonian = 1e3 * statistics.fmean(
                        result.hamiltonian_times_s
                    )
                    baseline_hamiltonian = 1e3 * statistics.fmean(
                        baseline.hamiltonian_times_s
                    )
                    candidate_backward = 1e3 * statistics.fmean(
                        result.backward_times_s
                    )
                    baseline_backward = 1e3 * statistics.fmean(
                        baseline.backward_times_s
                    )
                    energy_error = abs(result.energy - baseline.energy)
                    gradient_error = float(
                        np.max(np.abs(result.grad - baseline.grad))
                    )
                    correct = energy_error <= 1e-10 and gradient_error <= 1e-9
                    writer.writerow(
                        {
                            "repetition": repetition,
                            "order": "-".join(order),
                            "circuit": circuit,
                            "qubits": qubits,
                            "layers": args.layers,
                            "precision": "float64",
                            "steps": args.steps,
                            "candidate_variant": candidate,
                            "baseline_variant": BASELINE,
                            "candidate_median_ms": candidate_ms,
                            "baseline_median_ms": baseline_ms,
                            "baseline_over_candidate": baseline_ms / candidate_ms,
                            "candidate_forward_mean_ms": candidate_forward,
                            "baseline_forward_mean_ms": baseline_forward,
                            "baseline_over_candidate_forward": (
                                baseline_forward / candidate_forward
                            ),
                            "candidate_hamiltonian_mean_ms": candidate_hamiltonian,
                            "baseline_hamiltonian_mean_ms": baseline_hamiltonian,
                            "baseline_over_candidate_hamiltonian": (
                                baseline_hamiltonian / candidate_hamiltonian
                            ),
                            "candidate_backward_mean_ms": candidate_backward,
                            "baseline_backward_mean_ms": baseline_backward,
                            "baseline_over_candidate_backward": (
                                baseline_backward / candidate_backward
                            ),
                            "energy_abs_error": energy_error,
                            "gradient_max_abs_error": gradient_error,
                            "correct": int(correct),
                        }
                    )
                    stream.flush()
                    done.add((repetition, circuit, qubits, candidate))
                    print(
                        f"{circuit:8s} {qubits:2d}q {candidate}: "
                        f"{baseline_ms / candidate_ms:.3f}x correct={correct}",
                        flush=True,
                    )
                    if not correct:
                        failures.append(f"{circuit} {qubits}q {candidate}")
                gc.collect()
    os.environ.pop("SAD_LIBRARY_PATH", None)
    if failures:
        raise RuntimeError("correctness failures: " + ", ".join(failures))
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
