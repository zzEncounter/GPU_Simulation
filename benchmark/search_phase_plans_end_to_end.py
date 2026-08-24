"""Paired end-to-end confirmation of sparse nonuniform phase finalists.

The candidate plans come from the calibrated RX/RY schedule screen.  Forward,
backward, and the joint plan are timed separately against the canonical map in
the same F32r2/B32r2 binary, so the effect is not mixed with shape, mailbox,
diagonal, or fusion changes.  Rows are append/resume safe.
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
from sad_baseline import energy_and_grad  # noqa: E402
from sad_baseline import runner as sad_runner  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "phase_plan_paired_raw.csv"
)
VARIANT = "f32r2_b32r2"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa")
FIELDS = (
    "repetition",
    "order",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "steps",
    "candidate",
    "variant",
    "forward_phase_plan",
    "backward_phase_plan",
    "candidate_median_ms",
    "canonical_median_ms",
    "canonical_over_candidate",
    "candidate_forward_mean_ms",
    "canonical_forward_mean_ms",
    "canonical_over_candidate_forward",
    "candidate_hamiltonian_mean_ms",
    "canonical_hamiltonian_mean_ms",
    "candidate_backward_mean_ms",
    "canonical_backward_mean_ms",
    "canonical_over_candidate_backward",
    "energy_abs_error",
    "gradient_max_abs_error",
    "correct",
)


@dataclass(frozen=True)
class PlanPair:
    forward: str
    backward: str


def _gate(circuit: str) -> str:
    return "ry" if circuit in {"ra-hea", "su2-hea"} else "rx"


def finalists(circuit: str, qubits: int) -> dict[str, PlanPair]:
    gate = _gate(circuit)
    if qubits == 10 and gate == "rx":
        forward = "compact:L2R2W0-L4R2W0"
        backward = "compact:L4R2W0-L2R2W0"
    elif qubits == 10:
        forward = "compact:L4R0W0-L5R1W0"
        backward = "compact:L2R2W0-L4R2W0"
    elif qubits == 12 and gate == "rx":
        forward = backward = "compact:L4R2W0-L4R2W0"
    elif qubits == 12:
        forward = "compact:L5R1W0-L5R1W0"
        backward = "compact:L4R2W0-L4R2W0"
    else:
        raise ValueError(f"no sparse phase finalists for {circuit} {qubits}q")
    return {
        "forward": PlanPair(forward, ""),
        "backward": PlanPair("", backward),
        "joint": PlanPair(forward, backward),
    }


def _run(
    circuit: str,
    qubits: int,
    layers: int,
    steps: int,
    plans: PlanPair,
):
    return energy_and_grad(
        circuit=circuit,
        random_seed=42,
        scalability=(qubits, layers),
        precision="float64",
        steps=steps,
        warmup_steps=5,
        forward_phase_plan=plans.forward,
        backward_phase_plan=plans.backward,
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
                row["candidate"],
            )
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.layers, args.steps, args.rounds) < 1:
        parser.error("layers, steps, and rounds must be positive")

    library = sad_runner._VARIANT_LIBRARIES[VARIANT]
    sad_runner._build_library(library)
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    failures: list[str] = []
    scenarios = tuple((circuit, q) for circuit in CIRCUITS for q in (10, 12))
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            shuffled = list(scenarios)
            random.Random(20260817 + repetition).shuffle(shuffled)
            for circuit, qubits in shuffled:
                candidates = finalists(circuit, qubits)
                pending = {
                    name: plans
                    for name, plans in candidates.items()
                    if (repetition, circuit, qubits, name) not in done
                }
                if not pending:
                    continue
                order = ["canonical", *pending]
                random.Random(
                    f"phase-e2e-{repetition}-{circuit}-{qubits}"
                ).shuffle(order)
                measured = {
                    name: _run(
                        circuit,
                        qubits,
                        args.layers,
                        args.steps,
                        PlanPair("", "") if name == "canonical" else pending[name],
                    )
                    for name in order
                }
                canonical = measured["canonical"]
                for name, plans in pending.items():
                    result = measured[name]
                    candidate_ms = 1e3 * result.median_step_time_s
                    canonical_ms = 1e3 * canonical.median_step_time_s
                    candidate_forward = 1e3 * statistics.fmean(
                        result.forward_times_s
                    )
                    canonical_forward = 1e3 * statistics.fmean(
                        canonical.forward_times_s
                    )
                    candidate_hamiltonian = 1e3 * statistics.fmean(
                        result.hamiltonian_times_s
                    )
                    canonical_hamiltonian = 1e3 * statistics.fmean(
                        canonical.hamiltonian_times_s
                    )
                    candidate_backward = 1e3 * statistics.fmean(
                        result.backward_times_s
                    )
                    canonical_backward = 1e3 * statistics.fmean(
                        canonical.backward_times_s
                    )
                    energy_error = abs(result.energy - canonical.energy)
                    gradient_error = float(
                        np.max(np.abs(result.grad - canonical.grad))
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
                            "candidate": name,
                            "variant": VARIANT,
                            "forward_phase_plan": plans.forward,
                            "backward_phase_plan": plans.backward,
                            "candidate_median_ms": candidate_ms,
                            "canonical_median_ms": canonical_ms,
                            "canonical_over_candidate": canonical_ms / candidate_ms,
                            "candidate_forward_mean_ms": candidate_forward,
                            "canonical_forward_mean_ms": canonical_forward,
                            "canonical_over_candidate_forward": (
                                canonical_forward / candidate_forward
                            ),
                            "candidate_hamiltonian_mean_ms": candidate_hamiltonian,
                            "canonical_hamiltonian_mean_ms": canonical_hamiltonian,
                            "candidate_backward_mean_ms": candidate_backward,
                            "canonical_backward_mean_ms": canonical_backward,
                            "canonical_over_candidate_backward": (
                                canonical_backward / candidate_backward
                            ),
                            "energy_abs_error": energy_error,
                            "gradient_max_abs_error": gradient_error,
                            "correct": int(correct),
                        }
                    )
                    stream.flush()
                    done.add((repetition, circuit, qubits, name))
                    print(
                        f"{circuit:8s} {qubits}q {name:8s}: "
                        f"{canonical_ms / candidate_ms:.3f}x correct={correct}",
                        flush=True,
                    )
                    if not correct:
                        failures.append(f"{circuit} {qubits}q {name}")
                gc.collect()
    os.environ.pop("SAD_LIBRARY_PATH", None)
    if failures:
        raise RuntimeError("correctness failures: " + ", ".join(failures))
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
