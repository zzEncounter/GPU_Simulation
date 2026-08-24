"""Direction-only layer timing for joint-policy forward regressions.

All shapes run in the same multi-shape executable and operate on the same
kind of complete RX/RY layer.  This removes backward and wall time from shape
selection; end-to-end circuit timing is a later confirmation step.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from benchmark_heterogeneous_phases import ROOT, _build, _measure, _uniform_schedule


DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "directional_rotation_shape_raw.csv"
)


@dataclass(frozen=True)
class Scenario:
    circuit: str
    gate: str
    qubits: int
    direction: str
    baseline: str
    joint: str
    family: str = "compact"


SCENARIOS = (
    Scenario("ra-hea", "ry", 6, "forward", "t128r2m1", "t32r2m1"),
    Scenario("su2-hea", "ry", 8, "forward", "t128r2m1", "t64r2m1"),
    Scenario("su2-hea", "ry", 14, "forward", "t128r2m1", "t32r2m1"),
    Scenario("rzz-hea", "rx", 6, "forward", "t128r2m1", "t32r2m1"),
    Scenario("qaoa", "rx", 6, "forward", "t128r2m1", "t32r2m1"),
    Scenario("ra-hea", "ry", 22, "forward", "t128r2m1", "t64r4m1"),
    Scenario("su2-hea", "ry", 22, "forward", "t128r2m1", "t128r3m1"),
    Scenario("su2-hea", "ry", 24, "forward", "t128r2m1", "t128r3m1"),
    Scenario("rzz-hea", "rx", 24, "forward", "t128r2m1", "t64r3m1"),
    Scenario("rzz-hea", "rx", 26, "forward", "t128r2m1", "t64r4m1"),
    Scenario("qaoa", "rx", 18, "forward", "t128r2m1", "t32r2m1"),
    Scenario("qaoa", "rx", 26, "forward", "t128r2m1", "t64r4m1"),
    # Backward halves of the direction-independent mixed variants.  RA/SU2
    # preserve physical low-5 lane bits after the first compact phase, while
    # RZZ/QAOA use compact phases throughout.
    Scenario("ra-hea", "ry", 22, "backward", "t128r2m1", "t64r4m1", "fixed"),
    Scenario("su2-hea", "ry", 22, "backward", "t128r2m1", "t64r4m1", "fixed"),
    Scenario("su2-hea", "ry", 24, "backward", "t128r2m1", "t64r4m1", "fixed"),
    Scenario("rzz-hea", "rx", 24, "backward", "t128r2m1", "t64r4m1"),
    Scenario("rzz-hea", "rx", 26, "backward", "t128r2m1", "t128r3m1"),
    Scenario("qaoa", "rx", 18, "backward", "t128r2m1", "t32r2m1"),
    Scenario("qaoa", "rx", 26, "backward", "t128r2m1", "t128r3m1"),
)

FIELDS = (
    "repetition",
    "order",
    "circuit",
    "gate",
    "qubits",
    "direction",
    "candidate",
    "variant",
    "schedule",
    "average_ms",
    "baseline_average_ms",
    "baseline_over_candidate",
    "phi_checksum_abs_error",
    "lambda_checksum_abs_error",
    "gradient_checksum_abs_error",
    "correct",
)


def _completed(path: Path) -> set[tuple[int, str, int, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (
                int(row["repetition"]),
                row["circuit"],
                int(row["qubits"]),
                row["direction"],
                row["candidate"],
            )
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.rounds, args.iterations) < 1:
        parser.error("rounds and iterations must be positive")
    binary = _build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            scenarios = list(SCENARIOS)
            random.Random(20260817 + repetition).shuffle(scenarios)
            for scenario in scenarios:
                variants = {"baseline": scenario.baseline, "joint": scenario.joint}
                pending = [
                    name
                    for name in variants
                    if (
                        repetition,
                        scenario.circuit,
                        scenario.qubits,
                        scenario.direction,
                        name,
                    ) not in done
                ]
                if not pending:
                    continue
                order = list(pending)
                random.Random(
                    f"direction-shape-{repetition}-{scenario.circuit}-{scenario.qubits}"
                ).shuffle(order)
                schedules = {
                    name: _uniform_schedule(
                        scenario.qubits, variants[name], scenario.family
                    )
                    for name in variants
                }
                measured = {
                    name: _measure(
                        binary,
                        (scenario.gate, scenario.direction, scenario.qubits),
                        schedules[name],
                        args.iterations,
                    )
                    for name in order
                }
                if "baseline" not in measured:
                    measured["baseline"] = _measure(
                        binary,
                        (scenario.gate, scenario.direction, scenario.qubits),
                        schedules["baseline"],
                        args.iterations,
                    )
                baseline = measured["baseline"]
                for name in order:
                    result = measured[name]
                    errors = tuple(abs(result[index] - baseline[index]) for index in (1, 2, 3))
                    scale = max(1.0, *(abs(baseline[index]) for index in (1, 2, 3)))
                    correct = max(errors) <= 1e-8 * scale
                    writer.writerow(
                        {
                            "repetition": repetition,
                            "order": "-".join(order),
                            "circuit": scenario.circuit,
                            "gate": scenario.gate,
                            "qubits": scenario.qubits,
                            "direction": scenario.direction,
                            "candidate": name,
                            "variant": variants[name],
                            "schedule": schedules[name],
                            "average_ms": result[0],
                            "baseline_average_ms": baseline[0],
                            "baseline_over_candidate": baseline[0] / result[0],
                            "phi_checksum_abs_error": errors[0],
                            "lambda_checksum_abs_error": errors[1],
                            "gradient_checksum_abs_error": errors[2],
                            "correct": int(correct),
                        }
                    )
                    stream.flush()
                    done.add(
                        (
                            repetition,
                            scenario.circuit,
                            scenario.qubits,
                            scenario.direction,
                            name,
                        )
                    )
                    print(
                        f"{scenario.circuit:8s} q={scenario.qubits} {name}: "
                        f"{baseline[0] / result[0]:.3f}x correct={correct}",
                        flush=True,
                    )
                    if not correct:
                        raise RuntimeError(
                            f"checksum mismatch for {scenario.circuit} {scenario.qubits}q"
                        )
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
