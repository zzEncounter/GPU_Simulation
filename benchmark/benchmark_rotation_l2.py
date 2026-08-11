"""Measure whole-state L2 reuse and an explicit low-qubit blocking prototype."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_rotation_l2.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "rotation_l2.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")

FIELDS = (
    "mode",
    "policy",
    "qubits",
    "chunk_bits",
    "low_targets",
    "chunks",
    "baseline_phases",
    "low_phases",
    "high_phases",
    "state_bytes",
    "chunk_bytes",
    "average_ms",
    "verification_max_error",
)


def configurations() -> list[tuple[str, int, str, int, int, str]]:
    rows: list[tuple[str, int, str, int, int, str]] = []
    for qubits in (20, 21, 22, 23, 24):
        for policy in ("normal", "persist"):
            rows.append(("capacity", qubits, "baseline", 0, 0, policy))

    for qubits in (22, 23, 24, 25, 26):
        rows.append(("phase-aligned", qubits, "baseline", 0, 0, "normal"))
        for chunk_bits in range(18, min(23, qubits) + 1):
            rows.append(
                ("phase-aligned", qubits, "blocked", chunk_bits, 18,
                 "normal")
            )
        for chunk_bits in range(20, min(22, qubits) + 1):
            rows.append(
                ("phase-aligned", qubits, "blocked", chunk_bits, 18,
                 "persist")
            )

    for low_targets in (19, 20, 21, 22):
        for policy in ("normal", "persist"):
            rows.append(
                ("exact-cutoff", 24, "blocked", 22, low_targets, policy)
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sad-rotation-l2-") as tmp:
        binary = Path(tmp) / "microbench_rotation_l2"
        subprocess.run(
            [
                NVCC,
                "-std=c++17",
                "-O3",
                "-lineinfo",
                "-arch=sm_89",
                f"-I{INCLUDE}",
                str(SOURCE),
                "-o",
                str(binary),
            ],
            check=True,
        )
        output_rows: list[dict[str, str | int | float]] = []
        for suite, qubits, mode, chunk_bits, low_targets, policy in configurations():
            samples: list[float] = []
            parsed: dict[str, str] | None = None
            for repetition in range(args.repetitions):
                completed = subprocess.run(
                    [
                        str(binary),
                        str(qubits),
                        mode,
                        str(chunk_bits),
                        str(low_targets),
                        policy,
                        str(args.iterations),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                values = completed.stdout.strip().split(",")
                parsed = dict(zip(FIELDS, values, strict=True))
                samples.append(float(parsed["average_ms"]))
            assert parsed is not None
            row: dict[str, str | int | float] = {
                "suite": suite,
                **parsed,
                "repetitions": args.repetitions,
                "iterations": args.iterations,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
            output_rows.append(row)
            print(
                f"{suite:13s} q={qubits:2d} {mode:8s} "
                f"chunk={chunk_bits:2d} low={low_targets:2d} "
                f"{policy:7s} median={row['median_ms']:.6f} ms",
                flush=True,
            )

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=output_rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
