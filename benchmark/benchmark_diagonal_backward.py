"""Reduction and CTA geometry sweep for diagonal adjoint kernels."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_diagonal_backward.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "diagonal_backward.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    variants = (
        ("cta-64", 64, 1, 0),
        ("warp-64", 64, 1, 1),
        ("cta-128", 128, 1, 0),
        ("warp-128", 128, 1, 1),
        ("hierarchical-128", 128, 0, 0),
    )
    with tempfile.TemporaryDirectory(prefix="sad-diagonal-backward-") as tmp:
        for name, threads, legacy, warp_atomic in variants:
            binary = Path(tmp) / name
            subprocess.run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-lineinfo",
                    "-arch=sm_89",
                    f"-I{INCLUDE}",
                    f"-DSAD_DIAGONAL_BLOCK_THREADS={threads}",
                    f"-DSAD_SHARED_DIAGONAL_BLOCK_THREADS={threads}",
                    f"-DSAD_LEGACY_BLOCK_REDUCTION={legacy}",
                    f"-DSAD_DIAGONAL_WARP_ATOMIC={warp_atomic}",
                    str(SOURCE),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            for qubits in (20, 24, 26, 28):
                for strategy in ("rz", "combined", "qaoa"):
                    samples: list[float] = []
                    for _ in range(args.repetitions):
                        completed = subprocess.run(
                            [
                                str(binary),
                                str(qubits),
                                strategy,
                                str(args.iterations),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        samples.append(
                            float(completed.stdout.strip().split(",")[-1])
                        )
                    row = {
                        "variant": name,
                        "strategy": strategy,
                        "qubits": qubits,
                        "threads": threads,
                        "legacy_block_reduction": legacy,
                        "diagonal_warp_atomic": warp_atomic,
                        "repetitions": args.repetitions,
                        "iterations": args.iterations,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                    }
                    rows.append(row)
                    print(
                        f"{name:16s} {strategy:8s} q={qubits} "
                        f"{row['median_ms']:.6f} ms",
                        flush=True,
                    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
