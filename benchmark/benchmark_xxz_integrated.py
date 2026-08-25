"""Benchmark dependency-preserving partial fusion of even/odd XXZ matchings."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_xxz_integrated.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "xxz_integrated.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
FIELDS = (
    "direction",
    "mode",
    "qubits",
    "tile_bits",
    "baseline_phases",
    "integrated_phases",
    "average_ms",
    "verification_max_error",
    "gradient_max_error",
)
VARIANTS = {
    "128x4-f9-b9-current": (),
    "128x8-f10-b9": ("-DSAD_FORWARD_REGISTER_BITS=3",),
    "128x8-f10-32x8-b8": (
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=3",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str | int | float]] = []

    with tempfile.TemporaryDirectory(prefix="sad-xxz-integrated-") as tmp:
        for variant, flags in VARIANTS.items():
            binary = Path(tmp) / variant
            subprocess.run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-lineinfo",
                    "-arch=sm_89",
                    f"-I{INCLUDE}",
                    *flags,
                    str(SOURCE),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            for qubits in (20, 24, 26):
                for direction in ("forward", "backward"):
                    for mode in ("baseline", "integrated"):
                        samples: list[float] = []
                        parsed: dict[str, str] | None = None
                        for _ in range(args.repetitions):
                            completed = subprocess.run(
                                [
                                    str(binary),
                                    str(qubits),
                                    direction,
                                    mode,
                                    str(args.iterations),
                                ],
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            parsed = dict(
                                zip(
                                    FIELDS,
                                    completed.stdout.strip().split(","),
                                    strict=True,
                                )
                            )
                            samples.append(float(parsed["average_ms"]))
                        assert parsed is not None
                        row: dict[str, str | int | float] = {
                            "variant": variant,
                            **parsed,
                            "repetitions": args.repetitions,
                            "iterations": args.iterations,
                            "median_ms": statistics.median(samples),
                            "min_ms": min(samples),
                            "max_ms": max(samples),
                        }
                        output_rows.append(row)
                        print(
                            f"{variant:24s} q={qubits} {direction:8s} "
                            f"{mode:10s} "
                            f"median={row['median_ms']:.6f} ms",
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
