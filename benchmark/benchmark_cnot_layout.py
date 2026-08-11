"""Compare ring-CNOT gather/scatter and one-buffer in-place execution."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_cnot.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "cnot_layout.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
FIELDS = (
    "mode",
    "qubits",
    "state_bytes",
    "average_ms",
    "effective_gib_per_second",
    "verification_max_error",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sad-cnot-layout-") as tmp:
        binary = Path(tmp) / "microbench_cnot"
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
        for qubits in (20, 22, 24, 26):
            for mode in (
                "copy",
                "gather",
                "scatter",
                "gather-adjoint",
                "scatter-adjoint",
                "dual",
                "dual-scatter",
                "dual-adjoint",
                "dual-scatter-adjoint",
                "inplace",
            ):
                samples: list[float] = []
                bandwidths: list[float] = []
                parsed: dict[str, str] | None = None
                for _ in range(args.repetitions):
                    completed = subprocess.run(
                        [str(binary), str(qubits), mode, str(args.iterations)],
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
                    bandwidths.append(
                        float(parsed["effective_gib_per_second"])
                    )
                assert parsed is not None
                row: dict[str, str | int | float] = {
                    **parsed,
                    "repetitions": args.repetitions,
                    "iterations": args.iterations,
                    "median_ms": statistics.median(samples),
                    "median_effective_gib_per_second": statistics.median(
                        bandwidths
                    ),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
                output_rows.append(row)
                print(
                    f"q={qubits} {mode:7s} median={row['median_ms']:.6f} ms "
                    f"effective={row['median_effective_gib_per_second']:.1f} GiB/s",
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
