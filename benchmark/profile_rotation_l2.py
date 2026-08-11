"""Collect per-phase L2 and DRAM counters with application-realistic caches."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_rotation_l2.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "rotation_l2_ncu.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
NCU = os.environ.get("SAD_NCU", "ncu")
METRICS = (
    "lts__t_sector_hit_rate.pct",
    "lts__t_sector_op_read_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
)


def configurations() -> list[tuple[str, int, str, int, int, str, int]]:
    rows = [
        ("capacity", qubits, "baseline", 0, 0, "normal", 3)
        for qubits in (20, 21, 22, 23, 24)
    ]
    rows.extend(
        ("hint", qubits, "baseline", 0, 0, "persist", 3)
        for qubits in (22, 23)
    )
    rows.append(("blocked", 24, "blocked", 22, 18, "normal", 9))
    return rows


def parse_csv(stdout: str) -> list[dict[str, str]]:
    lines = stdout.splitlines()
    header = next(
        index for index, line in enumerate(lines) if line.startswith('"ID"')
    )
    return list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str | int]] = []

    with tempfile.TemporaryDirectory(prefix="sad-rotation-l2-ncu-") as tmp:
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
        for suite, qubits, mode, chunk_bits, low_targets, policy, launches in configurations():
            completed = subprocess.run(
                [
                    NCU,
                    "--replay-mode",
                    "application",
                    "--cache-control",
                    "none",
                    "--clock-control",
                    "base",
                    "--launch-skip",
                    str(3 * launches),
                    "--launch-count",
                    str(launches),
                    "--kill",
                    "yes",
                    "--csv",
                    "--metrics",
                    ",".join(METRICS),
                    str(binary),
                    str(qubits),
                    mode,
                    str(chunk_bits),
                    str(low_targets),
                    policy,
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = parse_csv(completed.stdout)
            for row in rows:
                output_rows.append(
                    {
                        "suite": suite,
                        "qubits": qubits,
                        "mode": mode,
                        "chunk_bits": chunk_bits,
                        "low_targets": low_targets,
                        "policy": policy,
                        "launch_id": row["ID"],
                        "grid_size": row["Grid Size"],
                        "metric_name": row["Metric Name"],
                        "metric_unit": row["Metric Unit"],
                        "metric_value": row["Metric Value"],
                    }
                )
            print(
                f"profiled {suite} q={qubits} {mode} {policy}: "
                f"{launches} launches",
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
