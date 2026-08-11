"""Compare odd/even XXZ tile sizes and pair capacities."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_xxz.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "xxz_tiles.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
FIELDS = (
    "direction",
    "qubits",
    "parity",
    "threads",
    "register_amplitudes",
    "tile_bits",
    "phase_count",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "registers_per_thread",
    "active_cta_per_sm",
    "average_ms",
)


@dataclass(frozen=True)
class Variant:
    name: str
    threads: int
    register_bits: int

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            f"-DSAD_FORWARD_BLOCK_THREADS={self.threads}",
            f"-DSAD_FORWARD_REGISTER_BITS={self.register_bits}",
            f"-DSAD_BLOCK_THREADS={self.threads}",
            f"-DSAD_REGISTER_BITS={self.register_bits}",
        )


VARIANTS = (
    Variant("32x8-tile8", 32, 3),
    Variant("64x4-tile8", 64, 2),
    Variant("128x4-tile9-current", 128, 2),
    Variant("64x16-tile10", 64, 4),
    Variant("128x8-tile10", 128, 3),
    Variant("256x4-tile10", 256, 2),
    Variant("128x16-tile11", 128, 4),
    # Tile 12 is intentionally excluded: the backward mailbox plus reduction
    # needs 0x10410 bytes of static shared memory, above ptxas' 0xc000 limit.
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str | int | float]] = []

    with tempfile.TemporaryDirectory(prefix="sad-xxz-tiles-") as tmp:
        tmp_path = Path(tmp)
        for variant in VARIANTS:
            binary = tmp_path / variant.name
            subprocess.run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-lineinfo",
                    "-arch=sm_89",
                    f"-I{INCLUDE}",
                    *variant.flags,
                    str(SOURCE),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            for qubits in (20, 24, 26):
                for direction in ("forward", "backward"):
                    for parity in (0, 1):
                        samples: list[float] = []
                        parsed: dict[str, str] | None = None
                        for _ in range(args.repetitions):
                            completed = subprocess.run(
                                [
                                    str(binary),
                                    str(qubits),
                                    direction,
                                    str(parity),
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
                            "variant": variant.name,
                            **parsed,
                            "repetitions": args.repetitions,
                            "iterations": args.iterations,
                            "median_ms": statistics.median(samples),
                            "min_ms": min(samples),
                            "max_ms": max(samples),
                        }
                        output_rows.append(row)
                        print(
                            f"{variant.name:24s} {direction:8s} "
                            f"q={qubits} p={parity} "
                            f"phases={parsed['phase_count']} "
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
