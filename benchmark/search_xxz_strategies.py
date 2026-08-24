"""Near-exhaustive XX/YY/ZZ bond-tile and phase-partition search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import os
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_xxz.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "xxz_search_raw.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
BUILD_CACHE = ROOT / "sad" / "build" / "xxz_search"
FIELDS = (
    "component", "direction", "qubits", "parity", "threads", "register_amplitudes",
    "tile_bits", "phase_count", "static_shared_bytes",
    "dynamic_shared_bytes", "registers_per_thread", "active_cta_per_sm",
    "average_ms", "local_bytes_per_thread", "multiprocessors",
    "iterations", "phase_pair_counts",
)
PREFIX = ("stage", "variant", "candidate", "repetition")
COMPONENT_MODES = {"xx+yy+zz": 0, "xx": 1, "yy": 2}


@dataclass(frozen=True, order=True)
class Variant:
    threads: int
    register_bits: int

    @property
    def tile_bits(self) -> int:
        return 5 + self.register_bits + int(math.log2(self.threads // 32))

    @property
    def name(self) -> str:
        return f"t{self.threads}r{self.register_bits}"

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            f"-DSAD_FORWARD_BLOCK_THREADS={self.threads}",
            f"-DSAD_FORWARD_REGISTER_BITS={self.register_bits}",
            f"-DSAD_BLOCK_THREADS={self.threads}",
            f"-DSAD_REGISTER_BITS={self.register_bits}",
            "-DSAD_XXZ_PERSISTENT=0",
        )


VARIANTS = tuple(
    Variant(threads, register_bits)
    for threads in (32, 64, 128, 256, 512)
    for register_bits in range(2, 7)
    if 7
    <= 5 + register_bits + int(math.log2(threads // 32))
    <= 11
)


def compositions(total: int, parts: int, capacity: int) -> Iterable[tuple[int, ...]]:
    """Enumerate every positive pair-count composition under tile capacity."""

    if parts == 0:
        if total == 0:
            yield ()
        return
    minimum_tail = parts - 1
    for value in range(1, min(capacity, total - minimum_tail) + 1):
        for tail in compositions(total - value, parts - 1, capacity):
            yield (value, *tail)


def candidate_partitions(qubits: int, tile_bits: int) -> tuple[tuple[int, ...], ...]:
    bonds = qubits // 2
    capacity = tile_bits // 2
    minimum = math.ceil(bonds / capacity)
    # Every minimum-phase composition plus representative +1 phase cases.
    exact = list(compositions(bonds, minimum, capacity))
    extra = list(compositions(bonds, minimum + 1, capacity))
    if len(extra) > 12:
        extra.sort(
            key=lambda values: (
                max(values) - min(values),
                sum(abs(a - b) for a, b in zip(values, values[1:])),
                values,
            )
        )
        extra = extra[:6] + extra[-6:]
    return tuple(dict.fromkeys((*exact, *extra)))


def _latest_source_mtime() -> float:
    return max(
        file.stat().st_mtime
        for file in (SOURCE, *INCLUDE.glob("**/*.cuh"))
    )


def _cached_binary(variant: Variant, component: str) -> Path:
    flags = (*variant.flags, f"-DSAD_XXZ_COMPONENT_MODE={COMPONENT_MODES[component]}")
    digest = hashlib.sha256("\0".join(flags).encode()).hexdigest()[:12]
    return BUILD_CACHE / f"{variant.name}-{component.replace('+', '_')}-{digest}"


def _compile(variant: Variant, component: str, binary: Path) -> None:
    if binary.exists() and binary.stat().st_mtime >= _latest_source_mtime():
        return
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            NVCC, "-O3", "-std=c++17", "-arch=native", "-lineinfo",
            f"-I{INCLUDE}", *variant.flags,
            f"-DSAD_XXZ_COMPONENT_MODE={COMPONENT_MODES[component]}",
            str(SOURCE), "-o", str(binary),
        ],
        cwd=ROOT,
        check=True,
    )


def _measure(
    binary: Path,
    q: int,
    direction: str,
    parity: int,
    iterations: int,
    partition: tuple[int, ...],
) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(binary), str(q), direction, str(parity), str(iterations),
            ",".join(map(str, partition)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()[-1].split(",")
    if len(values) != len(FIELDS):
        raise RuntimeError(completed.stdout)
    return dict(zip(FIELDS, values, strict=True))


def _parse_qubits(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item < 4 or item > 30 or item & 1 for item in result):
        raise argparse.ArgumentTypeError("qubits must be even comma-separated 4..30")
    return result


def _parse_components(value: str) -> tuple[str, ...]:
    result = tuple(value.split(","))
    if not result or any(item not in COMPONENT_MODES for item in result):
        raise argparse.ArgumentTypeError(
            "components must be comma-separated xx+yy+zz,xx,yy"
        )
    return result


def _completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            tuple(row[field] for field in ("stage", "component", "variant", "qubits", "direction", "parity", "candidate", "repetition", "iterations"))
            for row in csv.DictReader(stream)
        }


def _existing_shape_samples(
    path: Path, iterations: int
) -> dict[tuple[Variant, str, int, str, int], list[float]]:
    by_name = {variant.name: variant for variant in VARIANTS}
    result: dict[tuple[Variant, str, int, str, int], list[float]] = {}
    if not path.exists():
        return result
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] != "shape" or int(row["iterations"]) != iterations:
                continue
            key = (
                by_name[row["variant"]],
                row["component"],
                int(row["qubits"]),
                row["direction"],
                int(row["parity"]),
            )
            result.setdefault(key, []).append(float(row["average_ms"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=_parse_qubits, default=(16, 20, 24, 26, 28))
    parser.add_argument(
        "--components", type=_parse_components,
        default=tuple(COMPONENT_MODES),
    )
    parser.add_argument(
        "--partition-components",
        type=_parse_components,
        default=("xx+yy+zz",),
        help=(
            "components receiving exhaustive phase-partition search; all "
            "--components still receive the complete shape search"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--shape-survivors", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.repetitions, args.iterations, args.shape_survivors) < 1:
        parser.error("counts must be positive")
    if any(item not in args.components for item in args.partition_components):
        parser.error("partition components must be included in --components")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    medians: dict[
        tuple[str, str, str, int], list[tuple[float, Variant]]
    ] = {}
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(*PREFIX, *FIELDS), lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        binaries: dict[tuple[Variant, str], Path] = {}
        for variant, component in itertools.product(VARIANTS, args.components):
            print(f"compile {variant.name}/{component}", flush=True)
            binary = _cached_binary(variant, component)
            _compile(variant, component, binary)
            binaries[variant, component] = binary
        written = 0
        shape_jobs = list(
            itertools.product(
                VARIANTS, args.components, args.qubits,
                ("forward", "backward"), (0, 1)
            )
        )
        samples_by_shape = _existing_shape_samples(args.output, args.iterations)
        for job in shape_jobs:
            samples_by_shape.setdefault(job, [])
        for repetition in range(args.repetitions):
            ordered = list(shape_jobs)
            random.Random(f"sad-xxz-shape-{repetition}").shuffle(ordered)
            for variant, component, q, direction, parity in ordered:
                capacity = variant.tile_bits // 2
                packed = tuple(
                    min(capacity, q // 2 - offset)
                    for offset in range(0, q // 2, capacity)
                )
                candidate = ":".join(map(str, packed))
                key = (
                    "shape", component, variant.name, str(q), direction,
                    str(parity), candidate,
                    str(repetition), str(args.iterations),
                )
                if key in completed:
                    continue
                measured = _measure(
                    binaries[variant, component], q, direction, parity,
                    args.iterations, packed,
                )
                samples_by_shape[variant, component, q, direction, parity].append(
                    float(measured["average_ms"])
                )
                writer.writerow(
                    {
                        "stage": "shape", "variant": variant.name,
                        "candidate": candidate, "repetition": repetition,
                        **measured,
                    }
                )
                stream.flush()
                completed.add(key)
                written += 1
        for (
            variant, component, q, direction, parity
        ), samples in samples_by_shape.items():
            medians.setdefault((component, direction, str(q), parity), []).append(
                (statistics.median(samples), variant)
            )
        selected = {
            scenario: tuple(
                variant for _, variant in sorted(candidates)[: args.shape_survivors]
            )
            for scenario, candidates in medians.items()
        }
        print(
            "compiled survivor union:",
            ", ".join(
                item.name
                for item in sorted(
                    {v for values in selected.values() for v in values}
                )
            ),
        )
        partition_jobs = [
            (variant, component, q, direction, parity, partition)
            for component, q, direction, parity in itertools.product(
                args.partition_components,
                args.qubits,
                ("forward", "backward"),
                (0, 1),
            )
            for variant in selected[component, direction, str(q), parity]
            for partition in candidate_partitions(q, variant.tile_bits)
        ]
        for repetition in range(args.repetitions):
            ordered = list(partition_jobs)
            random.Random(f"sad-xxz-partition-{repetition}").shuffle(ordered)
            for variant, component, q, direction, parity, partition in ordered:
                candidate = ":".join(map(str, partition))
                key = (
                    "partition", component, variant.name, str(q), direction,
                    str(parity), candidate, str(repetition), str(args.iterations),
                )
                if key in completed:
                    continue
                measured = _measure(
                    binaries[variant, component], q, direction, parity,
                    args.iterations, partition,
                )
                writer.writerow(
                    {
                        "stage": "partition", "variant": variant.name,
                        "candidate": candidate, "repetition": repetition,
                        **measured,
                    }
                )
                stream.flush()
                completed.add(key)
                written += 1
    print(f"wrote {written} raw rows to {args.output}")


if __name__ == "__main__":
    main()
