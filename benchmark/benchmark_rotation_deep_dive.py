"""Compile and run the RX/RY deep-dive ablations.

The output is intentionally raw (one row per independent process run).  The
report uses medians grouped by experiment/variant/layout/direction/qubits.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_rotation.cu"
INCLUDE = ROOT / "sad" / "src"
OUTPUT = ROOT / "benchmark" / "results" / "rotation_deep_dive.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
PERSISTENT = ("-DSAD_ROTATION_PERSISTENT=1",)

MICRO_FIELDS = (
    "gate",
    "layout",
    "direction",
    "qubits",
    "threads",
    "register_amplitudes",
    "tile_bits",
    "phase_count",
    "gate_count",
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "active_cta_per_sm",
    "average_ms",
    "ms_per_gate",
    "mailbox_bytes",
    "mailbox_chunks",
    "scalar_mailbox",
    "persistent",
    "legacy_reduction",
)


@dataclass(frozen=True)
class Variant:
    name: str
    flags: tuple[str, ...] = ()


BASE = Variant("128x4-full", PERSISTENT)
NO_MAILBOX_SMALL = Variant(
    "32x4-no-mailbox",
    PERSISTENT
    + ("-DSAD_FORWARD_BLOCK_THREADS=32", "-DSAD_BLOCK_THREADS=32"),
)
NO_MAILBOX_SAME_TILE = Variant(
    "32x16-no-mailbox",
    PERSISTENT
    + (
        "-DSAD_FORWARD_BLOCK_THREADS=32",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=4",
    ),
)
FULL_64R4 = Variant(
    "64x16-full",
    PERSISTENT
    + (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
)
HALF_64R4 = Variant(
    "64x16-half",
    FULL_64R4.flags + ("-DSAD_MAILBOX_CHUNKS=2",),
)
QUARTER_64R4 = Variant(
    "64x16-quarter",
    FULL_64R4.flags + ("-DSAD_MAILBOX_CHUNKS=4",),
)
SCALAR_64R4 = Variant(
    "64x16-scalar-ry",
    FULL_64R4.flags + ("-DSAD_RY_SCALAR_MAILBOX=1",),
)
FULL_64R5 = Variant(
    "64x32-full",
    PERSISTENT
    + (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=5",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=5",
    ),
)
HALF_64R5 = Variant(
    "64x32-half",
    FULL_64R5.flags + ("-DSAD_MAILBOX_CHUNKS=2",),
)
QUARTER_64R5 = Variant(
    "64x32-quarter",
    FULL_64R5.flags + ("-DSAD_MAILBOX_CHUNKS=4",),
)
SCALAR_64R5 = Variant(
    "64x32-scalar-ry",
    FULL_64R5.flags + ("-DSAD_RY_SCALAR_MAILBOX=1",),
)
NONPERSISTENT = Variant(
    "128x4-nonpersistent", ("-DSAD_ROTATION_PERSISTENT=0",)
)
HIERARCHICAL = Variant(
    "128x4-hierarchical-reduction",
    PERSISTENT + ("-DSAD_LEGACY_BLOCK_REDUCTION=0",),
)
HIERARCHICAL_64R4 = Variant(
    "64x16-hierarchical-reduction",
    FULL_64R4.flags + ("-DSAD_LEGACY_BLOCK_REDUCTION=0",),
)
WARP_ATOMIC = Variant(
    "128x4-warp-atomic-reduction",
    PERSISTENT + ("-DSAD_ROTATION_WARP_ATOMIC=1",),
)
WARP_ATOMIC_64R4 = Variant(
    "64x16-warp-atomic-reduction",
    FULL_64R4.flags + ("-DSAD_ROTATION_WARP_ATOMIC=1",),
)


def configurations(suites: set[str]) -> list[tuple[str, Variant, int, str, str, str]]:
    rows: list[tuple[str, Variant, int, str, str, str]] = []
    if "shared-removal" in suites:
        for variant in (BASE, NO_MAILBOX_SMALL, NO_MAILBOX_SAME_TILE):
            for qubits, gates in ((20, ("ry",)), (24, ("rx", "ry"))):
                for gate in gates:
                    for direction in ("forward", "backward"):
                        rows.append(
                            ("shared-removal", variant, qubits, gate,
                             "full-fixed", direction)
                        )
    if "mailbox-capacity" in suites:
        for variants in (
            (FULL_64R4, HALF_64R4, QUARTER_64R4, SCALAR_64R4),
            (FULL_64R5, HALF_64R5, QUARTER_64R5, SCALAR_64R5),
        ):
            for variant in variants:
                gates = ("ry",) if "scalar" in variant.name else ("rx", "ry")
                for gate in gates:
                    for direction in ("forward", "backward"):
                        rows.append(
                            ("mailbox-capacity", variant, 24, gate,
                             "full-fixed", direction)
                        )
    if "warp-continuity" in suites:
        for variant in (BASE, FULL_64R4):
            for qubits in (20, 24, 26):
                for layout in ("full-fixed", "full-pairs"):
                    for direction in ("forward", "backward"):
                        rows.append(
                            ("warp-continuity", variant, qubits, "ry",
                             layout, direction)
                        )
            for layout in ("full-fixed", "full-pairs"):
                for direction in ("forward", "backward"):
                    rows.append(
                        ("warp-continuity", variant, 24, "rx", layout,
                         direction)
                    )
    if "persistent" in suites:
        for variant in (BASE, NONPERSISTENT):
            for qubits in (12, 20, 24):
                for layout in ("full", "full-fixed"):
                    for direction in ("forward", "backward"):
                        rows.append(
                            ("persistent", variant, qubits, "ry", layout,
                             direction)
                        )
            for layout in ("full", "full-fixed"):
                for direction in ("forward", "backward"):
                    rows.append(
                        ("persistent", variant, 24, "rx", layout, direction)
                    )
    if "reduction" in suites or "reduction-extended" in suites:
        variants = (
            (BASE, HIERARCHICAL, FULL_64R4, HIERARCHICAL_64R4)
            if "reduction-extended" not in suites
            else (
                BASE,
                HIERARCHICAL,
                WARP_ATOMIC,
                FULL_64R4,
                HIERARCHICAL_64R4,
                WARP_ATOMIC_64R4,
            )
        )
        for variant in variants:
            for qubits in (20, 24):
                for gate in ("rx", "ry"):
                    rows.append(
                        ("reduction", variant, qubits, gate, "full-fixed",
                         "backward")
                    )
    return rows


def compile_variant(variant: Variant, binary: Path, iterations: int) -> None:
    command = [
        NVCC,
        "-O3",
        "-std=c++17",
        "-arch=native",
        "-lineinfo",
        f"-I{INCLUDE}",
        f"-DSAD_MICRO_ITERATIONS={iterations}",
        *variant.flags,
        str(SOURCE),
        "-o",
        str(binary),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        action="append",
        choices=(
            "shared-removal",
            "mailbox-capacity",
            "warp-continuity",
            "persistent",
            "reduction",
            "reduction-extended",
        ),
        help="repeat to select suites; default runs all",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repetitions < 1 or args.iterations < 1:
        parser.error("repetitions and iterations must be positive")
    suites = set(args.suite or (
        "shared-removal",
        "mailbox-capacity",
        "warp-continuity",
        "persistent",
        "reduction",
    ))
    configs = configurations(suites)
    variants = {config[1] for config in configs}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fields = ("experiment", "variant", "repetition", *MICRO_FIELDS)
    with tempfile.TemporaryDirectory(prefix="sad-rotation-") as directory:
        build_dir = Path(directory)
        binaries: dict[Variant, Path] = {}
        for index, variant in enumerate(sorted(variants, key=lambda item: item.name)):
            binary = build_dir / f"micro_{index}"
            print(f"compile {variant.name}", flush=True)
            compile_variant(variant, binary, args.iterations)
            binaries[variant] = binary

        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for repetition in range(args.repetitions):
                # Rotate the configuration order to reduce thermal/order bias.
                offset = repetition % max(1, len(configs))
                ordered = configs[offset:] + configs[:offset]
                for experiment, variant, qubits, gate, layout, direction in ordered:
                    completed = subprocess.run(
                        [str(binaries[variant]), str(qubits), gate, layout,
                         direction],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    values = completed.stdout.strip().splitlines()[-1].split(",")
                    if len(values) != len(MICRO_FIELDS):
                        raise RuntimeError(
                            f"unexpected output from {variant.name}: {completed.stdout}"
                        )
                    row = dict(zip(MICRO_FIELDS, values, strict=True))
                    writer.writerow(
                        {
                            "experiment": experiment,
                            "variant": variant.name,
                            "repetition": repetition,
                            **row,
                        }
                    )
                    stream.flush()
                    print(
                        experiment,
                        variant.name,
                        qubits,
                        gate,
                        layout,
                        direction,
                        row["average_ms"],
                        flush=True,
                    )
    print(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
