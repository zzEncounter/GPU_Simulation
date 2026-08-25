"""End-to-end HEA checks for the RX/RY deep-dive variants."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
SAD_PYTHON = SAD_ROOT / "python"
sys.path.insert(0, str(SAD_PYTHON))
from sad_baseline import energy_and_grad  # noqa: E402


OUTPUT = ROOT / "benchmark" / "results" / "rotation_full_circuit.csv"
PERSISTENT = ("-DSAD_ROTATION_PERSISTENT=1",)


@dataclass(frozen=True)
class Variant:
    name: str
    target: str
    flags: tuple[str, ...] = ()


BASE = Variant(
    "128x4-persistent",
    "build/libsad_research_persistent.so",
    PERSISTENT,
)
NONPERSISTENT = Variant(
    "128x4-nonpersistent",
    "build/libsad_nonpersistent.so",
    ("-DSAD_ROTATION_PERSISTENT=0",),
)
NO_MAILBOX = Variant(
    "32x16-no-mailbox",
    "build/libsad_32r4.so",
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
    "build/libsad_research_64r4.so",
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
    "build/libsad_research_64r4_half.so",
    FULL_64R4.flags + ("-DSAD_MAILBOX_CHUNKS=2",),
)
FULL_64R5 = Variant(
    "64x32-full",
    "build/libsad_research_64r5.so",
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
    "build/libsad_research_64r5_half.so",
    FULL_64R5.flags + ("-DSAD_MAILBOX_CHUNKS=2",),
)
QUARTER_64R5 = Variant(
    "64x32-quarter",
    "build/libsad_research_64r5_quarter.so",
    FULL_64R5.flags + ("-DSAD_MAILBOX_CHUNKS=4",),
)


def build(variant: Variant) -> Path:
    command = [
        "make",
        "-C",
        str(SAD_ROOT),
        f"TARGET={variant.target}",
    ]
    if variant.flags:
        command.append(f"EXTRA_NVCCFLAGS={' '.join(variant.flags)}")
    subprocess.run(command, cwd=ROOT, check=True)
    return SAD_ROOT / variant.target


def configurations() -> list[tuple[str, Variant, str, int]]:
    rows: list[tuple[str, Variant, str, int]] = []
    for variant in (BASE, NONPERSISTENT):
        for circuit in ("su2-hea", "rzz-hea"):
            for qubits in (20, 24, 26):
                rows.append(("persistent", variant, circuit, qubits))
    for variant in (BASE, NO_MAILBOX, FULL_64R4):
        for circuit in ("su2-hea", "rzz-hea"):
            rows.append(("shared-removal", variant, circuit, 24))
    for variant in (
        FULL_64R4,
        HALF_64R4,
        FULL_64R5,
        HALF_64R5,
        QUARTER_64R5,
    ):
        for circuit in ("su2-hea", "rzz-hea"):
            rows.append(("mailbox-capacity", variant, circuit, 24))
    return rows


def main() -> None:
    variants = {row[1] for row in configurations()}
    paths = {variant: build(variant) for variant in variants}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment",
        "variant",
        "circuit",
        "qubits",
        "layers",
        "steps",
        "forward_median_ms",
        "hamiltonian_median_ms",
        "backward_median_ms",
        "total_median_ms",
        "energy",
        "grad_json",
    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment, variant, circuit, qubits in configurations():
            os.environ["SAD_LIBRARY_PATH"] = str(paths[variant])
            steps = 9 if qubits <= 24 else 5
            result = energy_and_grad(
                circuit=circuit,
                scalability=(qubits, 8),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "experiment": experiment,
                "variant": variant.name,
                "circuit": circuit,
                "qubits": qubits,
                "layers": 8,
                "steps": steps,
                "forward_median_ms": 1e3 * statistics.median(
                    result.forward_times_s
                ),
                "hamiltonian_median_ms": 1e3 * statistics.median(
                    result.hamiltonian_times_s
                ),
                "backward_median_ms": 1e3 * statistics.median(
                    result.backward_times_s
                ),
                "total_median_ms": 1e3 * statistics.median(result.step_times_s),
                "energy": result.energy,
                "grad_json": json.dumps(result.grad.tolist()),
            }
            writer.writerow(row)
            stream.flush()
            print(
                experiment,
                variant.name,
                circuit,
                qubits,
                f"{row['total_median_ms']:.3f} ms",
                flush=True,
            )
    print(f"CSV written to {OUTPUT}")


if __name__ == "__main__":
    main()
