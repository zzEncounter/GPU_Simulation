"""Search circuit-specific CNOT/RZ/RZZ/XXZ fusion boundaries end to end.

Every timed candidate is checked against the retained legacy execution path.
The output contains one row per CUDA-event sample and is append/resume safe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import EnergyGradResult, energy_and_grad  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "fusion_search_raw.csv"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
FIELDS = (
    "circuit",
    "qubits",
    "layers",
    "variant",
    "execution_mode",
    "shape",
    "sample",
    "forward_ms",
    "hamiltonian_ms",
    "backward_ms",
    "total_ms",
    "energy",
    "energy_abs_error",
    "gradient_max_abs_error",
    "gradient_l2_error",
    "correct",
    "library",
    "compile_flags",
)


@dataclass(frozen=True)
class Shape:
    name: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class FusionVariant:
    name: str
    execution_mode: str
    flags: tuple[str, ...] = ()


SAFE = Shape("f128r2_b128r2", ())
F64R4_B64R4 = Shape(
    "f64r4_b64r4",
    (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
)
F64R3_B64R4 = Shape(
    "f64r3_b64r4",
    (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
)
F128R3_B64R4 = Shape(
    "f128r3_b64r4",
    (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
)
F64R4_B128R3 = Shape(
    "f64r4_b128r3",
    (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=3",
    ),
)
F128R3_B32R3 = Shape(
    "f128r3_b32r3",
    (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=3",
    ),
)


VARIANTS = {
    "ra-hea": (
        FusionVariant("complex-split", "optimized", ("-DSAD_REAL_AMPLITUDE=0", "-DSAD_RA_FORWARD_FUSED=0", "-DSAD_RA_BACKWARD_FUSED=0")),
        FusionVariant("complex-fused-forward", "optimized", ("-DSAD_REAL_AMPLITUDE=0", "-DSAD_RA_FORWARD_FUSED=1", "-DSAD_RA_BACKWARD_FUSED=0")),
        FusionVariant("complex-fused-backward", "optimized", ("-DSAD_REAL_AMPLITUDE=0", "-DSAD_RA_FORWARD_FUSED=0", "-DSAD_RA_BACKWARD_FUSED=1")),
        FusionVariant("complex-fused-both", "optimized", ("-DSAD_REAL_AMPLITUDE=0", "-DSAD_RA_FORWARD_FUSED=1", "-DSAD_RA_BACKWARD_FUSED=1")),
        FusionVariant("real-fused-both", "optimized", ("-DSAD_REAL_AMPLITUDE=1",)),
    ),
    "su2-hea": (
        *tuple(
            FusionVariant(
                f"{forward}-forward_{backward}-backward",
                "optimized",
                (
                    f"-DSAD_SU2_FORWARD_STRATEGY={forward_id}",
                    f"-DSAD_SU2_BACKWARD_STRATEGY={backward_id}",
                ),
            )
            for forward_id, forward in enumerate(("split", "lookup", "phased"))
            for backward_id, backward in enumerate(("split", "lookup", "phased"))
        ),
        FusionVariant("auto", "optimized"),
    ),
    "rzz-hea": (
        *tuple(
            FusionVariant(
                f"{forward}-forward_{backward}-backward",
                "optimized",
                (
                    f"-DSAD_RZZ_FORWARD_FUSED={forward_id}",
                    f"-DSAD_RZZ_BACKWARD_STRATEGY={backward_id}",
                ),
            )
            for forward_id, forward in enumerate(("split", "fused"))
            for backward_id, backward in enumerate(("split", "combined", "fused"))
        ),
        FusionVariant("auto", "optimized"),
    ),
    "qaoa": (
        FusionVariant(
            "split-both",
            "optimized",
            ("-DSAD_QAOA_FUSE_COST_RX=0", "-DSAD_QAOA_FUSED_BACKWARD=0"),
        ),
        FusionVariant(
            "fused-forward",
            "optimized",
            ("-DSAD_QAOA_FUSE_COST_RX=1", "-DSAD_QAOA_FUSED_BACKWARD=0"),
        ),
        FusionVariant(
            "fused-backward",
            "optimized",
            ("-DSAD_QAOA_FUSE_COST_RX=0", "-DSAD_QAOA_FUSED_BACKWARD=2"),
        ),
        FusionVariant(
            "fused-both",
            "optimized",
            ("-DSAD_QAOA_FUSE_COST_RX=1", "-DSAD_QAOA_FUSED_BACKWARD=2"),
        ),
        FusionVariant(
            "auto",
            "optimized",
            ("-DSAD_QAOA_FUSE_COST_RX=-1", "-DSAD_QAOA_FUSED_BACKWARD=1"),
        ),
    ),
    "xxz-hva": (
        FusionVariant("separate-matchings", "optimized", ("-DSAD_XXZ_CROSS_MATCHING=0",)),
        FusionVariant("cross-matching", "optimized", ("-DSAD_XXZ_CROSS_MATCHING=1",)),
    ),
}


def shape_for(circuit: str, qubits: int) -> Shape:
    if qubits < 20:
        return SAFE
    if circuit == "ra-hea":
        return F64R3_B64R4 if qubits >= 28 else F64R4_B64R4
    if circuit == "su2-hea":
        return F128R3_B64R4
    if circuit in {"rzz-hea", "qaoa"}:
        if qubits >= 26:
            return F64R4_B128R3
        if qubits == 24:
            return F64R3_B64R4
        return SAFE
    if circuit == "xxz-hva":
        return F128R3_B32R3
    raise ValueError(circuit)


def _library_for(shape: Shape, variant: FusionVariant) -> Path:
    flags = (*shape.flags, *variant.flags)
    digest = hashlib.sha256("\0".join(flags).encode()).hexdigest()[:10]
    safe_name = variant.name.replace("-", "_")
    return SAD_ROOT / "build" / f"libsad_fusion_{shape.name}_{safe_name}_{digest}.so"


def _build(shape: Shape, variant: FusionVariant) -> Path:
    path = _library_for(shape, variant)
    if path.exists():
        source_mtime = max(
            file.stat().st_mtime
            for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h")
            for file in SAD_ROOT.glob(pattern)
        )
        if path.stat().st_mtime >= source_mtime:
            return path
    flags = (*shape.flags, *variant.flags)
    command = [
        "make",
        "-C",
        str(SAD_ROOT),
        f"TARGET={path.relative_to(SAD_ROOT)}",
    ]
    if flags:
        command.append(f"EXTRA_NVCCFLAGS={' '.join(flags)}")
    subprocess.run(command, cwd=ROOT, check=True)
    return path


def _parse_qubits(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item < 4 or item > 30 for item in result):
        raise argparse.ArgumentTypeError("qubits must be comma-separated 4..30")
    return result


def _completed(path: Path) -> set[tuple[str, int, str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (
            row["circuit"],
            int(row["qubits"]),
            row["variant"],
            int(row["layers"]),
            int(row["sample"]),
        )
        for row in rows
        if row.get("correct") == "1"
    }


def _reference(
    circuit: str,
    qubits: int,
    layers: int,
    library: Path,
) -> EnergyGradResult:
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = "legacy"
    return energy_and_grad(
        circuit=circuit,
        scalability=(qubits, layers),
        steps=1,
        warmup_steps=1,
    )


def _measure(
    circuit: str,
    qubits: int,
    layers: int,
    steps: int,
    variant: FusionVariant,
    library: Path,
) -> EnergyGradResult:
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = variant.execution_mode
    return energy_and_grad(
        circuit=circuit,
        scalability=(qubits, layers),
        steps=steps,
        warmup_steps=2 if qubits <= 24 else 1,
    )


def configurations(
    circuits: Iterable[str], qubits: Iterable[int]
) -> Iterable[tuple[str, int, Shape, FusionVariant]]:
    for circuit in circuits:
        for q in qubits:
            if circuit in {"rzz-hea", "qaoa", "xxz-hva"} and q & 1:
                continue
            shape = shape_for(circuit, q)
            for variant in VARIANTS[circuit]:
                yield circuit, q, shape, variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit", action="append", choices=CIRCUITS, help="repeat to select"
    )
    parser.add_argument("--qubits", type=_parse_qubits, default=(16, 20, 24, 26, 28))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.layers < 2 or min(args.steps, args.rounds) < 1:
        parser.error("layers must be >=2; steps and rounds must be positive")
    circuits = tuple(args.circuit or CIRCUITS)
    configs = tuple(configurations(circuits, args.qubits))
    unique_builds = {(shape, variant) for _, _, shape, variant in configs}
    libraries: dict[tuple[Shape, FusionVariant], Path] = {}
    for shape, variant in sorted(
        unique_builds, key=lambda item: (item[0].name, item[1].name)
    ):
        print(f"build {shape.name}/{variant.name}", flush=True)
        libraries[shape, variant] = _build(shape, variant)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    references: dict[tuple[str, int], EnergyGradResult] = {}
    failures: list[str] = []
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        pending = [
            config
            for config in configs
            if not all(
                (config[0], config[1], config[3].name, args.layers, sample)
                in done
                for sample in range(args.steps * args.rounds)
            )
        ]
        # Prime each code path after compilation before collecting any row.
        for circuit, q, shape, variant in pending:
            library = libraries[shape, variant]
            key = (circuit, q)
            if key not in references:
                references[key] = _reference(circuit, q, args.layers, library)
            _measure(circuit, q, args.layers, 1, variant, library)

        ordered: list[tuple[int, tuple[str, int, Shape, FusionVariant]]] = []
        for round_index in range(args.rounds):
            round_configs = list(pending)
            random.Random(1729 + round_index).shuffle(round_configs)
            ordered.extend((round_index, config) for config in round_configs)
        for round_index, (circuit, q, shape, variant) in ordered:
            library = libraries[shape, variant]
            key = (circuit, q)
            reference = references[key]
            result = _measure(
                circuit, q, args.layers, args.steps, variant, library
            )
            energy_error = abs(result.energy - reference.energy)
            difference = np.asarray(result.grad) - np.asarray(reference.grad)
            gradient_max_error = float(np.max(np.abs(difference)))
            gradient_l2_error = float(np.linalg.norm(difference))
            correct = energy_error <= 1e-10 and gradient_max_error <= 2e-9
            flags = (*shape.flags, *variant.flags)
            for local_sample, (forward, hamiltonian, backward, total) in enumerate(
                zip(
                    result.forward_times_s,
                    result.hamiltonian_times_s,
                    result.backward_times_s,
                    result.step_times_s,
                    strict=True,
                )
            ):
                sample = round_index * args.steps + local_sample
                writer.writerow(
                    {
                        "circuit": circuit,
                        "qubits": q,
                        "layers": args.layers,
                        "variant": variant.name,
                        "execution_mode": variant.execution_mode,
                        "shape": shape.name,
                        "sample": sample,
                        "forward_ms": 1e3 * forward,
                        "hamiltonian_ms": 1e3 * hamiltonian,
                        "backward_ms": 1e3 * backward,
                        "total_ms": 1e3 * total,
                        "energy": result.energy,
                        "energy_abs_error": energy_error,
                        "gradient_max_abs_error": gradient_max_error,
                        "gradient_l2_error": gradient_l2_error,
                        "correct": int(correct),
                        "library": library.name,
                        "compile_flags": json.dumps(flags),
                    }
                )
            stream.flush()
            print(
                f"{circuit:8s} q={q} {variant.name:25s} "
                f"F={1e3 * statistics.median(result.forward_times_s):.3f} "
                f"B={1e3 * statistics.median(result.backward_times_s):.3f} "
                f"correct={correct}",
                flush=True,
            )
            if not correct:
                failures.append(f"{circuit} q={q} {variant.name}")
    if failures:
        raise RuntimeError("correctness failures: " + ", ".join(failures))
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
