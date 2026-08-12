"""End-to-end A/B for the remaining cooperative multi-phase kernels."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402


OUTPUT = ROOT / "benchmark" / "results" / "persistent_gates.csv"


@dataclass(frozen=True)
class Variant:
    circuit: str
    shape: str
    persistent: bool

    @property
    def name(self) -> str:
        mode = "persistent" if self.persistent else "ordinary"
        return f"{self.circuit}-{self.shape}-{mode}"

    @property
    def target(self) -> Path:
        return SAD_ROOT / "build" / f"libsad_{self.name}.so"

    @property
    def flags(self) -> tuple[str, ...]:
        shape_flags = {
            "f64r4_b64r4": (
                "-DSAD_FORWARD_BLOCK_THREADS=64",
                "-DSAD_FORWARD_REGISTER_BITS=4",
                "-DSAD_BLOCK_THREADS=64",
                "-DSAD_REGISTER_BITS=4",
            ),
            "f64r3_b64r4": (
                "-DSAD_FORWARD_BLOCK_THREADS=64",
                "-DSAD_FORWARD_REGISTER_BITS=3",
                "-DSAD_BLOCK_THREADS=64",
                "-DSAD_REGISTER_BITS=4",
            ),
            "f128r3_b64r4": (
                "-DSAD_FORWARD_BLOCK_THREADS=128",
                "-DSAD_FORWARD_REGISTER_BITS=3",
                "-DSAD_BLOCK_THREADS=64",
                "-DSAD_REGISTER_BITS=4",
            ),
            "f128r2_b128r2": (),
        }[self.shape]
        macro = {
            "ra-hea": "SAD_REAL_PERSISTENT",
            "su2-hea": "SAD_PHASED_RY_PERSISTENT",
            "xxz-hva": "SAD_XXZ_PERSISTENT",
        }[self.circuit]
        extra = (
            ("-DSAD_XXZ_CROSS_MATCHING=0",)
            if self.circuit == "xxz-hva"
            else ()
        )
        return shape_flags + extra + (f"-D{macro}={int(self.persistent)}",)


def build(variant: Variant) -> None:
    relative_target = variant.target.relative_to(SAD_ROOT)
    subprocess.run(
        [
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={relative_target}",
            f"EXTRA_NVCCFLAGS={' '.join(variant.flags)}",
        ],
        cwd=ROOT,
        check=True,
    )


def configurations() -> list[tuple[Variant, int]]:
    rows: list[tuple[Variant, int]] = []
    for persistent in (True, False):
        for qubits in (20, 24, 26):
            rows.append(
                (Variant("ra-hea", "f64r4_b64r4", persistent), qubits)
            )
        rows.append((Variant("ra-hea", "f64r3_b64r4", persistent), 28))
        rows.append((Variant("su2-hea", "f128r3_b64r4", persistent), 28))
        for qubits in (20, 24, 26, 28):
            rows.append(
                (Variant("xxz-hva", "f128r2_b128r2", persistent), qubits)
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    variants = {variant for variant, _ in configurations()}
    for variant in sorted(variants, key=lambda item: item.name):
        build(variant)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
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
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for variant, qubits in configurations():
            os.environ["SAD_LIBRARY_PATH"] = str(variant.target)
            os.environ["SAD_EXECUTION_MODE"] = "optimized"
            steps = 7 if qubits <= 20 else 5 if qubits <= 24 else 3
            result = energy_and_grad(
                circuit=variant.circuit,
                scalability=(qubits, args.layers),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "variant": variant.name,
                "circuit": variant.circuit,
                "qubits": qubits,
                "layers": args.layers,
                "steps": steps,
                "forward_median_ms": 1e3
                * statistics.median(result.forward_times_s),
                "hamiltonian_median_ms": 1e3
                * statistics.median(result.hamiltonian_times_s),
                "backward_median_ms": 1e3
                * statistics.median(result.backward_times_s),
                "total_median_ms": 1e3
                * statistics.median(result.step_times_s),
                "energy": result.energy,
                "grad_json": json.dumps(result.grad.tolist()),
            }
            writer.writerow(row)
            stream.flush()
            print(
                f"{variant.name:39s} q={qubits} "
                f"F={row['forward_median_ms']:.3f} "
                f"B={row['backward_median_ms']:.3f} "
                f"T={row['total_median_ms']:.3f} ms",
                flush=True,
            )
    print(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
