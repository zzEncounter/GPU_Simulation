"""End-to-end SU2 A/B for phased RZ+RY adjoint processing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402


OUTPUT = ROOT / "benchmark" / "results" / "su2_phased_backward.csv"
SHAPE_FLAGS = (
    "-DSAD_FORWARD_BLOCK_THREADS=128",
    "-DSAD_FORWARD_REGISTER_BITS=3",
    "-DSAD_BLOCK_THREADS=64",
    "-DSAD_REGISTER_BITS=4",
)


def build(phased: bool) -> Path:
    name = "phased" if phased else "lookup"
    relative = Path("build") / f"libsad_su2_backward_{name}.so"
    subprocess.run(
        [
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={relative}",
            "EXTRA_NVCCFLAGS="
            + " ".join(
                SHAPE_FLAGS + (f"-DSAD_SU2_PHASED_BACKWARD={int(phased)}",)
            ),
        ],
        cwd=ROOT,
        check=True,
    )
    return SAD_ROOT / relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    libraries = {phased: build(phased) for phased in (False, True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for phased in (False, True):
        os.environ["SAD_LIBRARY_PATH"] = str(libraries[phased])
        os.environ["SAD_EXECUTION_MODE"] = "optimized"
        for qubits in (20, 24, 26, 28):
            steps = 7 if qubits <= 20 else 5 if qubits <= 24 else 3
            result = energy_and_grad(
                circuit="su2-hea",
                scalability=(qubits, args.layers),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "variant": "phased" if phased else "lookup",
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
            rows.append(row)
            print(
                f"{row['variant']:6s} q={qubits} "
                f"B={row['backward_median_ms']:.3f} "
                f"T={row['total_median_ms']:.3f} ms",
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
