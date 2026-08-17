"""End-to-end XXZ schedule and tile selection benchmark."""

from __future__ import annotations

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


OUTPUT = ROOT / "benchmark" / "results" / "xxz_full_circuit.csv"
VARIANTS = {
    "generic-separate": ("-DSAD_XXZ_CROSS_MATCHING=0",),
    "tuned-separate": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=3",
        "-DSAD_XXZ_CROSS_MATCHING=0",
    ),
    "tuned-cross": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=3",
        "-DSAD_XXZ_CROSS_MATCHING=1",
    ),
}


def build(name: str, flags: tuple[str, ...]) -> Path:
    relative = Path("build") / f"libsad_xxz_full_{name}.so"
    subprocess.run(
        [
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={relative}",
            f"EXTRA_NVCCFLAGS={' '.join(flags)}",
        ],
        cwd=ROOT,
        check=True,
    )
    return SAD_ROOT / relative


def main() -> None:
    libraries = {name: build(name, flags) for name, flags in VARIANTS.items()}
    rows: list[dict[str, object]] = []
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    for name in VARIANTS:
        os.environ["SAD_LIBRARY_PATH"] = str(libraries[name])
        for qubits in (20, 24, 26, 28):
            steps = 7 if qubits <= 20 else 5 if qubits <= 24 else 3
            result = energy_and_grad(
                circuit="xxz-hva",
                scalability=(qubits, 8),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "variant": name,
                "qubits": qubits,
                "layers": 8,
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
                f"{name:16s} q={qubits} F={row['forward_median_ms']:.3f} "
                f"B={row['backward_median_ms']:.3f} "
                f"T={row['total_median_ms']:.3f} ms",
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
