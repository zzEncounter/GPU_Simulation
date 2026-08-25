"""Legacy HEA end-to-end A/B for directional standalone ring CNOT."""

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


OUTPUT = ROOT / "benchmark" / "results" / "cnot_end_to_end.csv"


def build(scatter: bool) -> Path:
    mode = "scatter" if scatter else "gather"
    relative = Path("build") / f"libsad_cnot_{mode}.so"
    subprocess.run(
        [
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={relative}",
            f"EXTRA_NVCCFLAGS=-DSAD_CNOT_FORWARD_SCATTER={int(scatter)}",
        ],
        cwd=ROOT,
        check=True,
    )
    return SAD_ROOT / relative


def main() -> None:
    libraries = {scatter: build(scatter) for scatter in (False, True)}
    configurations = (
        ("ra-hea", 20),
        ("ra-hea", 24),
        ("su2-hea", 20),
        ("su2-hea", 24),
        ("su2-hea", 28),
    )
    rows: list[dict[str, object]] = []
    os.environ["SAD_EXECUTION_MODE"] = "legacy"
    for scatter in (False, True):
        os.environ["SAD_LIBRARY_PATH"] = str(libraries[scatter])
        for circuit, qubits in configurations:
            steps = 7 if qubits <= 20 else 5 if qubits <= 24 else 3
            result = energy_and_grad(
                circuit=circuit,
                scalability=(qubits, 8),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "variant": "scatter-forward" if scatter else "gather-both",
                "circuit": circuit,
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
                f"{row['variant']:15s} {circuit:7s} q={qubits} "
                f"F={row['forward_median_ms']:.3f} "
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
