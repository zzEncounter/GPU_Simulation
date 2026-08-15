"""Join the fixed baselines with the optimized SAD main experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = RESULTS / "research_main.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["circuit"], int(row["qubits"]), int(row["layers"])


def main() -> None:
    optimized = [
        row
        for row in read(RESULTS / "sad_optimized_gpu.csv")
        if row["circuit"] != "qaoa"
    ]
    extension_sad = RESULTS / "qaoa_xxz_sad_gpu.csv"
    if extension_sad.exists():
        optimized += [
            row for row in read(extension_sad) if row["circuit"] == "xxz-hva"
        ]
    qaoa_sad = RESULTS / "qaoa_shared_sad_gpu.csv"
    if qaoa_sad.exists():
        optimized += read(qaoa_sad)
    elif extension_sad.exists():
        optimized += [
            row for row in read(extension_sad) if row["circuit"] == "qaoa"
        ]
    fixed_sad = {key(row): row for row in read(RESULTS / "sad_gpu.csv")}
    pennylane_rows = read(RESULTS / "pennylane_lightning_gpu.csv")
    extension_reference = RESULTS / "qaoa_xxz_pennylane_gpu.csv"
    if extension_reference.exists():
        pennylane_rows += read(extension_reference)
    else:
        pennylane_rows += read(RESULTS / "qaoa_pennylane_gpu.csv")
    pennylane = {key(row): row for row in pennylane_rows}
    fields = (
        "circuit",
        "qubits",
        "layers",
        "kernel_variant",
        "optimized_time_median_s",
        "optimized_time_mean_s",
        "fixed_sad_time_median_s",
        "speedup_vs_fixed_sad_median",
        "pennylane_time_median_s",
        "speedup_vs_pennylane_median",
        "forward_mean_s",
        "hamiltonian_mean_s",
        "backward_mean_s",
        "energy_abs_error_vs_reference",
        "gradient_max_abs_error_vs_reference",
        "state_vector_mib",
        "total_workspace_mib",
    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in optimized:
            row_key = key(row)
            old = fixed_sad.get(row_key)
            reference = pennylane[row_key]
            optimized_median = float(row["time_median_s"])
            reference_median = float(reference["time_median_s"])
            old_median = float(old["time_median_s"]) if old else None
            gradient = np.asarray(json.loads(row["grad_json"]), dtype=np.float64)
            reference_gradient = np.asarray(
                json.loads(reference["grad_json"]), dtype=np.float64
            )
            writer.writerow(
                {
                    "circuit": row["circuit"],
                    "qubits": row["qubits"],
                    "layers": row["layers"],
                    "kernel_variant": row["kernel_variant"],
                    "optimized_time_median_s": optimized_median,
                    "optimized_time_mean_s": row["time_mean_s"],
                    "fixed_sad_time_median_s": "" if old is None else old_median,
                    "speedup_vs_fixed_sad_median": ""
                    if old_median is None
                    else old_median / optimized_median,
                    "pennylane_time_median_s": reference_median,
                    "speedup_vs_pennylane_median": reference_median
                    / optimized_median,
                    "forward_mean_s": row["forward_mean_s"],
                    "hamiltonian_mean_s": row["hamiltonian_mean_s"],
                    "backward_mean_s": row["backward_mean_s"],
                    "energy_abs_error_vs_reference": abs(
                        float(row["energy"]) - float(reference["energy"])
                    ),
                    "gradient_max_abs_error_vs_reference": float(
                        np.max(np.abs(gradient - reference_gradient))
                    ),
                    "state_vector_mib": row["state_vector_mib"],
                    "total_workspace_mib": row["total_workspace_mib"],
                }
            )
    print(f"CSV written to {OUTPUT}")


if __name__ == "__main__":
    main()
