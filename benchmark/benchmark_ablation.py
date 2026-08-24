"""Small reproducible ablations used by the optimization report."""

from __future__ import annotations

import csv
import os
import statistics
import sys
from pathlib import Path

SAD_PYTHON = Path(__file__).resolve().parents[1] / "sad" / "python"
sys.path.insert(0, str(SAD_PYTHON))
from sad_baseline import energy_and_grad  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "results" / "research_ablation.csv"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa")
FIELDS = (
    "experiment",
    "circuit",
    "qubits",
    "layers",
    "mode",
    "forward_median_s",
    "hamiltonian_median_s",
    "backward_median_s",
    "total_median_s",
)


def measure(
    writer: csv.DictWriter,
    experiment: str,
    circuit: str,
    qubits: int,
    layers: int,
    mode: str,
) -> None:
    os.environ["SAD_EXECUTION_MODE"] = mode
    steps = 20 if qubits <= 20 else (6 if qubits <= 24 else 2)
    result = energy_and_grad(
        circuit=circuit,
        scalability=(qubits, layers),
        steps=steps,
        warmup_steps=2 if qubits <= 24 else 1,
    )
    writer.writerow(
        {
            "experiment": experiment,
            "circuit": circuit,
            "qubits": qubits,
            "layers": layers,
            "mode": mode,
            "forward_median_s": statistics.median(result.forward_times_s),
            "hamiltonian_median_s": statistics.median(
                result.hamiltonian_times_s
            ),
            "backward_median_s": statistics.median(result.backward_times_s),
            "total_median_s": statistics.median(result.step_times_s),
        }
    )
    print(experiment, circuit, qubits, mode, flush=True)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for circuit in CIRCUITS:
            for qubits in range(4, 29, 4):
                for mode in ("legacy", "initial-only"):
                    measure(writer,
                            "first-layer",
                            circuit,
                            qubits,
                            1,
                            mode)
        for circuit in CIRCUITS:
            for qubits in (16, 20, 24):
                for mode in ("initial-only", "fused-forward", "optimized"):
                    measure(writer,
                            "layer-fusion",
                            circuit,
                            qubits,
                            8,
                            mode)
    print(f"CSV written to {OUTPUT}")


if __name__ == "__main__":
    main()
