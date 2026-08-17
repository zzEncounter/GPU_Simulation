"""Run the non-shared-angle QAOA PennyLane-Lightning experiment."""

from pathlib import Path

import benchmark_pennylane_lightning as benchmark

benchmark.CIRCUITS = ("qaoa-ns",)
benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent / "results" / "qaoa_ns_pennylane_gpu.csv"
)

if __name__ == "__main__":
    benchmark.main()
