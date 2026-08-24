"""Run matching PennyLane references for shared-angle QAOA and XXZ-HVA."""

from pathlib import Path

import benchmark_pennylane_lightning as benchmark

benchmark.CIRCUITS = ("qaoa", "xxz-hva")
benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent
    / "results"
    / "qaoa_xxz_pennylane_gpu.csv"
)

if __name__ == "__main__":
    benchmark.main()
