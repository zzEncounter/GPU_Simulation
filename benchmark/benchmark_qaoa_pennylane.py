"""Run only the added QAOA circuit without overwriting the fixed baseline."""

from pathlib import Path

import benchmark_pennylane_lightning as benchmark

benchmark.CIRCUITS = ("qaoa",)
benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent / "results" / "qaoa_pennylane_gpu.csv"
)

if __name__ == "__main__":
    benchmark.main()
