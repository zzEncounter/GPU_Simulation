"""Run the standard shared-angle QAOA and XXZ-HVA SAD experiments."""

from pathlib import Path

import benchmark_sad as benchmark

benchmark.CIRCUITS = ("qaoa", "xxz-hva")
benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent
    / "results"
    / "qaoa_xxz_sad_gpu.csv"
)

if __name__ == "__main__":
    benchmark.main()
