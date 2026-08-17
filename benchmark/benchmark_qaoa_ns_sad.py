"""Run the non-shared-angle QAOA SAD experiment."""

from pathlib import Path

import benchmark_sad as benchmark

benchmark.CIRCUITS = ("qaoa-ns",)
benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent / "results" / "qaoa_ns_sad_gpu.csv"
)

if __name__ == "__main__":
    benchmark.main()
