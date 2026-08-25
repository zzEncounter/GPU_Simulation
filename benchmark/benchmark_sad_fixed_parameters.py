"""Run the SAD main experiment with one uniform compile-time shape.

The implementation and optimized circuit execution paths stay unchanged.  Only
the circuit/size-dependent library dispatch is disabled, giving every scenario
the same F128r2/B128r2 kernel geometry.
"""

from __future__ import annotations

import os
from pathlib import Path

import benchmark_sad as benchmark


benchmark.OUTPUT_CSV = (
    Path(__file__).resolve().parent / "results" / "sad_fixed_parameters_gpu.csv"
)


if __name__ == "__main__":
    os.environ["SAD_DISABLE_VARIANT_DISPATCH"] = "1"
    benchmark.main()
