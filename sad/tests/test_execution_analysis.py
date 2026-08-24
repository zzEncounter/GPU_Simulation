from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from analyze_execution_search import (  # noqa: E402
    Aggregate,
    confirmed_rows,
    nonnegative_least_squares,
    relative_errors,
)


def test_nonnegative_least_squares_recovers_sparse_physical_model():
    x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ]
    )
    expected = np.asarray([2.0, 0.0, 4.0])
    fitted = nonnegative_least_squares(x, x @ expected)
    np.testing.assert_allclose(fitted, expected, atol=1e-12)


def test_relative_error_is_scale_independent():
    actual = np.asarray([1.0, 1000.0])
    predicted = np.asarray([1.1, 1100.0])
    np.testing.assert_allclose(relative_errors(actual, predicted), [0.1, 0.1])


def _aggregate(stage: str, samples: tuple[float, ...]) -> Aggregate:
    return Aggregate(
        stage=stage, variant="v", family="compact", candidate="c",
        gate="rx", direction="forward", qubits=20, layout="plan:x",
        threads=32, register_amplitudes=4, tile_bits=7, phase_count=1,
        gate_count=1, registers_per_thread=1, static_shared_bytes=0,
        dynamic_shared_bytes=0, active_cta_per_sm=1, mailbox_bytes=0,
        mailbox_chunks=1, local_bytes_per_thread=0, multiprocessors=1,
        phase_targets=(1,), phase_lane_targets=(1,),
        phase_register_targets=(0,), phase_warp_targets=(0,),
        samples=samples,
    )


def test_confirmed_rows_only_filters_under_repeated_schedules():
    shape = _aggregate("shape", (1.0,))
    screening = _aggregate("schedule", (1.0,))
    confirmed = _aggregate("schedule", (1.0, 1.1, 0.9))
    assert confirmed_rows((shape, screening, confirmed), 3) == [
        shape, confirmed
    ]
