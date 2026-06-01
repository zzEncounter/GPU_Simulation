"""Interface-level checks for baseline/standalone training workflows."""

from __future__ import annotations

import unittest

from ring_ising.baseline import BaselineConfig, run_baseline
from ring_ising.training import StepMetric as CanonicalStepMetric
from ring_ising.workflows.pennylane import BaselineConfig as CanonicalBaselineConfig
from ring_ising.workflows.standalone import StandaloneRunConfig as CanonicalStandaloneRunConfig
from ring_ising.optimization_loop import StepMetric as LegacyStepMetric
from standalone_backend import StandaloneRunConfig, run_standalone


class TrainingWorkflowInterfaceTest(unittest.TestCase):
    def test_legacy_and_canonical_interfaces_match(self) -> None:
        self.assertIs(BaselineConfig, CanonicalBaselineConfig)
        self.assertIs(StandaloneRunConfig, CanonicalStandaloneRunConfig)
        self.assertIs(LegacyStepMetric, CanonicalStepMetric)

    def test_run_functions_return_comparable_results(self) -> None:
        baseline = run_baseline(
            BaselineConfig(
                num_qubits=4,
                layers=2,
                steps=1,
                seed=123,
                verbose=False,
            )
        )
        standalone = run_standalone(
            StandaloneRunConfig(
                num_qubits=4,
                layers=2,
                steps=1,
                seed=123,
                verbose=False,
                measure_backend_timings=False,
            )
        )

        self.assertGreater(baseline.timings.measured_loop_s, 0.0)
        self.assertGreater(standalone.timings.measured_loop_s, 0.0)
        self.assertIsNone(standalone.timings.backend_total_ms)
        self.assertAlmostEqual(baseline.final_energy, standalone.final_energy, places=7)

    def test_standalone_backend_timings_switch(self) -> None:
        with_timings = run_standalone(
            StandaloneRunConfig(
                num_qubits=4,
                layers=2,
                steps=1,
                seed=321,
                verbose=False,
                measure_backend_timings=True,
            )
        )
        self.assertIsNotNone(with_timings.timings.backend_total_ms)
        self.assertGreater(with_timings.timings.backend_total_ms or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
