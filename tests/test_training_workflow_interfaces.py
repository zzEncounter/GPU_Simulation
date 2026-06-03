"""Interface-level checks for PennyLane/standalone workflow interfaces."""

from __future__ import annotations

import unittest

from ring_ising import (
    RunConfig,
    run,
    run_pennylane,
    run_standalone,
)
from ring_ising.training import StepMetric as CanonicalStepMetric
from ring_ising.workflows import BACKENDS


class TrainingWorkflowInterfaceTest(unittest.TestCase):
    def test_package_level_interfaces_are_canonical(self) -> None:
        self.assertEqual(BACKENDS, ("pennylane", "standalone"))
        self.assertEqual(CanonicalStepMetric.__name__, "StepMetric")

    def test_run_functions_return_comparable_results(self) -> None:
        pennylane = run_pennylane(
            RunConfig(
                backend="pennylane",
                num_qubits=4,
                layers=2,
                steps=1,
                report_steps=True,
                show_progress=False,
                seed=123,
                verbose=False,
            )
        )
        standalone = run_standalone(
            RunConfig(
                backend="standalone",
                num_qubits=4,
                layers=2,
                steps=1,
                report_steps=True,
                show_progress=False,
                gradient_strategy="save_param_states",
                seed=123,
                verbose=False,
            )
        )

        self.assertGreater(pennylane.timings.measured_loop_s, 0.0)
        self.assertGreater(standalone.timings.measured_loop_s, 0.0)
        self.assertAlmostEqual(pennylane.final_energy, standalone.final_energy, places=7)
        self.assertAlmostEqual(
            pennylane.step_metrics[0].energy,
            standalone.step_metrics[0].energy,
            places=7,
        )

    def test_unified_run_dispatcher_supports_both_backends(self) -> None:
        pennylane = run(
            RunConfig(
                backend="pennylane",
                num_qubits=4,
                layers=2,
                steps=1,
                report_steps=True,
                show_progress=False,
                seed=123,
                verbose=False,
            )
        )
        standalone = run(
            RunConfig(
                backend="standalone",
                num_qubits=4,
                layers=2,
                steps=1,
                report_steps=True,
                show_progress=False,
                seed=123,
                gradient_strategy="save_param_states",
                verbose=False,
            )
        )

        self.assertAlmostEqual(pennylane.final_energy, standalone.final_energy, places=7)
        self.assertAlmostEqual(
            pennylane.step_metrics[0].energy,
            standalone.step_metrics[0].energy,
            places=7,
        )

if __name__ == "__main__":
    unittest.main()
