"""Qiskit AerSimulator baseline: parameter-shift-rule gradient benchmark."""

from .runner import energy_and_grad, EnergyGradResult

__all__ = ["energy_and_grad", "EnergyGradResult"]
