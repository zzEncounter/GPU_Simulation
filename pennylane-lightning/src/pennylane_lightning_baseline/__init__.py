"""PennyLane Lightning-GPU adjoint-diff benchmark baseline."""

from .circuits import (
    CIRCUIT_REGISTRY,
    CircuitSpec,
    available_circuits,
    build_hamiltonian,
    get_circuit,
    register_circuit,
)
from .runner import EnergyGradResult, MemoryUsage, energy_and_grad
from .native import NativeEnergyGradResult, native_energy_and_grad

__all__ = [
    "CIRCUIT_REGISTRY",
    "CircuitSpec",
    "EnergyGradResult",
    "MemoryUsage",
    "NativeEnergyGradResult",
    "available_circuits",
    "build_hamiltonian",
    "energy_and_grad",
    "native_energy_and_grad",
    "get_circuit",
    "register_circuit",
]
