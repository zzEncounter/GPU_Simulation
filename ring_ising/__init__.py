"""Python frontend package for the ring-Ising baseline and CUDA backend demo."""

from .baseline import BaselineConfig, BaselineResult, BaselineWorkflow, create_workflow, run_baseline
from .models import apply_ring_layer, build_ring_ising_hamiltonian, hardware_efficient_ring, make_initial_params

__all__ = [
    "BaselineConfig",
    "BaselineResult",
    "BaselineWorkflow",
    "apply_ring_layer",
    "build_ring_ising_hamiltonian",
    "create_workflow",
    "hardware_efficient_ring",
    "make_initial_params",
    "run_baseline",
]
