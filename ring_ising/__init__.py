"""Python frontend package for the ring-Ising PennyLane and CUDA backend demo."""

from __future__ import annotations

__all__ = [
    "BACKENDS",
    "DEFAULT_PROGRESS_PARTITIONS",
    "PennyLaneResult",
    "PennyLaneWorkflow",
    "RunConfig",
    "STANDALONE_GRADIENT_STRATEGIES",
    "SUPPORTED_STANDALONE_GRADIENT_STRATEGIES",
    "StandaloneBackendConfig",
    "StandaloneResult",
    "StandaloneWorkflow",
    "StepEvaluation",
    "StepMetric",
    "TrainingRunResult",
    "RingIsingAdjointBackend",
    "apply_ring_layer",
    "build_ring_ising_hamiltonian",
    "hardware_efficient_ring",
    "run",
    "run_pennylane",
    "run_standalone",
]


def __getattr__(name: str):
    if name in {
        "BACKENDS",
        "DEFAULT_PROGRESS_PARTITIONS",
        "RunConfig",
        "STANDALONE_GRADIENT_STRATEGIES",
        "SUPPORTED_STANDALONE_GRADIENT_STRATEGIES",
    }:
        from .config import (
            BACKENDS,
            DEFAULT_PROGRESS_PARTITIONS,
            RunConfig,
            STANDALONE_GRADIENT_STRATEGIES,
            SUPPORTED_STANDALONE_GRADIENT_STRATEGIES,
        )

        namespace = {
            "BACKENDS": BACKENDS,
            "DEFAULT_PROGRESS_PARTITIONS": DEFAULT_PROGRESS_PARTITIONS,
            "RunConfig": RunConfig,
            "STANDALONE_GRADIENT_STRATEGIES": STANDALONE_GRADIENT_STRATEGIES,
            "SUPPORTED_STANDALONE_GRADIENT_STRATEGIES": (
                SUPPORTED_STANDALONE_GRADIENT_STRATEGIES
            ),
        }
        return namespace[name]

    if name in {
        "apply_ring_layer",
        "build_ring_ising_hamiltonian",
        "hardware_efficient_ring",
    }:
        from .models import (
            apply_ring_layer,
            build_ring_ising_hamiltonian,
            hardware_efficient_ring,
        )

        namespace = {
            "apply_ring_layer": apply_ring_layer,
            "build_ring_ising_hamiltonian": build_ring_ising_hamiltonian,
            "hardware_efficient_ring": hardware_efficient_ring,
        }
        return namespace[name]

    if name in {"StepEvaluation", "StepMetric", "TrainingRunResult"}:
        from .training import StepEvaluation, StepMetric, TrainingRunResult

        namespace = {
            "StepEvaluation": StepEvaluation,
            "StepMetric": StepMetric,
            "TrainingRunResult": TrainingRunResult,
        }
        return namespace[name]

    if name == "run":
        from .workflows import run

        return run

    if name in {
        "PennyLaneResult",
        "PennyLaneWorkflow",
        "run_pennylane",
    }:
        from .workflows.pennylane import PennyLaneResult, PennyLaneWorkflow, run_pennylane

        namespace = {
            "PennyLaneResult": PennyLaneResult,
            "PennyLaneWorkflow": PennyLaneWorkflow,
            "run_pennylane": run_pennylane,
        }
        return namespace[name]

    if name in {
        "RingIsingAdjointBackend",
        "StandaloneBackendConfig",
    }:
        from .backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig

        namespace = {
            "RingIsingAdjointBackend": RingIsingAdjointBackend,
            "StandaloneBackendConfig": StandaloneBackendConfig,
        }
        return namespace[name]

    if name in {
        "StandaloneResult",
        "StandaloneWorkflow",
        "run_standalone",
    }:
        from .workflows.standalone import StandaloneResult, StandaloneWorkflow, run_standalone

        namespace = {
            "StandaloneResult": StandaloneResult,
            "StandaloneWorkflow": StandaloneWorkflow,
            "run_standalone": run_standalone,
        }
        return namespace[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
