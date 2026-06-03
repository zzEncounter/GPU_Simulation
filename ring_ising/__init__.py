"""Python frontend package for the ring-Ising PennyLane and CUDA backend demo."""

from __future__ import annotations

__all__ = [
    "BACKENDS",
    "DEFAULT_PROGRESS_PARTITIONS",
    "LoopTimingBreakdown",
    "PennyLaneResult",
    "PennyLaneWorkflow",
    "RunConfig",
    "STANDALONE_GRADIENT_STRATEGIES",
    "StandaloneBackendConfig",
    "StandaloneResult",
    "StandaloneWorkflow",
    "StepMetric",
    "TrainingRunResult",
    "RingIsingAdjointBackend",
    "apply_ring_layer",
    "build_ring_ising_hamiltonian",
    "create_backend_workflow",
    "create_pennylane_workflow",
    "create_standalone_workflow",
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
    }:
        from .config import (
            BACKENDS,
            DEFAULT_PROGRESS_PARTITIONS,
            RunConfig,
            STANDALONE_GRADIENT_STRATEGIES,
        )

        namespace = {
            "BACKENDS": BACKENDS,
            "DEFAULT_PROGRESS_PARTITIONS": DEFAULT_PROGRESS_PARTITIONS,
            "RunConfig": RunConfig,
            "STANDALONE_GRADIENT_STRATEGIES": STANDALONE_GRADIENT_STRATEGIES,
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

    if name in {"LoopTimingBreakdown", "StepMetric", "TrainingRunResult"}:
        from .training import LoopTimingBreakdown, StepMetric, TrainingRunResult

        namespace = {
            "LoopTimingBreakdown": LoopTimingBreakdown,
            "StepMetric": StepMetric,
            "TrainingRunResult": TrainingRunResult,
        }
        return namespace[name]

    if name in {"run", "create_backend_workflow"}:
        from .workflows import create_workflow, run

        namespace = {
            "create_backend_workflow": create_workflow,
            "run": run,
        }
        return namespace[name]

    if name in {
        "PennyLaneResult",
        "PennyLaneWorkflow",
        "create_pennylane_workflow",
        "run_pennylane",
    }:
        from .workflows.pennylane import (
            PennyLaneResult,
            PennyLaneWorkflow,
            create_pennylane_workflow,
            run_pennylane,
        )

        namespace = {
            "PennyLaneResult": PennyLaneResult,
            "PennyLaneWorkflow": PennyLaneWorkflow,
            "create_pennylane_workflow": create_pennylane_workflow,
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
        "create_standalone_workflow",
        "run_standalone",
    }:
        from .workflows.standalone import (
            StandaloneResult,
            StandaloneWorkflow,
            create_workflow,
            run_standalone,
        )

        namespace = {
            "StandaloneResult": StandaloneResult,
            "StandaloneWorkflow": StandaloneWorkflow,
            "create_standalone_workflow": create_workflow,
            "run_standalone": run_standalone,
        }
        return namespace[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
