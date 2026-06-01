"""Standalone CUDA-backed runtime for the ring Ising demo."""

from __future__ import annotations

from .config import RingIsingConfig, StrategyResolution
from .ising_runtime import RingIsingAdjointBackend, make_initial_params

__all__ = [
    "RingIsingAdjointBackend",
    "RingIsingConfig",
    "StandaloneResult",
    "StandaloneRunConfig",
    "StandaloneWorkflow",
    "StrategyResolution",
    "create_workflow",
    "make_initial_params",
    "print_runtime_summary",
    "run_standalone",
]


def __getattr__(name: str):
    if name in {
        "StandaloneResult",
        "StandaloneRunConfig",
        "StandaloneWorkflow",
        "create_workflow",
        "print_runtime_summary",
        "run_standalone",
    }:
        from .workflow import (
            StandaloneResult,
            StandaloneRunConfig,
            StandaloneWorkflow,
            create_workflow,
            print_runtime_summary,
            run_standalone,
        )

        namespace = {
            "StandaloneResult": StandaloneResult,
            "StandaloneRunConfig": StandaloneRunConfig,
            "StandaloneWorkflow": StandaloneWorkflow,
            "create_workflow": create_workflow,
            "print_runtime_summary": print_runtime_summary,
            "run_standalone": run_standalone,
        }
        return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
