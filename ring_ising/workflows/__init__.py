"""Unified workflow entrypoints for PennyLane and standalone training paths."""

from __future__ import annotations

from ring_ising.config import BACKENDS, RunConfig

from .pennylane import PennyLaneWorkflow, create_pennylane_workflow, run_pennylane
from .standalone import StandaloneWorkflow, create_workflow as create_standalone_workflow, run_standalone

WorkflowHandle = PennyLaneWorkflow | StandaloneWorkflow


def create_workflow(config: RunConfig) -> WorkflowHandle:
    """Create a prepared workflow for the requested backend."""

    if config.backend == "pennylane":
        return create_pennylane_workflow(config)
    if config.backend == "standalone":
        return create_standalone_workflow(config)
    choices = ", ".join(BACKENDS)
    raise ValueError(f"Unknown backend {config.backend!r}. Expected one of: {choices}")


def run(config: RunConfig):
    """Run one full training workflow on the requested backend."""

    if config.backend == "pennylane":
        return run_pennylane(config)
    if config.backend == "standalone":
        return run_standalone(config)
    choices = ", ".join(BACKENDS)
    raise ValueError(f"Unknown backend {config.backend!r}. Expected one of: {choices}")


__all__ = [
    "BACKENDS",
    "RunConfig",
    "WorkflowHandle",
    "create_workflow",
    "run",
    "run_pennylane",
    "run_standalone",
]
