"""Compatibility wrappers for the moved standalone workflow."""

from ring_ising.workflows.standalone import (
    StandaloneResult,
    StandaloneRunConfig,
    StandaloneWorkflow,
    create_workflow,
    print_runtime_summary,
    run_standalone,
)

__all__ = [
    "StandaloneResult",
    "StandaloneRunConfig",
    "StandaloneWorkflow",
    "create_workflow",
    "print_runtime_summary",
    "run_standalone",
]
