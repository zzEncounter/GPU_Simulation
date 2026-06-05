"""Unified workflow entrypoints for PennyLane and standalone training paths."""

from __future__ import annotations

from ring_ising.config import BACKENDS, RunConfig

from .pennylane import run_pennylane
from .standalone import run_standalone


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
    "run",
    "run_pennylane",
    "run_standalone",
]
