"""Configuration and sizing helpers for the standalone backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RingIsingConfig:
    """Configuration for the standalone ring-Ising workflow."""

    num_qubits: int = 12
    layers: int = 3
    field: float = 1.0
    gradient_strategy: str = "auto"
    checkpoint_interval_ops: int | None = None
    gate_fusion: bool = True
    auto_memory_budget_fraction: float = 0.85
    auto_memory_reserve_mib: int = 1024

    @property
    def param_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.num_qubits, 2)

    @property
    def num_params(self) -> int:
        return self.layers * self.num_qubits * 2

    @property
    def num_ops(self) -> int:
        if self.gate_fusion:
            return self.layers * (self.num_qubits + 1)
        return 2 * self.layers * self.num_qubits

    @property
    def num_parametric_gates(self) -> int:
        return self.layers * self.num_qubits

    @property
    def statevector_nbytes(self) -> int:
        return (1 << self.num_qubits) * 16

    def default_checkpoint_interval_ops(self) -> int:
        if self.num_ops <= 1:
            return 0
        auto_interval = int(np.ceil(np.sqrt(self.num_ops)))
        return max(1, min(auto_interval, self.num_ops - 1))

    def resolve_checkpoint_interval_ops(self, strategy: str = "checkpoint") -> int:
        if strategy == "save_param_states":
            return 0
        if self.num_ops <= 1:
            return 0
        if self.checkpoint_interval_ops is not None:
            return max(1, min(int(self.checkpoint_interval_ops), self.num_ops - 1))
        return self.default_checkpoint_interval_ops()

    def estimated_gradient_state_buffers_for(
        self, strategy: str, checkpoint_interval_ops: int | None = None
    ) -> int:
        if strategy == "save_param_states":
            return self.num_parametric_gates + 3
        if strategy != "checkpoint":
            raise ValueError("strategy must be 'save_param_states' or 'checkpoint'.")
        checkpoint_interval_ops = self.resolve_checkpoint_interval_ops(
            strategy="checkpoint"
        ) if checkpoint_interval_ops is None else checkpoint_interval_ops
        if checkpoint_interval_ops == 0:
            return self.num_parametric_gates + 3
        num_chunks = int(np.ceil(self.num_ops / checkpoint_interval_ops))
        return num_chunks + checkpoint_interval_ops + 5

    def estimated_gradient_workspace_gib_for(
        self, strategy: str, checkpoint_interval_ops: int | None = None
    ) -> float:
        return (
            self.estimated_gradient_state_buffers_for(strategy, checkpoint_interval_ops)
            * self.statevector_nbytes
            / (1024**3)
        )

    def validate(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2.")
        if self.layers < 1:
            raise ValueError("layers must be at least 1.")
        if self.gradient_strategy not in {
            "auto",
            "save_param_states",
            "checkpoint",
        }:
            raise ValueError(
                "gradient_strategy must be 'auto', 'save_param_states', "
                "or 'checkpoint'."
            )
        if self.checkpoint_interval_ops is not None and self.checkpoint_interval_ops < 1:
            raise ValueError("checkpoint_interval_ops must be positive when provided.")
        if not (0.0 < self.auto_memory_budget_fraction <= 1.0):
            raise ValueError("auto_memory_budget_fraction must be in (0, 1].")
        if self.auto_memory_reserve_mib < 0:
            raise ValueError("auto_memory_reserve_mib must be non-negative.")


@dataclass(frozen=True)
class StrategyResolution:
    requested_strategy: str
    resolved_strategy: str
    checkpoint_interval_ops: int
    available_gpu_memory_mib: int | None
    total_gpu_memory_mib: int | None
    memory_budget_mib: float | None
    estimated_workspace_gib: float
    note: str | None = None
