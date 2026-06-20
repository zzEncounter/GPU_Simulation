"""Configuration and sizing helpers for the standalone CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ring_ising.config import STANDALONE_GRADIENT_STRATEGIES


def normalize_strategy_name(strategy: str) -> str:
    if strategy == "reverse_walk":
        return "inverse_walk"
    if strategy == "bruteforce_parallel_q6":
        return "dense_scan"
    return strategy


@dataclass(frozen=True)
class StandaloneBackendConfig:
    """Configuration for the standalone CUDA backend runtime."""

    num_qubits: int
    layers: int
    field: float
    gradient_strategy: str
    checkpoint_interval_ops: int | None
    intrablock_block_size: int | None
    gate_fusion: bool

    @property
    def param_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.num_qubits, 2)

    @property
    def num_params(self) -> int:
        return self.layers * self.num_qubits * 2

    @property
    def normalized_gradient_strategy(self) -> str:
        return normalize_strategy_name(self.gradient_strategy)

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

    @property
    def dense_matrix_nbytes(self) -> int:
        dim = 1 << self.num_qubits
        return dim * dim * 16

    def default_checkpoint_interval_ops(self) -> int:
        if self.num_ops <= 1:
            return 0
        heuristic_interval = int(math.ceil(math.sqrt(self.num_ops)))
        return max(1, min(heuristic_interval, self.num_ops - 1))

    def resolve_checkpoint_interval_ops(self, strategy: str | None = None) -> int:
        strategy = normalize_strategy_name(
            self.gradient_strategy if strategy is None else strategy
        )
        if strategy in {
            "inverse_walk",
            "save_param_states",
            "dense_scan",
            "intrablock_parallel",
        }:
            return 0
        if self.num_ops <= 1:
            return 0
        if self.checkpoint_interval_ops is not None:
            return max(1, min(int(self.checkpoint_interval_ops), self.num_ops - 1))
        return self.default_checkpoint_interval_ops()

    def resolve_intrablock_block_size(self, strategy: str | None = None) -> int:
        strategy = normalize_strategy_name(
            self.gradient_strategy if strategy is None else strategy
        )
        if strategy != "intrablock_parallel":
            return 0
        if self.intrablock_block_size is not None:
            return max(1, int(self.intrablock_block_size))
        return 64

    def estimated_gradient_state_buffers_for(
        self, strategy: str, checkpoint_interval_ops: int | None = None
    ) -> int:
        strategy = normalize_strategy_name(strategy)
        if strategy == "inverse_walk":
            return 4
        if strategy == "save_param_states":
            return self.num_params + 3
        if strategy == "dense_scan":
            padded = 1 << (self.num_ops - 1).bit_length() if self.num_ops > 0 else 1
            return 3 * padded + max(self.num_ops + 1, padded) + 3
        if strategy == "intrablock_parallel":
            block_size = self.resolve_intrablock_block_size(strategy)
            num_blocks = int(math.ceil(self.num_ops / block_size)) if self.num_ops > 0 else 0
            return (num_blocks + 1) * 2 + num_blocks * (block_size + 1) + 4
        if strategy != "checkpoint" and strategy != "block_fused_adjoint":
            raise ValueError(
                "strategy must be 'inverse_walk', 'reverse_walk', 'save_param_states', "
                "'checkpoint', 'dense_scan', 'block_fused_adjoint', or "
                "'intrablock_parallel'."
            )
        interval = (
            self.resolve_checkpoint_interval_ops(strategy)
            if checkpoint_interval_ops is None
            else checkpoint_interval_ops
        )
        if interval == 0:
            return self.num_parametric_gates + 3
        num_chunks = int(math.ceil(self.num_ops / interval))
        if strategy == "block_fused_adjoint":
            return num_chunks + 2 * interval + 6
        return num_chunks + interval + 5

    def estimated_gradient_workspace_gib_for(
        self, strategy: str, checkpoint_interval_ops: int | None = None
    ) -> float:
        strategy = normalize_strategy_name(strategy)
        if strategy == "inverse_walk":
            return 4 * self.statevector_nbytes / (1024**3)
        if strategy == "dense_scan":
            if self.num_ops <= 0:
                return 0.0
            padded = 1 << (self.num_ops - 1).bit_length()
            matrix_count = (
                self.num_ops
                + self.num_params
                + padded
                + max(0, padded - 1)
                + 2
            )
            vector_count = (
                3 * padded
                + max(self.num_ops + 1, padded)
                + 3
            )
            bytes_total = (
                matrix_count * self.dense_matrix_nbytes
                + vector_count * self.statevector_nbytes
                + self.num_params * (8 + 4)
            )
            return bytes_total / (1024**3)
        if strategy == "intrablock_parallel":
            block_size = self.resolve_intrablock_block_size(strategy)
            num_blocks = int(math.ceil(self.num_ops / block_size)) if self.num_ops > 0 else 0
            vector_count = 2 * (num_blocks + 1) + num_blocks * (block_size + 1) + 4
            bytes_total = (
                vector_count * self.statevector_nbytes
                + self.num_ops * 56
                + self.num_params * 8
            )
            return bytes_total / (1024**3)
        return (
            self.estimated_gradient_state_buffers_for(strategy, checkpoint_interval_ops)
            * self.statevector_nbytes
            / (1024**3)
        )

    def validate(self) -> None:
        strategy = self.normalized_gradient_strategy
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2.")
        if self.layers < 1:
            raise ValueError("layers must be at least 1.")
        if strategy not in STANDALONE_GRADIENT_STRATEGIES:
            raise ValueError(
                "gradient_strategy must be 'inverse_walk', 'reverse_walk', "
                "'save_param_states', "
                "'checkpoint', 'dense_scan', 'block_fused_adjoint', or "
                "'intrablock_parallel'."
            )
        if strategy == "dense_scan" and self.num_qubits > 6:
            raise ValueError("dense_scan requires num_qubits <= 6.")
        if self.checkpoint_interval_ops is not None and self.checkpoint_interval_ops < 1:
            raise ValueError("checkpoint_interval_ops must be positive when provided.")
        if self.intrablock_block_size is not None and self.intrablock_block_size < 1:
            raise ValueError("intrablock_block_size must be positive when provided.")
