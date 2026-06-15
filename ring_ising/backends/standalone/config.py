"""Configuration and sizing helpers for the standalone CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass

from ring_ising.config import STANDALONE_GRADIENT_STRATEGIES


@dataclass(frozen=True)
class StandaloneBackendConfig:
    """Configuration for the standalone CUDA backend runtime."""

    num_qubits: int
    layers: int
    field: float
    gradient_strategy: str

    @property
    def param_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.num_qubits, 2)

    @property
    def num_params(self) -> int:
        return self.layers * self.num_qubits * 2

    @property
    def normalized_gradient_strategy(self) -> str:
        return self.gradient_strategy

    @property
    def statevector_nbytes(self) -> int:
        return (1 << self.num_qubits) * 16

    @property
    def dense_matrix_nbytes(self) -> int:
        dim = 1 << self.num_qubits
        return dim * dim * 16

    def estimated_gradient_state_buffers_for(self, strategy: str) -> int:
        if strategy == "inverse_walk":
            return 4
        if strategy == "save_param_states":
            return self.num_params + 3
        if strategy == "dense_scan":
            num_ops = self.layers * (self.num_qubits + 1)
            padded = 1 << (num_ops - 1).bit_length() if num_ops > 0 else 1
            return 3 * padded + max(num_ops + 1, padded) + 3
        raise ValueError(
            "strategy must be 'inverse_walk', 'save_param_states', or 'dense_scan'."
        )

    def estimated_gradient_workspace_gib_for(self, strategy: str) -> float:
        if strategy == "dense_scan":
            num_ops = self.layers * (self.num_qubits + 1)
            if num_ops <= 0:
                return 0.0
            padded = 1 << (num_ops - 1).bit_length()
            matrix_count = (
                num_ops
                + self.num_params
                + padded
                + max(0, padded - 1)
                + 2
            )
            vector_count = 3 * padded + max(num_ops + 1, padded) + 3
            bytes_total = (
                matrix_count * self.dense_matrix_nbytes
                + vector_count * self.statevector_nbytes
                + self.num_params * (8 + 4)
            )
            return bytes_total / (1024**3)
        return (
            self.estimated_gradient_state_buffers_for(strategy)
            * self.statevector_nbytes
            / (1024**3)
        )

    def validate(self) -> None:
        strategy = self.gradient_strategy
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2.")
        if self.layers < 1:
            raise ValueError("layers must be at least 1.")
        if strategy not in STANDALONE_GRADIENT_STRATEGIES:
            raise ValueError(
                "gradient_strategy must be 'inverse_walk', 'save_param_states', "
                "or 'dense_scan'."
            )
        if strategy == "dense_scan" and self.num_qubits > 6:
            raise ValueError("dense_scan requires num_qubits <= 6.")
