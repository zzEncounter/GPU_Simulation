"""Configuration and sizing helpers for the standalone CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass

from ring_ising.config import (
    STRUCTURED_ROTATION_CHUNK_WIDTH_MAX,
    STANDALONE_GRADIENT_STRATEGIES,
    STANDALONE_GRADIENT_STRATEGY_ALIASES,
    SUPPORTED_STANDALONE_GRADIENT_STRATEGIES,
)

STRUCTURED_ADJOINT_GRADIENT_STRATEGIES = {"structured_adjoint"}


@dataclass(frozen=True)
class StandaloneBackendConfig:
    """Configuration for the standalone CUDA backend runtime."""

    num_qubits: int
    layers: int
    field: float
    gradient_strategy: str
    structured_rotation_chunk_width: int = 8
    mode2_rotation_chunk_width: int | None = None
    double_buffer: bool = False

    @property
    def param_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.num_qubits, 2)

    @property
    def num_params(self) -> int:
        return self.layers * self.num_qubits * 2

    @property
    def normalized_gradient_strategy(self) -> str:
        return STANDALONE_GRADIENT_STRATEGY_ALIASES.get(
            self.gradient_strategy, self.gradient_strategy
        )

    @property
    def effective_structured_rotation_chunk_width(self) -> int:
        if self.mode2_rotation_chunk_width is not None:
            return self.mode2_rotation_chunk_width
        return self.structured_rotation_chunk_width

    @property
    def statevector_nbytes(self) -> int:
        return (1 << self.num_qubits) * 16

    @property
    def dense_matrix_nbytes(self) -> int:
        dim = 1 << self.num_qubits
        return dim * dim * 16

    def estimated_gradient_state_buffers_for(self, strategy: str) -> int:
        normalized = STANDALONE_GRADIENT_STRATEGY_ALIASES.get(strategy, strategy)
        if normalized in {
            "inverse_walk_cuQuantum",
            *STRUCTURED_ADJOINT_GRADIENT_STRATEGIES,
        }:
            return 4
        if normalized == "dense_scan":
            num_ops = self.layers * 2
            padded = 1 << (num_ops - 1).bit_length() if num_ops > 0 else 1
            return 3 * padded + max(num_ops + 1, padded) + 3
        raise ValueError(
            "strategy must be one of STANDALONE_GRADIENT_STRATEGIES."
        )

    def estimated_gradient_workspace_gib_for(self, strategy: str) -> float:
        normalized = STANDALONE_GRADIENT_STRATEGY_ALIASES.get(strategy, strategy)
        if normalized == "dense_scan":
            num_ops = self.layers * 2
            if num_ops <= 0:
                return 0.0
            padded = 1 << (num_ops - 1).bit_length()
            matrix_count = num_ops + padded + max(0, padded - 1) + 2
            vector_count = 3 * padded + max(num_ops + 1, padded) + 3
            bytes_total = (
                matrix_count * self.dense_matrix_nbytes
                + vector_count * self.statevector_nbytes
                + self.num_params * (8 + 8 + 4)
            )
            return bytes_total / (1024**3)
        return (
            self.estimated_gradient_state_buffers_for(strategy)
            * self.statevector_nbytes
            / (1024**3)
        )

    def validate(self) -> None:
        strategy = self.normalized_gradient_strategy
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2.")
        if self.layers < 1:
            raise ValueError("layers must be at least 1.")
        if self.gradient_strategy not in SUPPORTED_STANDALONE_GRADIENT_STRATEGIES:
            raise ValueError(
            "gradient_strategy must be one of public strategies "
                f"{STANDALONE_GRADIENT_STRATEGIES!r}; experimental/legacy "
                f"accepted strategies are {SUPPORTED_STANDALONE_GRADIENT_STRATEGIES!r}."
            )
        if strategy == "dense_scan" and self.num_qubits > 8:
            raise ValueError("dense_scan requires num_qubits <= 8.")
        chunk_width = self.effective_structured_rotation_chunk_width
        if chunk_width < 1:
            raise ValueError(
                "structured_rotation_chunk_width must be at least 1."
            )
        if chunk_width > STRUCTURED_ROTATION_CHUNK_WIDTH_MAX:
            raise ValueError(
                "structured_rotation_chunk_width must be at most "
                f"{STRUCTURED_ROTATION_CHUNK_WIDTH_MAX}."
            )
