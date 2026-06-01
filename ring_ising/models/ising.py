"""Reusable Ising-model pieces for the adjoint-diff baseline."""

from __future__ import annotations

from typing import Any

import pennylane as qml
from pennylane import numpy as pnp

from ring_ising.params import make_initial_params_array


def build_ring_ising_hamiltonian(num_qubits: int, field: float) -> Any:
    """Create a transverse-field Ising Hamiltonian on a ring."""
    coeffs: list[float] = []
    ops: list[Any] = []

    for wire in range(num_qubits):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % num_qubits))

    for wire in range(num_qubits):
        coeffs.append(-field)
        ops.append(qml.PauliX(wire))

    return qml.Hamiltonian(coeffs, ops)


def apply_ring_layer(layer_params: pnp.ndarray) -> None:
    """Apply one layer of local rotations followed by ring entanglers."""
    num_qubits, _ = layer_params.shape

    for wire in range(num_qubits):
        qml.RY(layer_params[wire, 0], wires=wire)
        qml.RZ(layer_params[wire, 1], wires=wire)

    for wire in range(num_qubits):
        qml.CNOT(wires=[wire, (wire + 1) % num_qubits])


def hardware_efficient_ring(params: pnp.ndarray) -> None:
    """Apply all trainable Ising ansatz layers."""
    num_layers, _, _ = params.shape

    for layer in range(num_layers):
        apply_ring_layer(params[layer])


def make_initial_params(
    num_qubits: int, layers: int, seed: int, init_scale: float
) -> pnp.ndarray:
    """Create a trainable parameter tensor for the ansatz."""
    initial = make_initial_params_array(
        num_qubits=num_qubits,
        layers=layers,
        seed=seed,
        init_scale=init_scale,
    )
    return pnp.array(initial, requires_grad=True)
