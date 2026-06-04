"""Reference implementations for standalone adjoint-gradient strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


OpKind = Literal["fused_ryrz", "cnot", "ring_cnot_layer"]


@dataclass(frozen=True)
class ReferenceOp:
    kind: OpKind
    wire0: int
    wire1: int
    theta0: float
    theta1: float
    param_index0: int
    param_index1: int
    is_parametric: bool


@dataclass(frozen=True)
class ReferenceTrace:
    ops: tuple[ReferenceOp, ...]
    forward_states: np.ndarray
    backward_states: np.ndarray
    gradient: np.ndarray
    energy: float
    block_propagators: tuple[np.ndarray, ...]


def build_reference_ops(
    *,
    num_qubits: int,
    layers: int,
    params: np.ndarray,
    gate_fusion: bool,
) -> tuple[ReferenceOp, ...]:
    flat = np.asarray(params, dtype=np.float64).reshape(-1)
    ops: list[ReferenceOp] = []
    for layer in range(layers):
        for wire in range(num_qubits):
            base = (layer * num_qubits + wire) * 2
            ops.append(
                ReferenceOp(
                    kind="fused_ryrz",
                    wire0=wire,
                    wire1=0,
                    theta0=float(flat[base]),
                    theta1=float(flat[base + 1]),
                    param_index0=base,
                    param_index1=base + 1,
                    is_parametric=True,
                )
            )
        if gate_fusion:
            ops.append(
                ReferenceOp(
                    kind="ring_cnot_layer",
                    wire0=0,
                    wire1=0,
                    theta0=0.0,
                    theta1=0.0,
                    param_index0=0,
                    param_index1=0,
                    is_parametric=False,
                )
            )
        else:
            for wire in range(num_qubits):
                ops.append(
                    ReferenceOp(
                        kind="cnot",
                        wire0=wire,
                        wire1=(wire + 1) % num_qubits,
                        theta0=0.0,
                        theta1=0.0,
                        param_index0=0,
                        param_index1=0,
                        is_parametric=False,
                    )
                )
    return tuple(ops)


def zero_state(num_qubits: int, dtype: np.dtype) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=dtype)
    state[0] = 1.0
    return state


def apply_ring_ising_hamiltonian(
    state: np.ndarray,
    *,
    num_qubits: int,
    field: float,
) -> np.ndarray:
    dtype = state.dtype
    dim = state.shape[0]
    out = np.zeros(dim, dtype=dtype)
    for index in range(dim):
        diag_coeff = 0.0
        for wire in range(num_qubits):
            next_wire = (wire + 1) % num_qubits
            zi = -1.0 if ((index >> wire) & 1) else 1.0
            zj = -1.0 if ((index >> next_wire) & 1) else 1.0
            diag_coeff += -(zi * zj)
        value = dtype.type(diag_coeff) * state[index]
        for wire in range(num_qubits):
            value += dtype.type(-field) * state[index ^ (1 << wire)]
        out[index] = value
    return out


def apply_fused_ryrz(state: np.ndarray, *, wire: int, theta: float, phi: float) -> np.ndarray:
    dim = state.shape[0]
    out = state.copy()
    half_stride = 1 << wire
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    phase0 = np.exp(-0.5j * phi).astype(state.dtype)
    phase1 = np.exp(0.5j * phi).astype(state.dtype)
    for pair_index in range(dim // 2):
        low_mask = half_stride - 1
        base = ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask)
        i0 = base
        i1 = base | half_stride
        a0 = state[i0]
        a1 = state[i1]
        b0 = c * a0 - s * a1
        b1 = s * a0 + c * a1
        out[i0] = phase0 * b0
        out[i1] = phase1 * b1
    return out


def apply_fused_ryrz_inverse(
    state: np.ndarray, *, wire: int, theta: float, phi: float
) -> np.ndarray:
    dim = state.shape[0]
    out = state.copy()
    half_stride = 1 << wire
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    phase0 = np.exp(-0.5j * phi).astype(state.dtype)
    phase1 = np.exp(0.5j * phi).astype(state.dtype)
    for pair_index in range(dim // 2):
        low_mask = half_stride - 1
        base = ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask)
        i0 = base
        i1 = base | half_stride
        a0 = state[i0]
        a1 = state[i1]
        out[i0] = phase1 * c * a0 + phase0 * s * a1
        out[i1] = phase1 * (-s) * a0 + phase0 * c * a1
    return out


def apply_cnot(state: np.ndarray, *, control: int, target: int) -> np.ndarray:
    out = np.empty_like(state)
    control_mask = 1 << control
    target_mask = 1 << target
    for index in range(state.shape[0]):
        transformed = index
        if transformed & control_mask:
            transformed ^= target_mask
        out[transformed] = state[index]
    return out


def apply_ring_cnot_layer(
    state: np.ndarray,
    *,
    num_qubits: int,
    inverse: bool = False,
) -> np.ndarray:
    out = np.empty_like(state)
    for index in range(state.shape[0]):
        transformed = index
        if not inverse:
            for wire in range(num_qubits):
                if (transformed >> wire) & 1:
                    transformed ^= 1 << ((wire + 1) % num_qubits)
        else:
            for wire in range(num_qubits - 1, -1, -1):
                if (transformed >> wire) & 1:
                    transformed ^= 1 << ((wire + 1) % num_qubits)
        out[transformed] = state[index]
    return out


def apply_op(state: np.ndarray, op: ReferenceOp, *, num_qubits: int, inverse: bool) -> np.ndarray:
    if op.kind == "fused_ryrz":
        if inverse:
            return apply_fused_ryrz_inverse(state, wire=op.wire0, theta=op.theta0, phi=op.theta1)
        return apply_fused_ryrz(state, wire=op.wire0, theta=op.theta0, phi=op.theta1)
    if op.kind == "cnot":
        return apply_cnot(state, control=op.wire0, target=op.wire1)
    if op.kind == "ring_cnot_layer":
        return apply_ring_cnot_layer(state, num_qubits=num_qubits, inverse=inverse)
    raise ValueError(f"Unsupported op kind {op.kind!r}.")


def apply_local_gate_to_all_columns(
    matrix: np.ndarray,
    op: ReferenceOp,
    *,
    num_qubits: int,
    inverse: bool = False,
) -> np.ndarray:
    out = np.empty_like(matrix)
    dim = matrix.shape[0]
    if op.kind == "fused_ryrz":
        wire = op.wire0
        half_stride = 1 << wire
        c = np.cos(op.theta0 * 0.5)
        s = np.sin(op.theta0 * 0.5)
        phase0 = np.exp(-0.5j * op.theta1).astype(matrix.dtype)
        phase1 = np.exp(0.5j * op.theta1).astype(matrix.dtype)
        for pair_index in range(dim // 2):
            low_mask = half_stride - 1
            base = ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask)
            i0 = base
            i1 = base | half_stride
            a0 = matrix[i0, :]
            a1 = matrix[i1, :]
            if not inverse:
                b0 = c * a0 - s * a1
                b1 = s * a0 + c * a1
                out[i0, :] = phase0 * b0
                out[i1, :] = phase1 * b1
            else:
                out[i0, :] = phase1 * c * a0 + phase0 * s * a1
                out[i1, :] = phase1 * (-s) * a0 + phase0 * c * a1
        return out
    if op.kind == "cnot":
        for index in range(dim):
            transformed = index
            if (transformed >> op.wire0) & 1:
                transformed ^= 1 << op.wire1
            out[transformed, :] = matrix[index, :]
        return out
    if op.kind == "ring_cnot_layer":
        for index in range(dim):
            transformed = index
            if not inverse:
                for wire in range(num_qubits):
                    if (transformed >> wire) & 1:
                        transformed ^= 1 << ((wire + 1) % num_qubits)
            else:
                for wire in range(num_qubits - 1, -1, -1):
                    if (transformed >> wire) & 1:
                        transformed ^= 1 << ((wire + 1) % num_qubits)
            out[transformed, :] = matrix[index, :]
        return out
    raise ValueError(f"Unsupported op kind {op.kind!r}.")


def fused_ryrz_gradients(
    state: np.ndarray,
    lambda_state: np.ndarray,
    *,
    wire: int,
    theta: float,
    phi: float,
) -> tuple[float, float]:
    dim = state.shape[0]
    half_stride = 1 << wire
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    phase0 = np.exp(-0.5j * phi).astype(state.dtype)
    phase1 = np.exp(0.5j * phi).astype(state.dtype)
    pref0 = np.array(-0.5j, dtype=state.dtype)
    pref1 = np.array(0.5j, dtype=state.dtype)
    grad_theta = 0.0
    grad_phi = 0.0
    for pair_index in range(dim // 2):
        low_mask = half_stride - 1
        base = ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask)
        i0 = base
        i1 = base | half_stride
        a0 = state[i0]
        a1 = state[i1]
        l0 = lambda_state[i0]
        l1 = lambda_state[i1]
        dtheta0 = phase0 * ((-0.5 * s) * a0 + (-0.5 * c) * a1)
        dtheta1 = phase1 * ((0.5 * c) * a0 + (-0.5 * s) * a1)
        grad_theta += np.real(np.conj(l0) * dtheta0 + np.conj(l1) * dtheta1)
        b0 = c * a0 - s * a1
        b1 = s * a0 + c * a1
        dphi0 = pref0 * phase0 * b0
        dphi1 = pref1 * phase1 * b1
        grad_phi += np.real(np.conj(l0) * dphi0 + np.conj(l1) * dphi1)
    return 2.0 * float(grad_theta), 2.0 * float(grad_phi)


def ordinary_forward(block_ops: tuple[ReferenceOp, ...], psi0: np.ndarray, *, num_qubits: int) -> np.ndarray:
    state = np.array(psi0, copy=True)
    for op in block_ops:
        state = apply_op(state, op, num_qubits=num_qubits, inverse=False)
    return state


def sequential_adjoint_trace(
    *,
    num_qubits: int,
    layers: int,
    field: float,
    params: np.ndarray,
    gate_fusion: bool,
    dtype: np.dtype,
) -> ReferenceTrace:
    ops = build_reference_ops(
        num_qubits=num_qubits,
        layers=layers,
        params=params,
        gate_fusion=gate_fusion,
    )
    state = zero_state(num_qubits, dtype)
    forward_states = [state.copy()]
    for op in ops:
        state = apply_op(state, op, num_qubits=num_qubits, inverse=False)
        forward_states.append(state.copy())

    lambda_state = apply_ring_ising_hamiltonian(state, num_qubits=num_qubits, field=field)
    energy = float(np.real(np.vdot(state, lambda_state)))
    backward_states = [np.zeros_like(state) for _ in ops]
    gradient = np.zeros(layers * num_qubits * 2, dtype=np.float64)
    for op_index in range(len(ops) - 1, -1, -1):
        op = ops[op_index]
        backward_states[op_index] = lambda_state.copy()
        if op.is_parametric:
            grad_theta, grad_phi = fused_ryrz_gradients(
                forward_states[op_index],
                lambda_state,
                wire=op.wire0,
                theta=op.theta0,
                phi=op.theta1,
            )
            gradient[op.param_index0] = grad_theta
            gradient[op.param_index1] = grad_phi
        lambda_state = apply_op(lambda_state, op, num_qubits=num_qubits, inverse=True)

    return ReferenceTrace(
        ops=ops,
        forward_states=np.stack(forward_states),
        backward_states=np.stack(backward_states),
        gradient=gradient.reshape(layers, num_qubits, 2),
        energy=energy,
        block_propagators=(),
    )


def intrablock_reference_trace(
    *,
    num_qubits: int,
    layers: int,
    field: float,
    params: np.ndarray,
    gate_fusion: bool,
    block_size: int,
    dtype: np.dtype,
) -> ReferenceTrace:
    ops = build_reference_ops(
        num_qubits=num_qubits,
        layers=layers,
        params=params,
        gate_fusion=gate_fusion,
    )
    dim = 1 << num_qubits
    num_ops = len(ops)
    forward_states: list[np.ndarray] = []
    backward_states = [np.zeros(dim, dtype=dtype) for _ in ops]
    block_propagators: list[np.ndarray] = []
    block_prefixes: list[np.ndarray] = []
    block_boundaries = [zero_state(num_qubits, dtype)]

    op_cursor = 0
    while op_cursor < num_ops:
        block_ops = ops[op_cursor : op_cursor + block_size]
        prefix = np.empty((len(block_ops) + 1, dim, dim), dtype=dtype)
        prefix[0] = np.eye(dim, dtype=dtype)
        for local_index, op in enumerate(block_ops, start=1):
            prefix[local_index] = apply_local_gate_to_all_columns(
                prefix[local_index - 1], op, num_qubits=num_qubits, inverse=False
            )
        block_prefixes.append(prefix)
        block_propagators.append(prefix[-1].copy())
        psi_start = block_boundaries[-1]
        block_states = np.einsum("bij,j->bi", prefix, psi_start, optimize=True)
        forward_states.extend(block_states[:-1])
        block_boundaries.append(block_states[-1].copy())
        op_cursor += len(block_ops)

    psi_final = block_boundaries[-1]
    lambda_end = apply_ring_ising_hamiltonian(psi_final, num_qubits=num_qubits, field=field)
    energy = float(np.real(np.vdot(psi_final, lambda_end)))
    gradient = np.zeros(layers * num_qubits * 2, dtype=np.float64)

    for block_idx in range(len(block_prefixes) - 1, -1, -1):
        prefix = block_prefixes[block_idx]
        block_ops = ops[
            block_idx * block_size : min((block_idx + 1) * block_size, num_ops)
        ]
        psi_start = block_boundaries[block_idx]
        block_states = np.einsum("bij,j->bi", prefix, psi_start, optimize=True)
        block_length = len(block_ops)
        suffix = np.empty((block_length, dim, dim), dtype=dtype)
        suffix[-1] = np.eye(dim, dtype=dtype)
        running = suffix[-1]
        for local_index in range(block_length - 1, 0, -1):
            running = apply_local_gate_to_all_columns(
                running, block_ops[local_index], num_qubits=num_qubits, inverse=True
            )
            suffix[local_index - 1] = running
        lambda_states = np.einsum("bij,j->bi", suffix, lambda_end, optimize=True)
        for local_index, op in enumerate(block_ops):
            backward_states[block_idx * block_size + local_index] = lambda_states[local_index]
            if op.is_parametric:
                grad_theta, grad_phi = fused_ryrz_gradients(
                    block_states[local_index],
                    lambda_states[local_index],
                    wire=op.wire0,
                    theta=op.theta0,
                    phi=op.theta1,
                )
                gradient[op.param_index0] = grad_theta
                gradient[op.param_index1] = grad_phi
        lambda_end = block_propagators[block_idx].conj().T @ lambda_end

    return ReferenceTrace(
        ops=ops,
        forward_states=np.stack(forward_states + [psi_final]),
        backward_states=np.stack(backward_states),
        gradient=gradient.reshape(layers, num_qubits, 2),
        energy=energy,
        block_propagators=tuple(block_propagators),
    )


__all__ = [
    "ReferenceOp",
    "ReferenceTrace",
    "apply_local_gate_to_all_columns",
    "build_reference_ops",
    "intrablock_reference_trace",
    "ordinary_forward",
    "sequential_adjoint_trace",
    "zero_state",
]
