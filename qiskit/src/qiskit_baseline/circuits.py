"""Qiskit QuantumCircuit builders mirroring the PennyLane circuit definitions.

Each builder returns a (QuantumCircuit, param_list, param_map, n_model_params) tuple:
  - circuit:       QuantumCircuit with distinct Parameter objects for every gate
  - param_list:    list of Parameter objects in insertion order (one per parametric gate)
  - param_map:     param_map[i] = model-parameter index for the i-th circuit parameter
  - n_model_params: total number of unique model parameters

For circuits where every model parameter appears exactly once (most circuits),
param_map is the identity [0, 1, ..., n-1].  For circuits with shared parameters
(e.g. equivariant-qnn, qaoa), param_map sums contributions back to model params.

Gate conventions (Qiskit == PennyLane for all gates used here):
  RY/RZ/RX:    exp(-i*theta/2 * Y/Z/X)
  RZZ/RXX/RYY: exp(-i*theta/2 * ZZ/XX/YY)  (Qiskit rzz/rxx/ryy)
  CNOT/CZ/H/X: non-parametric, no gradient
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_param(name: str, index: int) -> Parameter:
    return Parameter(f"{name}[{index}]")


def _pauli_str(n_qubits: int, qubit_ops: dict[int, str]) -> str:
    """Build a Qiskit Pauli string (little-endian: index 0 = rightmost = qubit 0)."""
    chars = ["I"] * n_qubits
    for qubit, pauli in qubit_ops.items():
        chars[qubit] = pauli
    return "".join(reversed(chars))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircuitBundle:
    """Everything the runner needs for a single circuit instance."""

    circuit: QuantumCircuit
    param_list: list  # ordered Parameter objects (one per parametric gate slot)
    param_map: list   # param_map[i] = model-parameter index for param_list[i]
    n_model_params: int


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _ring_cnot(qc: QuantumCircuit, qubits: int) -> None:
    for control in range(qubits):
        qc.cx(control, (control + 1) % qubits)


def _build_ra_hea(qubits: int, layers: int) -> CircuitBundle:
    n = qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.ry(p, wire)
            cursor += 1
        _ring_cnot(qc, qubits)
    return CircuitBundle(qc, param_list, list(range(n)), n)


def _build_su2_hea(qubits: int, layers: int) -> CircuitBundle:
    n = 2 * qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.ry(p, wire)
            cursor += 1
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rz(p, wire)
            cursor += 1
        _ring_cnot(qc, qubits)
    return CircuitBundle(qc, param_list, list(range(n)), n)


def _build_rzz_hea(qubits: int, layers: int) -> CircuitBundle:
    # RX(qubits) + RZ(qubits) + IsingZZ pairs(qubits) per layer = 3*qubits*layers
    n = 3 * qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rx(p, wire)
            cursor += 1
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rz(p, wire)
            cursor += 1
        for left in range(0, qubits, 2):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rzz(p, left, (left + 1) % qubits)
            cursor += 1
        for left in range(1, qubits, 2):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rzz(p, left, (left + 1) % qubits)
            cursor += 1
    return CircuitBundle(qc, param_list, list(range(n)), n)


def _build_qaoa(qubits: int, layers: int) -> CircuitBundle:
    # Model params: [beta_0, gamma_0, beta_1, gamma_1, ...] — 2*layers total.
    # beta appears in qubits RX gates per layer; gamma in qubits RZZ gates per layer.
    # We use a distinct Parameter per gate occurrence and build param_map accordingly.
    n_model = 2 * layers
    param_list: list[Parameter] = []
    param_map: list[int] = []
    qc = QuantumCircuit(qubits)
    for wire in range(qubits):
        qc.h(wire)
    for layer in range(layers):
        model_gamma = 2 * layer + 1
        model_beta = 2 * layer
        # IsingZZ(gamma) gates — even parity
        for left in range(0, qubits, 2):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_gamma)
            qc.rzz(p, left, (left + 1) % qubits)
        # IsingZZ(gamma) gates — odd parity
        for left in range(1, qubits, 2):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_gamma)
            qc.rzz(p, left, (left + 1) % qubits)
        # RX(beta) gates
        for wire in range(qubits):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_beta)
            qc.rx(p, wire)
    return CircuitBundle(qc, param_list, param_map, n_model)


def _build_qaoa_bd(qubits: int, layers: int) -> CircuitBundle:
    """QAOA with RZZ decomposed into CNOT-RZ-CNOT."""
    n_model = 2 * layers
    param_list: list[Parameter] = []
    param_map: list[int] = []
    qc = QuantumCircuit(qubits)
    for wire in range(qubits):
        qc.h(wire)
    for layer in range(layers):
        model_gamma = 2 * layer + 1
        model_beta = 2 * layer
        for left in range(0, qubits, 2):
            right = (left + 1) % qubits
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_gamma)
            qc.cx(left, right)
            qc.rz(p, right)
            qc.cx(left, right)
        for left in range(1, qubits, 2):
            right = (left + 1) % qubits
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_gamma)
            qc.cx(left, right)
            qc.rz(p, right)
            qc.cx(left, right)
        for wire in range(qubits):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_beta)
            qc.rx(p, wire)
    return CircuitBundle(qc, param_list, param_map, n_model)


def _build_qaoa_ns(qubits: int, layers: int) -> CircuitBundle:
    """QAOA non-shared: per-edge gamma and per-qubit beta — 2*qubits*layers params."""
    # params layout per layer: [beta_0..beta_{q-1}, gamma_0..gamma_{q-1}]
    # (matching PennyLane: base = 2*layer*qubits; RX uses base+wire; RZZ uses base+qubits+edge)
    n = 2 * qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    for wire in range(qubits):
        qc.h(wire)
    cursor_map: list[int] = []
    for layer in range(layers):
        base = 2 * layer * qubits
        for edge in range(qubits):
            model_idx = base + qubits + edge
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            cursor_map.append(model_idx)
            qc.rzz(p, edge, (edge + 1) % qubits)
        for wire in range(qubits):
            model_idx = base + wire
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            cursor_map.append(model_idx)
            qc.rx(p, wire)
    return CircuitBundle(qc, param_list, cursor_map, n)


def _build_qaoa_ns_bd(qubits: int, layers: int) -> CircuitBundle:
    """Non-shared QAOA with RZZ decomposed."""
    n = 2 * qubits * layers
    param_list: list[Parameter] = []
    param_map: list[int] = []
    qc = QuantumCircuit(qubits)
    for wire in range(qubits):
        qc.h(wire)
    for layer in range(layers):
        base = 2 * layer * qubits
        for edge in range(qubits):
            right = (edge + 1) % qubits
            model_idx = base + qubits + edge
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_idx)
            qc.cx(edge, right)
            qc.rz(p, right)
            qc.cx(edge, right)
        for wire in range(qubits):
            model_idx = base + wire
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_idx)
            qc.rx(p, wire)
    return CircuitBundle(qc, param_list, param_map, n)


def _build_xxz_hva(qubits: int, layers: int) -> CircuitBundle:
    n = 3 * qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    for wire in range(1, qubits, 2):
        qc.x(wire)
    for layer in range(layers):
        base = layer * 3 * qubits
        x_offset = base
        y_offset = base + qubits
        z_offset = base + 2 * qubits
        for parity in (0, 1):
            for left in range(parity, qubits, 2):
                wires = (left, (left + 1) % qubits)
                px = _new_param("θ", len(param_list))
                param_list.append(px)
                qc.rxx(px, *wires)
                py = _new_param("θ", len(param_list))
                param_list.append(py)
                qc.ryy(py, *wires)
                pz = _new_param("θ", len(param_list))
                param_list.append(pz)
                qc.rzz(pz, *wires)
    # param_map: each circuit param maps to a unique model param index
    # Order: for each layer, for each parity, for each left: x_idx, y_idx, z_idx
    param_map: list[int] = []
    for layer in range(layers):
        base = layer * 3 * qubits
        for parity in (0, 1):
            for left in range(parity, qubits, 2):
                param_map.append(base + left)            # IsingXX → x_offset + left
                param_map.append(base + qubits + left)   # IsingYY → y_offset + left
                param_map.append(base + 2 * qubits + left)  # IsingZZ → z_offset + left
    return CircuitBundle(qc, param_list, param_map, n)


def _build_xxz_hva_bd(qubits: int, layers: int) -> CircuitBundle:
    """XXZ-HVA with RXX/RYY/RZZ decomposed into CNOT+single-qubit gates."""
    n = 3 * qubits * layers
    param_list: list[Parameter] = []
    param_map: list[int] = []
    qc = QuantumCircuit(qubits)
    for wire in range(1, qubits, 2):
        qc.x(wire)
    for layer in range(layers):
        base = layer * 3 * qubits
        for parity in (0, 1):
            for left in range(parity, qubits, 2):
                right = (left + 1) % qubits
                # RXX = H H RZZ H H
                px = _new_param("θ", len(param_list))
                param_list.append(px)
                param_map.append(base + left)
                qc.h(left); qc.h(right)
                qc.cx(left, right)
                qc.rz(px, right)
                qc.cx(left, right)
                qc.h(left); qc.h(right)
                # RYY = RX(pi/2) RX(pi/2) RZZ RX(-pi/2) RX(-pi/2)
                py = _new_param("θ", len(param_list))
                param_list.append(py)
                param_map.append(base + qubits + left)
                qc.rx(math.pi / 2, left); qc.rx(math.pi / 2, right)
                qc.cx(left, right)
                qc.rz(py, right)
                qc.cx(left, right)
                qc.rx(-math.pi / 2, left); qc.rx(-math.pi / 2, right)
                # RZZ
                pz = _new_param("θ", len(param_list))
                param_list.append(pz)
                param_map.append(base + 2 * qubits + left)
                qc.cx(left, right)
                qc.rz(pz, right)
                qc.cx(left, right)
    return CircuitBundle(qc, param_list, param_map, n)


def _build_mera(qubits: int, layers: int) -> CircuitBundle:
    # layers must equal (qubits - 1).bit_length() for non-zero parameter count
    active = list(range(qubits))
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    while len(active) > 1:
        for index in range(1, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            p0 = _new_param("θ", len(param_list))
            p1 = _new_param("θ", len(param_list) + 1)
            param_list.extend([p0, p1])
            qc.ry(p0, left)
            qc.ry(p1, right)
            qc.cx(left, right)
        next_active = []
        for index in range(0, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            p0 = _new_param("θ", len(param_list))
            p1 = _new_param("θ", len(param_list) + 1)
            param_list.extend([p0, p1])
            qc.ry(p0, left)
            qc.ry(p1, right)
            qc.cx(left, right)
            next_active.append(right)
        if len(active) % 2:
            next_active.append(active[-1])
        active = next_active
    n = len(param_list)
    return CircuitBundle(qc, param_list, list(range(n)), n)


def _build_equivariant_qnn(qubits: int, layers: int) -> CircuitBundle:
    """Equivariant QNN: 3 model parameters per layer (a, b, g), shared across all qubits."""
    n_model = 3 * layers
    param_list: list[Parameter] = []
    param_map: list[int] = []
    qc = QuantumCircuit(qubits)
    participant_count = qubits if qubits % 2 == 0 else qubits + 1
    dummy = qubits if participant_count != qubits else None
    for layer in range(layers):
        model_a = 3 * layer
        model_b = 3 * layer + 1
        model_g = 3 * layer + 2
        for wire in range(qubits):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_a)
            qc.rx(p, wire)
        for wire in range(qubits):
            p = _new_param("θ", len(param_list))
            param_list.append(p)
            param_map.append(model_b)
            qc.ry(p, wire)
        participants = list(range(participant_count))
        for _ in range(participant_count - 1):
            for index in range(participant_count // 2):
                first, second = participants[index], participants[-1 - index]
                if first == dummy or second == dummy:
                    continue
                control, target = sorted((first, second))
                p = _new_param("θ", len(param_list))
                param_list.append(p)
                param_map.append(model_g)
                qc.cx(control, target)
                qc.rz(p, target)
                qc.cx(control, target)
            participants[1:] = participants[-1:] + participants[1:-1]
    return CircuitBundle(qc, param_list, param_map, n_model)


def _build_data_reuploading(qubits: int, layers: int) -> CircuitBundle:
    n = 3 * qubits * layers
    param_list: list[Parameter] = []
    qc = QuantumCircuit(qubits)
    cursor = 0
    for layer in range(layers):
        base = 3 * layer * qubits
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rz(p, wire)
            cursor += 1
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.ry(p, wire)
            cursor += 1
        for wire in range(qubits):
            p = _new_param("θ", cursor)
            param_list.append(p)
            qc.rz(p, wire)
            cursor += 1
        parity = layer & 1
        for left in range(parity, qubits, 2):
            right = (left + 1) % qubits
            qc.cz(min(left, right), max(left, right))
    return CircuitBundle(qc, param_list, list(range(n)), n)


# ---------------------------------------------------------------------------
# Hamiltonian builders (SparsePauliOp, matching PennyLane build_hamiltonian)
# ---------------------------------------------------------------------------

def build_hamiltonian(circuit_name: str, qubits: int) -> SparsePauliOp:
    name = circuit_name.lower().replace("_", "-")
    if name == "mera":
        return SparsePauliOp.from_list([(_pauli_str(qubits, {qubits - 1: "Z"}), 1.0)])

    if name in {"qaoa", "qaoa-bd"}:
        terms = []
        for wire in range(qubits):
            terms.append((_pauli_str(qubits, {wire: "Z", (wire + 1) % qubits: "Z"}), 0.5))
            terms.append((_pauli_str(qubits, {}), -0.5))  # -0.5 * I
        return SparsePauliOp.from_list(terms)

    if name in {"qaoa-ns", "qaoa-ns-bd"}:
        terms = []
        for wire in range(qubits):
            terms.append((_pauli_str(qubits, {wire: "Z", (wire + 1) % qubits: "Z"}), 0.5))
            terms.append((_pauli_str(qubits, {}), -0.5))
        return SparsePauliOp.from_list(terms)

    if name in {"xxz-hva", "xxz-hva-bd"}:
        terms = []
        for wire in range(qubits):
            nw = (wire + 1) % qubits
            terms.append((_pauli_str(qubits, {wire: "X", nw: "X"}), 1.0))
            terms.append((_pauli_str(qubits, {wire: "Y", nw: "Y"}), 1.0))
            terms.append((_pauli_str(qubits, {wire: "Z", nw: "Z"}), 0.5))
        return SparsePauliOp.from_list(terms)

    if name == "equivariant-qnn":
        terms = [(_pauli_str(qubits, {wire: "X"}), 1.0 / qubits) for wire in range(qubits)]
        return SparsePauliOp.from_list(terms)

    if name == "data-reuploading":
        return SparsePauliOp.from_list([(_pauli_str(qubits, {0: "Z"}), 1.0)])

    # Default (ra-hea, su2-hea, rzz-hea): H = -∑ ZZ_{i,i+1} - ∑ X_i
    terms = []
    for wire in range(qubits):
        terms.append((_pauli_str(qubits, {wire: "Z", (wire + 1) % qubits: "Z"}), -1.0))
    for wire in range(qubits):
        terms.append((_pauli_str(qubits, {wire: "X"}), -1.0))
    return SparsePauliOp.from_list(terms)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Callable, int, int, bool]] = {
    # name: (builder, min_qubits, min_layers, requires_even)
    "ra-hea":          (_build_ra_hea,           2, 1, False),
    "su2-hea":         (_build_su2_hea,          2, 1, False),
    "rzz-hea":         (_build_rzz_hea,          2, 1, True),
    "qaoa":            (_build_qaoa,             4, 1, True),
    "qaoa-bd":         (_build_qaoa_bd,          4, 1, True),
    "qaoa-ns":         (_build_qaoa_ns,          4, 1, True),
    "qaoa-ns-bd":      (_build_qaoa_ns_bd,       4, 1, True),
    "xxz-hva":         (_build_xxz_hva,          4, 1, True),
    "xxz-hva-bd":      (_build_xxz_hva_bd,       4, 1, True),
    "mera":            (_build_mera,             2, 1, False),
    "equivariant-qnn": (_build_equivariant_qnn,  2, 1, False),
    "data-reuploading":(_build_data_reuploading, 4, 1, True),
}

_ALIASES: dict[str, str] = {
    "ra": "ra-hea",
    "su2": "su2-hea",
    "rzz": "rzz-hea",
    "xxz": "xxz-hva",
    "xxz_hva": "xxz-hva",
    "xxz_hva_bd": "xxz-hva-bd",
    "qaoa_bd": "qaoa-bd",
    "qaoa_ns": "qaoa-ns",
    "qaoa_ns_bd": "qaoa-ns-bd",
    "qaoa-nonshared": "qaoa-ns",
    "eqnn": "equivariant-qnn",
    "data-reupload": "data-reuploading",
    "drqnn": "data-reuploading",
}


def available_circuits() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_circuit(circuit: str, qubits: int, layers: int) -> CircuitBundle:
    name = circuit.strip().lower().replace("_", "-")
    name = _ALIASES.get(name, name)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown circuit {circuit!r}; available: {available_circuits()}"
        )
    builder, min_q, min_l, req_even = _REGISTRY[name]
    if qubits < min_q:
        raise ValueError(f"{name} requires at least {min_q} qubits, got {qubits}")
    if layers < min_l:
        raise ValueError(f"{name} requires at least {min_l} layers, got {layers}")
    if req_even and qubits % 2 != 0:
        raise ValueError(f"{name} requires an even number of qubits, got {qubits}")
    return builder(qubits, layers)
