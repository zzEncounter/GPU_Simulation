"""Correctness-first inverse-walk statevector implementation.

The module intentionally applies one gate at a time.  Its gate order mirrors
the SAD circuit definitions and the inverse-walk recurrence used by
GPU_Simulation.  It uses only the Python standard library so it remains useful
as a reference when CUDA/cuStateVec is not installed.
"""

from __future__ import annotations

import cmath
import ctypes
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

supported_circuits = (
    "ra-hea", "su2-hea", "rzz-hea", "qaoa", "qaoa-ns", "xxz-hva",
    "mera", "equivariant-qnn", "data-reuploading",
)


class _NativeGate(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("wire0", ctypes.c_int),
                ("wire1", ctypes.c_int), ("parameter", ctypes.c_int),
                ("angle", ctypes.c_double)]


_NATIVE_GATE_KIND = {"rx": 0, "ry": 1, "rz": 2, "rzz": 3, "cnot": 4}
_NATIVE_CIRCUIT = {
    "ra-hea": 0, "su2-hea": 1, "rzz-hea": 2, "qaoa": 3,
    "xxz-hva": 4, "mera": 5, "equivariant-qnn": 6,
    "data-reuploading": 7, "qaoa-ns": 8,
}


def _native_library() -> ctypes.CDLL | None:
    if os.environ.get("SAD_CUQUANTUM_NATIVE", "1").lower() in {"0", "false", "no"}:
        return None
    path = Path(__file__).resolve().parents[2] / "build" / "libcuquantum_sad.so"
    if not path.exists():
        root = path.parent.parent
        completed = subprocess.run(
            ["make", "-C", str(root)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to build cuStateVec backend:\n"
                + completed.stdout + "\n" + completed.stderr
            )
    library = ctypes.CDLL(str(path))
    library.cuquantum_energy_and_grad.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(_NativeGate), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    library.cuquantum_energy_and_grad.restype = ctypes.c_int
    return library


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def expected_parameter_count(circuit: str, qubits: int, layers: int) -> int:
    name = _norm(circuit)
    if name == "qaoa":
        return 2 * layers
    if name == "qaoa-ns":
        return 2 * qubits * layers
    if name == "mera":
        if layers != (qubits - 1).bit_length():
            raise ValueError("MERA layers must equal ceil(log2(qubits))")
        return 4 * (qubits - 1) - 2 * (qubits - 1).bit_count()
    if name == "equivariant-qnn":
        return 3 * layers
    return {"ra-hea": 1, "su2-hea": 2, "rzz-hea": 3,
            "xxz-hva": 3, "data-reuploading": 3}[name] * qubits * layers


@dataclass(frozen=True)
class Gate:
    kind: str
    wires: tuple[int, ...]
    parameter: int | None = None
    angle: float = 0.0
    matrix: tuple[tuple[complex, ...], ...] | None = None
    derivative: tuple[tuple[complex, ...], ...] | None = None


def _mat(kind: str, theta: float) -> tuple[tuple[complex, ...], ...]:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if kind == "rx":
        return ((c, -1j * s), (-1j * s, c))
    if kind == "ry":
        return ((c, -s), (s, c))
    if kind == "rz":
        return ((cmath.exp(-1j * theta / 2), 0j),
                (0j, cmath.exp(1j * theta / 2)))
    raise ValueError(f"unknown one-qubit gate {kind}")


def _dmat(kind: str, theta: float) -> tuple[tuple[complex, ...], ...]:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if kind == "rx":
        return ((-s / 2, -1j * c / 2), (-1j * c / 2, -s / 2))
    if kind == "ry":
        return ((-s / 2, -c / 2), (c / 2, -s / 2))
    if kind == "rz":
        return ((-0.5j * cmath.exp(-1j * theta / 2), 0j),
                (0j, 0.5j * cmath.exp(1j * theta / 2)))
    raise ValueError(f"unknown one-qubit gate {kind}")


def _add(gates: list[Gate], kind: str, wire: int, parameter: int | None,
         angle: float) -> None:
    gates.append(Gate(kind, (wire,), parameter, angle, _mat(kind, angle),
                      _dmat(kind, angle) if parameter is not None else None))


def _add_rzz(gates: list[Gate], left: int, right: int,
             parameter: int | None, angle: float) -> None:
    # SAD diagonal gates use exp(-i theta/2 * Z_left Z_right).
    phases = [cmath.exp(-0.5j * angle), cmath.exp(0.5j * angle)]
    # Keep a diagonal matrix and its exact derivative.
    diag = (phases[0], phases[1], phases[1], phases[0])
    # d/dtheta exp(-i theta ZZ/2) = (-i ZZ/2) exp(...).
    d = tuple((-0.5j if a in (0, 3) else 0.5j) * z
              for a, z in enumerate(diag))
    gates.append(Gate("rzz", (left, right), parameter, angle,
                      tuple(tuple(diag[a] if a == b else 0j for b in range(4))
                            for a in range(4)) if parameter is not None else
                      tuple(tuple(diag[a] if a == b else 0j for b in range(4))
                            for a in range(4)),
                      tuple(tuple(d[a] if a == b else 0j for b in range(4))
                            for a in range(4)) if parameter is not None else None))


def _add_cnot(gates: list[Gate], control: int, target: int) -> None:
    gates.append(Gate("cnot", (control, target), matrix=(
        (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))))


def _product_initial(gates: list[Gate], kind: str, params: list[float],
                     offset: int, qubits: int, rz_offset: int | None = None):
    for wire in range(qubits):
        _add(gates, kind, wire, offset + wire, params[offset + wire])
        if rz_offset is not None:
            _add(gates, "rz", wire, rz_offset + wire,
                 params[rz_offset + wire])


def build_gates(circuit: str, qubits: int, layers: int,
                params: list[float]) -> list[Gate]:
    name = _norm(circuit)
    expected = expected_parameter_count(name, qubits, layers)
    if len(params) != expected:
        raise ValueError(f"parameter_count mismatch: expected {expected}, got {len(params)}")
    gates: list[Gate] = []
    if name == "ra-hea":
        _product_initial(gates, "ry", params, 0, qubits)
        for layer in range(1, layers):
            base = layer * qubits
            for wire in range(qubits): _add(gates, "ry", wire, base + wire, params[base + wire])
            for wire in range(qubits): _add_cnot(gates, wire, (wire + 1) % qubits)
        return gates
    if name == "su2-hea":
        _product_initial(gates, "ry", params, 0, qubits, qubits)
        for layer in range(1, layers):
            base = 2 * layer * qubits
            for wire in range(qubits):
                _add(gates, "ry", wire, base + wire, params[base + wire])
            for wire in range(qubits):
                _add(gates, "rz", wire, base + qubits + wire, params[base + qubits + wire])
            for wire in range(qubits): _add_cnot(gates, wire, (wire + 1) % qubits)
        return gates
    if name == "rzz-hea":
        if qubits % 2: raise ValueError("RZZ-HEA requires an even qubit count")
        _product_initial(gates, "rx", params, 0, qubits, qubits)
        for layer in range(1, layers):
            base = 3 * layer * qubits
            for wire in range(qubits): _add(gates, "rx", wire, base + wire, params[base + wire])
            for wire in range(qubits): _add(gates, "rz", wire, base + qubits + wire, params[base + qubits + wire])
            for edge in range(qubits):
                idx = base + 2 * qubits + edge
                _add_rzz(gates, edge, (edge + 1) % qubits, idx, params[idx])
        return gates
    if name in ("qaoa", "qaoa-ns"):
        for wire in range(qubits):
            # H = RY(pi/2) followed by RZ(0), represented as a fixed matrix.
            _add(gates, "ry", wire, None, math.pi / 2)
        for layer in range(layers):
            if name == "qaoa":
                beta, gamma = 2 * layer, 2 * layer + 1
                for edge in range(qubits): _add_rzz(gates, edge, (edge + 1) % qubits, gamma, params[gamma])
                for wire in range(qubits): _add(gates, "rx", wire, beta, params[beta])
            else:
                base = 2 * layer * qubits
                # Match SAD qaoa-ns layout: beta block then gamma block.
                for edge in range(qubits):
                    idx = base + qubits + edge
                    _add_rzz(gates, edge, (edge + 1) % qubits, idx, params[idx])
                for wire in range(qubits):
                    _add(gates, "rx", wire, base + wire, params[base + wire])
        return gates
    if name == "data-reuploading":
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits): _add(gates, "rz", wire, base + wire, params[base + wire])
            for wire in range(qubits): _add(gates, "ry", wire, base + qubits + wire, params[base + qubits + wire])
            for edge in range(qubits // 2):
                left = 2 * edge + (layer & 1); _add_cnot(gates, left, (left + 1) % qubits)
            for wire in range(qubits): _add(gates, "rz", wire, base + 2 * qubits + wire, params[base + 2 * qubits + wire])
        return gates
    if name == "xxz-hva":
        for wire in range(0, qubits, 2):
            _add(gates, "ry", wire, None, math.pi)
        for layer in range(layers):
            base = 3 * layer * qubits
            for parity in (0, 1):
                for edge in range(parity, qubits, 2):
                    right = (edge + 1) % qubits
                    _add_rzz(gates, edge, right, base + 2 * qubits + edge, params[base + 2 * qubits + edge])
                    _add(gates, "rx", edge, base + edge, params[base + edge])
                    _add(gates, "rx", right, base + right, params[base + right])
        return gates
    if name == "mera":
        active = list(range(qubits)); cursor = 0
        for layer in range(layers):
            pairs = [(active[i], active[i + 1]) for i in range(1, len(active) - 1, 2)]
            for left, right in pairs:
                _add(gates, "ry", left, cursor, params[cursor]); _add(gates, "ry", right, cursor + 1, params[cursor + 1]); cursor += 2; _add_cnot(gates, left, right)
            pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
            for left, right in pairs:
                _add(gates, "ry", left, cursor, params[cursor]); _add(gates, "ry", right, cursor + 1, params[cursor + 1]); cursor += 2; _add_cnot(gates, left, right)
            active = active[1::2] + (active[-1:] if len(active) % 2 else [])
        return gates
    if name == "equivariant-qnn":
        for layer in range(layers):
            base = 3 * layer
            for wire in range(qubits): _add(gates, "rx", wire, base, params[base])
            for wire in range(qubits): _add(gates, "ry", wire, base + 1, params[base + 1])
            for left in range(qubits): _add_rzz(gates, left, (left + 1) % qubits, base + 2, params[base + 2])
        return gates
    raise ValueError(f"unsupported circuit {circuit}")


def _apply(state: list[complex], gate: Gate, qubits: int, inverse: bool = False) -> list[complex]:
    matrix = gate.matrix
    if matrix is None: raise ValueError("gate has no matrix")
    if inverse: matrix = tuple(tuple(matrix[j][i].conjugate() for j in range(len(matrix))) for i in range(len(matrix)))
    out = state[:]; mask0 = 1 << gate.wires[0]
    if len(gate.wires) == 1:
        for base in range(len(state)):
            if base & mask0: continue
            pair = [base, base | mask0]
            out[pair[0]] = matrix[0][0] * state[pair[0]] + matrix[0][1] * state[pair[1]]
            out[pair[1]] = matrix[1][0] * state[pair[0]] + matrix[1][1] * state[pair[1]]
        return out
    mask1 = 1 << gate.wires[1]
    if gate.kind == "cnot":
        for i, value in enumerate(state): out[i ^ mask1 if i & mask0 else i] = value
        return out
    for base in range(len(state)):
        if base & mask0 or base & mask1: continue
        ids = [base, base | mask1, base | mask0, base | mask0 | mask1]
        vals = [state[i] for i in ids]
        for row, idx in enumerate(ids): out[idx] = sum(matrix[row][col] * vals[col] for col in range(4))
    return out


def _hamiltonian(name: str, state: list[complex], qubits: int) -> list[complex]:
    out = [0j] * len(state); name = _norm(name)
    for i, amp in enumerate(state):
        if name in ("qaoa", "qaoa-ns"):
            zz = sum(1 if ((i >> q) & 1) == ((i >> ((q + 1) % qubits)) & 1) else -1 for q in range(qubits))
            out[i] = 0.5 * (zz - qubits) * amp
        elif name == "mera":
            target = qubits - 1
            out[i] = (1 if not (i & (1 << target)) else -1) * amp
        elif name == "data-reuploading": out[i] = (1 if not (i & 1) else -1) * amp
        elif name == "equivariant-qnn": out[i] = sum(state[i ^ (1 << q)] for q in range(qubits)) / qubits
        elif name == "xxz-hva":
            zz = 0
            for q in range(qubits):
                right = (q + 1) % qubits
                unequal = ((i >> q) & 1) != ((i >> right) & 1)
                zz += -1 if unequal else 1
                if unequal:
                    out[i] += 2 * state[i ^ (1 << q) ^ (1 << right)]
            out[i] += 0.5 * zz * amp
        else:
            zz = sum(1 if ((i >> q) & 1) == ((i >> ((q + 1) % qubits)) & 1) else -1 for q in range(qubits))
            out[i] = -zz * amp - sum(state[i ^ (1 << q)] for q in range(qubits))
    return out


def _run_native(name: str, qubits: int, gates: list[Gate], parameter_count: int,
                precision: str) -> dict[str, object] | None:
    if precision != "float64":
        if os.environ.get("SAD_CUQUANTUM_NATIVE", "1").lower() not in {"0", "false", "no"}:
            raise ValueError("native cuStateVec backend currently supports float64 only")
        return None
    library = _native_library()
    if library is None:
        return None
    native_gates = (_NativeGate * len(gates))()
    for index, gate in enumerate(gates):
        if gate.kind not in _NATIVE_GATE_KIND:
            return None
        native_gates[index] = _NativeGate(
            _NATIVE_GATE_KIND[gate.kind], gate.wires[0],
            gate.wires[1] if len(gate.wires) > 1 else 0,
            gate.parameter if gate.parameter is not None else -1,
            gate.angle,
        )
    energy = ctypes.c_double()
    gradient = (ctypes.c_double * parameter_count)()
    error = ctypes.create_string_buffer(2048)
    status = library.cuquantum_energy_and_grad(
        _NATIVE_CIRCUIT[name], qubits, native_gates, len(gates),
        parameter_count, ctypes.byref(energy), gradient, error, len(error),
    )
    if status:
        raise RuntimeError(error.value.decode("utf-8", errors="replace"))
    return {
        "energy": energy.value,
        "grad": [gradient[i] for i in range(parameter_count)],
        "circuit": name,
        "qubits": qubits,
        "layers": 0,
        "precision": precision,
        "backend": "custatevec",
    }


def run(circuit: str, qubits: int, layers: int, params: Iterable[float], precision: str = "float64") -> dict:
    name = _norm(circuit)
    params = [float(x) for x in params]; gates = build_gates(name, qubits, layers, params)
    native = _run_native(name, qubits, gates, len(params), precision)
    if native is not None:
        native["layers"] = layers
        return native
    state = [0j] * (1 << qubits); state[0] = 1 + 0j
    for gate in gates: state = _apply(state, gate, qubits)
    lam = _hamiltonian(circuit, state, qubits)
    energy = sum(a.conjugate() * b for a, b in zip(state, lam)).real
    gradient = [0.0] * len(params); current = state[:]
    for gate in reversed(gates):
        previous = _apply(current, gate, qubits, inverse=True)
        if gate.parameter is not None and gate.derivative is not None:
            dgate = Gate(gate.kind, gate.wires, matrix=gate.derivative)
            dstate = _apply(previous, dgate, qubits)
            gradient[gate.parameter] += 2.0 * sum(a.conjugate() * b for a, b in zip(lam, dstate)).real
        current = previous; lam = _apply(lam, gate, qubits, inverse=True)
    return {"energy": energy, "grad": gradient, "circuit": _norm(circuit), "qubits": qubits, "layers": layers, "precision": precision}


def random_parameters(circuit: str, qubits: int, layers: int, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-0.3, 0.3) for _ in range(expected_parameter_count(circuit, qubits, layers))]
