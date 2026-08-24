"""HEA circuit definitions and the fixed benchmark Hamiltonian."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pennylane as qml

ParameterCount = Callable[[int, int], int]
CircuitBuilder = Callable[[object, int, int], None]


def _validate_size(qubits: int, layers: int) -> None:
    if isinstance(qubits, bool) or not isinstance(qubits, int) or qubits < 2:
        raise ValueError(f"qubits must be an integer >= 2, got {qubits!r}")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError(f"layers must be an integer >= 1, got {layers!r}")


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class CircuitSpec:
    """An extensible circuit definition consumed by :func:`energy_and_grad`."""

    name: str
    parameter_count_fn: ParameterCount
    builder: CircuitBuilder
    aliases: tuple[str, ...] = ()
    requires_even_qubits: bool = False
    minimum_qubits: int = 2

    def parameter_count(self, qubits: int, layers: int) -> int:
        self.validate(qubits, layers)
        count = self.parameter_count_fn(qubits, layers)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"circuit {self.name!r} returned invalid parameter count {count!r}"
            )
        return count

    def validate(self, qubits: int, layers: int) -> None:
        _validate_size(qubits, layers)
        if qubits < self.minimum_qubits:
            raise ValueError(
                f"{self.name} requires at least {self.minimum_qubits} qubits"
            )
        if self.requires_even_qubits and qubits % 2:
            raise ValueError(
                f"{self.name} requires an even number of qubits so that the periodic "
                "ring can be split into two checkerboard bond matchings"
            )

    def apply(self, params: object, qubits: int, layers: int) -> None:
        expected = self.parameter_count(qubits, layers)
        shape = qml.math.shape(params)
        if len(shape) != 1 or shape[0] != expected:
            raise ValueError(
                f"{self.name} expects a flat parameter vector of length {expected}, "
                f"got shape {shape}"
            )
        self.builder(params, qubits, layers)


CIRCUIT_REGISTRY: dict[str, CircuitSpec] = {}


def register_circuit(spec: CircuitSpec, *, overwrite: bool = False) -> None:
    """Register a circuit and all of its aliases.

    Custom circuits can be added without changing the benchmark runner. Names are
    case-insensitive; underscores and hyphens are treated equivalently.
    """

    keys = tuple(_normalise_name(value) for value in (spec.name, *spec.aliases))
    if any(not key for key in keys):
        raise ValueError("circuit names and aliases must not be empty")
    collisions = [key for key in keys if key in CIRCUIT_REGISTRY]
    if collisions and not overwrite:
        raise ValueError(f"circuit names already registered: {collisions}")
    for key in keys:
        CIRCUIT_REGISTRY[key] = spec


def get_circuit(circuit: str | CircuitSpec) -> CircuitSpec:
    if isinstance(circuit, CircuitSpec):
        return circuit
    if not isinstance(circuit, str):
        raise TypeError("circuit must be a registered name or CircuitSpec")
    key = _normalise_name(circuit)
    try:
        return CIRCUIT_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown circuit {circuit!r}; available circuits: {available_circuits()}"
        ) from exc


def available_circuits() -> tuple[str, ...]:
    """Return canonical built-in and user-registered circuit names."""

    return tuple(sorted({spec.name for spec in CIRCUIT_REGISTRY.values()}))


def _ring_cnot(qubits: int) -> None:
    for control in range(qubits):
        qml.CNOT(wires=(control, (control + 1) % qubits))


def _ra_hea(params: object, qubits: int, layers: int) -> None:
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            qml.RY(params[cursor], wires=wire)
            cursor += 1
        _ring_cnot(qubits)


def _su2_hea(params: object, qubits: int, layers: int) -> None:
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            qml.RY(params[cursor], wires=wire)
            cursor += 1
        for wire in range(qubits):
            qml.RZ(params[cursor], wires=wire)
            cursor += 1
        _ring_cnot(qubits)


def _rzz_hea(params: object, qubits: int, layers: int) -> None:
    cursor = 0
    for _ in range(layers):
        for wire in range(qubits):
            qml.RX(params[cursor], wires=wire)
            cursor += 1
        for wire in range(qubits):
            qml.RZ(params[cursor], wires=wire)
            cursor += 1

        # For even rings, each pass is a disjoint nearest-neighbour matching.
        for left in range(0, qubits, 2):
            qml.IsingZZ(params[cursor], wires=(left, (left + 1) % qubits))
            cursor += 1
        for left in range(1, qubits, 2):
            qml.IsingZZ(params[cursor], wires=(left, (left + 1) % qubits))
            cursor += 1


def _qaoa(params: object, qubits: int, layers: int) -> None:
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        beta = params[2 * layer]
        gamma = params[2 * layer + 1]
        for left in range(0, qubits, 2):
            qml.IsingZZ(gamma, wires=(left, (left + 1) % qubits))
        for left in range(1, qubits, 2):
            qml.IsingZZ(gamma, wires=(left, (left + 1) % qubits))
        for wire in range(qubits):
            qml.RX(beta, wires=wire)


def _qaoa_bd(params: object, qubits: int, layers: int) -> None:
    """QAOA with every cost RZZ explicitly decomposed into CNOT-RZ-CNOT."""
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        beta = params[2 * layer]
        gamma = params[2 * layer + 1]
        for left in range(0, qubits, 2):
            right = (left + 1) % qubits
            qml.CNOT(wires=(left, right))
            qml.RZ(gamma, wires=right)
            qml.CNOT(wires=(left, right))
        for left in range(1, qubits, 2):
            right = (left + 1) % qubits
            qml.CNOT(wires=(left, right))
            qml.RZ(gamma, wires=right)
            qml.CNOT(wires=(left, right))
        for wire in range(qubits):
            qml.RX(beta, wires=wire)


def _qaoa_ns(params: object, qubits: int, layers: int) -> None:
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        base = 2 * layer * qubits
        for edge in range(qubits):
            qml.IsingZZ(
                params[base + qubits + edge],
                wires=(edge, (edge + 1) % qubits),
            )
        for wire in range(qubits):
            qml.RX(params[base + wire], wires=wire)


def _qaoa_ns_bd(params: object, qubits: int, layers: int) -> None:
    """Non-shared-angle QAOA with every RZZ decomposed into basic gates."""
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        base = 2 * layer * qubits
        for edge in range(qubits):
            right = (edge + 1) % qubits
            gamma = params[base + qubits + edge]
            qml.CNOT(wires=(edge, right))
            qml.RZ(gamma, wires=right)
            qml.CNOT(wires=(edge, right))
        for wire in range(qubits):
            qml.RX(params[base + wire], wires=wire)


def _xxz_hva(params: object, qubits: int, layers: int) -> None:
    for wire in range(1, qubits, 2):
        qml.PauliX(wires=wire)
    for layer in range(layers):
        base = layer * 3 * qubits
        x_offset = base
        y_offset = base + qubits
        z_offset = base + 2 * qubits
        for parity in (0, 1):
            for left in range(parity, qubits, 2):
                wires = (left, (left + 1) % qubits)
                qml.IsingXX(params[x_offset + left], wires=wires)
                qml.IsingYY(params[y_offset + left], wires=wires)
                qml.IsingZZ(params[z_offset + left], wires=wires)


def _xxz_hva_bd(params: object, qubits: int, layers: int) -> None:
    """XXZ-HVA with RXX/RYY/RZZ expanded into one-qubit gates and CNOTs."""
    for wire in range(1, qubits, 2):
        qml.PauliX(wires=wire)
    for layer in range(layers):
        base = layer * 3 * qubits
        for parity in (0, 1):
            for left in range(parity, qubits, 2):
                right = (left + 1) % qubits
                # RXX = H H RZZ H H.
                qml.Hadamard(wires=left)
                qml.Hadamard(wires=right)
                qml.CNOT(wires=(left, right))
                qml.RZ(params[base + left], wires=right)
                qml.CNOT(wires=(left, right))
                qml.Hadamard(wires=left)
                qml.Hadamard(wires=right)
                # RYY = RX(pi/2) RX(pi/2) RZZ RX(-pi/2) RX(-pi/2).
                qml.RX(math.pi / 2, wires=left)
                qml.RX(math.pi / 2, wires=right)
                qml.CNOT(wires=(left, right))
                qml.RZ(params[base + qubits + left], wires=right)
                qml.CNOT(wires=(left, right))
                qml.RX(-math.pi / 2, wires=left)
                qml.RX(-math.pi / 2, wires=right)
                # RZZ.
                qml.CNOT(wires=(left, right))
                qml.RZ(params[base + 2 * qubits + left], wires=right)
                qml.CNOT(wires=(left, right))


def _mera(params: object, qubits: int, layers: int) -> None:
    active = list(range(qubits))
    cursor = 0
    while len(active) > 1:
        for index in range(1, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            qml.RY(params[cursor], wires=left)
            qml.RY(params[cursor + 1], wires=right)
            qml.CNOT(wires=(left, right))
            cursor += 2

        next_active = []
        for index in range(0, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            qml.RY(params[cursor], wires=left)
            qml.RY(params[cursor + 1], wires=right)
            qml.CNOT(wires=(left, right))
            cursor += 2
            next_active.append(right)

        if len(active) % 2:
            next_active.append(active[-1])
        active = next_active


def _equivariant_qnn(params: object, qubits: int, layers: int) -> None:
    participant_count = qubits if qubits % 2 == 0 else qubits + 1
    dummy = qubits if participant_count != qubits else None
    for layer in range(layers):
        a, b, g = (params[3 * layer + offset] for offset in range(3))
        for wire in range(qubits):
            qml.RX(a, wires=wire)
        for wire in range(qubits):
            qml.RY(b, wires=wire)
        participants = list(range(participant_count))
        for _ in range(participant_count - 1):
            for index in range(participant_count // 2):
                first, second = participants[index], participants[-1 - index]
                if first == dummy or second == dummy:
                    continue
                control, target = sorted((first, second))
                qml.CNOT(wires=(control, target))
                qml.RZ(g, wires=target)
                qml.CNOT(wires=(control, target))
            participants[1:] = participants[-1:] + participants[1:-1]


def _data_reuploading_qnn(params: object, qubits: int, layers: int) -> None:
    for layer in range(layers):
        base = 3 * layer * qubits
        for wire in range(qubits):
            qml.RZ(params[base + wire], wires=wire)
        for wire in range(qubits):
            qml.RY(params[base + qubits + wire], wires=wire)
        for wire in range(qubits):
            qml.RZ(params[base + 2 * qubits + wire], wires=wire)
        parity = layer & 1
        for left in range(parity, qubits, 2):
            right = (left + 1) % qubits
            qml.CZ(wires=(min(left, right), max(left, right)))


def build_hamiltonian(
    qubits: int, circuit: str | CircuitSpec = "su2-hea"
) -> qml.Hamiltonian:
    """Build the circuit's benchmark Hamiltonian."""

    _validate_size(qubits, 1)
    circuit_spec = get_circuit(circuit)
    coefficients: list[float] = []
    observables: list[qml.operation.Operator] = []
    if circuit_spec.name == "mera":
        return qml.Hamiltonian([1.0], [qml.PauliZ(qubits - 1)])
    if circuit_spec.name in {"qaoa", "qaoa-bd", "qaoa-ns", "qaoa-ns-bd"}:
        for wire in range(qubits):
            coefficients.extend((0.5, -0.5))
            observables.extend(
                (
                    qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % qubits),
                    qml.Identity(wire),
                )
            )
        return qml.Hamiltonian(coefficients, observables)
    if circuit_spec.name in {"xxz-hva", "xxz-hva-bd"}:
        for wire in range(qubits):
            next_wire = (wire + 1) % qubits
            coefficients.extend((1.0, 1.0, 0.5))
            observables.extend(
                (
                    qml.PauliX(wire) @ qml.PauliX(next_wire),
                    qml.PauliY(wire) @ qml.PauliY(next_wire),
                    qml.PauliZ(wire) @ qml.PauliZ(next_wire),
                )
            )
        return qml.Hamiltonian(coefficients, observables)
    if circuit_spec.name == "equivariant-qnn":
        return qml.Hamiltonian(
            [1.0 / qubits] * qubits,
            [qml.PauliX(wire) for wire in range(qubits)],
        )
    if circuit_spec.name == "data-reuploading":
        return qml.Hamiltonian([1.0], [qml.PauliZ(0)])
    for wire in range(qubits):
        coefficients.append(-1.0)
        observables.append(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % qubits))
    for wire in range(qubits):
        coefficients.append(-1.0)
        observables.append(qml.PauliX(wire))
    return qml.Hamiltonian(coefficients, observables)


register_circuit(
    CircuitSpec(
        name="ra-hea",
        aliases=("ra",),
        parameter_count_fn=lambda qubits, layers: qubits * layers,
        builder=_ra_hea,
    )
)
register_circuit(
    CircuitSpec(
        name="xxz-hva-bd",
        aliases=("xxz_hva_bd",),
        parameter_count_fn=lambda qubits, layers: 3 * qubits * layers,
        builder=_xxz_hva_bd,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="qaoa-ns-bd",
        aliases=("qaoa_ns_bd",),
        parameter_count_fn=lambda qubits, layers: 2 * qubits * layers,
        builder=_qaoa_ns_bd,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="xxz-hva",
        aliases=("xxz",),
        parameter_count_fn=lambda qubits, layers: 3 * qubits * layers,
        builder=_xxz_hva,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="qaoa",
        parameter_count_fn=lambda qubits, layers: 2 * layers,
        builder=_qaoa,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="qaoa-bd",
        aliases=("qaoa_bd",),
        parameter_count_fn=lambda qubits, layers: 2 * layers,
        builder=_qaoa_bd,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="qaoa-ns",
        aliases=("qaoa_ns", "qaoa-nonshared"),
        parameter_count_fn=lambda qubits, layers: 2 * qubits * layers,
        builder=_qaoa_ns,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
register_circuit(
    CircuitSpec(
        name="su2-hea",
        aliases=("su2",),
        parameter_count_fn=lambda qubits, layers: 2 * qubits * layers,
        builder=_su2_hea,
    )
)
register_circuit(
    CircuitSpec(
        name="rzz-hea",
        aliases=("rzz",),
        parameter_count_fn=lambda qubits, layers: 3 * qubits * layers,
        builder=_rzz_hea,
        requires_even_qubits=True,
    )
)
register_circuit(
    CircuitSpec(
        name="mera",
        parameter_count_fn=lambda qubits, layers: (
            4 * (qubits - 1) - 2 * (qubits - 1).bit_count()
            if layers == (qubits - 1).bit_length()
            else 0
        ),
        builder=_mera,
        minimum_qubits=2,
    )
)
register_circuit(
    CircuitSpec(
        name="equivariant-qnn",
        aliases=("eqnn",),
        parameter_count_fn=lambda qubits, layers: 3 * layers,
        builder=_equivariant_qnn,
        minimum_qubits=2,
    )
)
register_circuit(
    CircuitSpec(
        name="data-reuploading",
        aliases=("data-reupload", "drqnn"),
        parameter_count_fn=lambda qubits, layers: 3 * qubits * layers,
        builder=_data_reuploading_qnn,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
