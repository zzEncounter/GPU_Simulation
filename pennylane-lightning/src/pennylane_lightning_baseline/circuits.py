"""HEA circuit definitions and the fixed benchmark Hamiltonian."""

from __future__ import annotations

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
        if self.requires_even_qubits and qubits % 2:
            raise ValueError(
                f"{self.name} requires an even number of qubits so that the periodic "
                "ring can be split into two checkerboard RZZ matchings"
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


def build_hamiltonian(qubits: int) -> qml.Hamiltonian:
    """Build ``-sum_i Z_i Z_(i+1) - sum_i X_i`` with periodic boundary."""

    _validate_size(qubits, 1)
    coefficients: list[float] = []
    observables: list[qml.operation.Operator] = []
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
