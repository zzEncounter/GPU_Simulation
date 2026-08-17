"""Low-level Lightning-GPU energy-and-adjoint-gradient benchmark.

This runner bypasses PennyLane's QNode, Autograd interface, device transforms,
and per-call tape serialization.  Circuit/observable construction and native
``OpsData`` creation happen once.  Each measured step contains synchronized
state reset, forward gate application, Hamiltonian expectation, adjoint
Jacobian evaluation, and shared-parameter gradient reduction.

The forward path still crosses the ``lightning_gpu_ops`` nanobind boundary once
per gate.  This is therefore a low-level packaged-binding baseline rather than
a standalone C++ Lightning benchmark.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from .circuits import CircuitSpec, get_circuit
from .memory import MemorySnapshot, take_memory_snapshot
from .runner import (
    MemoryUsage,
    Precision,
    _build_memory_usage,
    _resolve_precision,
    _validate_positive_integer,
)


@dataclass(frozen=True)
class NativeOperation:
    """One operation in the low-level Lightning execution stream."""

    name: str
    wires: tuple[int, ...]
    parameters: tuple[float, ...] = ()
    source_parameter: int | None = None


@dataclass(frozen=True)
class NativeEnergyGradResult:
    """Result of repeated low-level Lightning-GPU evaluations."""

    energy: float
    grad: np.ndarray
    step_times_s: tuple[float, ...]
    forward_times_s: tuple[float, ...]
    hamiltonian_times_s: tuple[float, ...]
    backward_times_s: tuple[float, ...]
    memory: MemoryUsage
    circuit: str
    qubits: int
    layers: int
    parameter_count: int
    precision: str
    random_seed: int
    batches: int
    warmup_steps: int
    device_name: str
    execution_scope: str

    def __iter__(self) -> Iterator[object]:
        yield self.energy
        yield self.grad
        yield self.step_times_s
        yield self.memory

    @property
    def mean_step_time_s(self) -> float:
        return statistics.fmean(self.step_times_s)

    @property
    def median_step_time_s(self) -> float:
        return statistics.median(self.step_times_s)

    def as_dict(self, *, include_grad: bool = True) -> dict[str, object]:
        payload = asdict(self)
        payload["grad"] = self.grad.copy() if include_grad else None
        return payload


@dataclass(frozen=True)
class _NativeBindings:
    state_vector: type
    measurements: type
    adjoint_jacobian: type
    create_ops_list: Callable[..., object]
    named_observable: type
    tensor_observable: type
    hamiltonian: type


@lru_cache(maxsize=1)
def _cuda_runtime() -> ctypes.CDLL:
    """Load the CUDA runtime used to establish synchronized phase boundaries."""

    candidates = (
        "libcudart.so.12",
        ctypes.util.find_library("cudart"),
        "libcudart.so",
    )
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        synchronize = library.cudaDeviceSynchronize
        synchronize.argtypes = []
        synchronize.restype = ctypes.c_int
        return library
    raise RuntimeError("could not load CUDA runtime: " + "; ".join(errors))


def _synchronize() -> None:
    status = _cuda_runtime().cudaDeviceSynchronize()
    if status != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed with status {status}")


@lru_cache(maxsize=2)
def _native_bindings(precision_name: str) -> _NativeBindings:
    try:
        from pennylane_lightning import lightning_gpu_ops
    except ImportError as exc:
        raise RuntimeError(
            "pennylane-lightning-gpu native bindings are unavailable"
        ) from exc

    suffix = "64" if precision_name == "float32" else "128"
    algorithms = lightning_gpu_ops.algorithms
    observables = lightning_gpu_ops.observables
    return _NativeBindings(
        state_vector=getattr(lightning_gpu_ops, f"StateVectorC{suffix}"),
        measurements=getattr(lightning_gpu_ops, f"MeasurementsC{suffix}"),
        adjoint_jacobian=getattr(algorithms, f"AdjointJacobianC{suffix}"),
        create_ops_list=getattr(algorithms, f"create_ops_listC{suffix}"),
        named_observable=getattr(observables, f"NamedObsC{suffix}"),
        tensor_observable=getattr(observables, f"TensorProdObsC{suffix}"),
        hamiltonian=getattr(observables, f"HamiltonianC{suffix}"),
    )


def _append_parameterized(
    operations: list[NativeOperation],
    name: str,
    wires: Sequence[int],
    params: np.ndarray,
    parameter: int,
) -> None:
    operations.append(
        NativeOperation(
            name=name,
            wires=tuple(wires),
            parameters=(float(params[parameter]),),
            source_parameter=parameter,
        )
    )


def _build_operations(
    circuit: CircuitSpec, qubits: int, layers: int, params: np.ndarray
) -> tuple[NativeOperation, ...]:
    """Build the registered benchmark circuits without creating a PennyLane tape."""

    operations: list[NativeOperation] = []
    name = circuit.name
    if name == "ra-hea":
        cursor = 0
        for _ in range(layers):
            for wire in range(qubits):
                _append_parameterized(operations, "RY", (wire,), params, cursor)
                cursor += 1
            for wire in range(qubits):
                operations.append(
                    NativeOperation("CNOT", (wire, (wire + 1) % qubits))
                )
        return tuple(operations)

    if name == "su2-hea":
        cursor = 0
        for _ in range(layers):
            for wire in range(qubits):
                _append_parameterized(operations, "RY", (wire,), params, cursor)
                cursor += 1
            for wire in range(qubits):
                _append_parameterized(operations, "RZ", (wire,), params, cursor)
                cursor += 1
            for wire in range(qubits):
                operations.append(
                    NativeOperation("CNOT", (wire, (wire + 1) % qubits))
                )
        return tuple(operations)

    if name == "rzz-hea":
        cursor = 0
        for _ in range(layers):
            for wire in range(qubits):
                _append_parameterized(operations, "RX", (wire,), params, cursor)
                cursor += 1
            for wire in range(qubits):
                _append_parameterized(operations, "RZ", (wire,), params, cursor)
                cursor += 1
            for left in range(0, qubits, 2):
                _append_parameterized(
                    operations,
                    "IsingZZ",
                    (left, (left + 1) % qubits),
                    params,
                    cursor,
                )
                cursor += 1
            for left in range(1, qubits, 2):
                _append_parameterized(
                    operations,
                    "IsingZZ",
                    (left, (left + 1) % qubits),
                    params,
                    cursor,
                )
                cursor += 1
        return tuple(operations)

    if name == "qaoa":
        for wire in range(qubits):
            operations.append(NativeOperation("Hadamard", (wire,)))
        for layer in range(layers):
            beta = 2 * layer
            gamma = beta + 1
            for left in range(0, qubits, 2):
                _append_parameterized(
                    operations,
                    "IsingZZ",
                    (left, (left + 1) % qubits),
                    params,
                    gamma,
                )
            for left in range(1, qubits, 2):
                _append_parameterized(
                    operations,
                    "IsingZZ",
                    (left, (left + 1) % qubits),
                    params,
                    gamma,
                )
            for wire in range(qubits):
                _append_parameterized(operations, "RX", (wire,), params, beta)
        return tuple(operations)

    if name == "qaoa-ns":
        for wire in range(qubits):
            operations.append(NativeOperation("Hadamard", (wire,)))
        for layer in range(layers):
            base = 2 * layer * qubits
            for edge in range(qubits):
                _append_parameterized(
                    operations,
                    "IsingZZ",
                    (edge, (edge + 1) % qubits),
                    params,
                    base + qubits + edge,
                )
            for wire in range(qubits):
                _append_parameterized(
                    operations, "RX", (wire,), params, base + wire
                )
        return tuple(operations)

    if name == "xxz-hva":
        for wire in range(1, qubits, 2):
            operations.append(NativeOperation("PauliX", (wire,)))
        for layer in range(layers):
            base = layer * 3 * qubits
            for parity in (0, 1):
                for left in range(parity, qubits, 2):
                    wires = (left, (left + 1) % qubits)
                    _append_parameterized(
                        operations, "IsingXX", wires, params, base + left
                    )
                    _append_parameterized(
                        operations, "IsingYY", wires, params, base + qubits + left
                    )
                    _append_parameterized(
                        operations,
                        "IsingZZ",
                        wires,
                        params,
                        base + 2 * qubits + left,
                    )
        return tuple(operations)

    if name == "mera":
        active = list(range(qubits))
        cursor = 0
        while len(active) > 1:
            for index in range(1, len(active) - 1, 2):
                left, right = active[index], active[index + 1]
                _append_parameterized(operations, "RY", (left,), params, cursor)
                _append_parameterized(
                    operations, "RY", (right,), params, cursor + 1
                )
                operations.append(NativeOperation("CNOT", (left, right)))
                cursor += 2

            next_active = []
            for index in range(0, len(active) - 1, 2):
                left, right = active[index], active[index + 1]
                _append_parameterized(operations, "RY", (left,), params, cursor)
                _append_parameterized(
                    operations, "RY", (right,), params, cursor + 1
                )
                operations.append(NativeOperation("CNOT", (left, right)))
                cursor += 2
                next_active.append(right)
            if len(active) % 2:
                next_active.append(active[-1])
            active = next_active
        return tuple(operations)

    if name == "equivariant-qnn":
        participant_count = qubits if qubits % 2 == 0 else qubits + 1
        dummy = qubits if participant_count != qubits else None
        for layer in range(layers):
            base = 3 * layer
            for wire in range(qubits):
                _append_parameterized(operations, "RX", (wire,), params, base)
            for wire in range(qubits):
                _append_parameterized(operations, "RY", (wire,), params, base + 1)
            participants = list(range(participant_count))
            for _ in range(participant_count - 1):
                for index in range(participant_count // 2):
                    first, second = participants[index], participants[-1 - index]
                    if first == dummy or second == dummy:
                        continue
                    control, target = sorted((first, second))
                    operations.append(NativeOperation("CNOT", (control, target)))
                    _append_parameterized(operations, "RZ", (target,), params, base + 2)
                    operations.append(NativeOperation("CNOT", (control, target)))
                participants[1:] = participants[-1:] + participants[1:-1]
        return tuple(operations)

    if name == "data-reuploading":
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits):
                _append_parameterized(operations, "RZ", (wire,), params, base + wire)
            for wire in range(qubits):
                _append_parameterized(
                    operations, "RY", (wire,), params, base + qubits + wire
                )
            for wire in range(qubits):
                _append_parameterized(
                    operations, "RZ", (wire,), params, base + 2 * qubits + wire
                )
            parity = layer & 1
            for left in range(parity, qubits, 2):
                operations.append(
                    NativeOperation("CZ", (left, (left + 1) % qubits))
                )
        return tuple(operations)

    raise ValueError(f"native Lightning runner does not support circuit {name!r}")


def _build_hamiltonian(
    bindings: _NativeBindings, circuit: CircuitSpec, qubits: int
) -> object:
    named = bindings.named_observable
    tensor = bindings.tensor_observable
    coefficients: list[float] = []
    terms: list[object] = []

    def pauli_product(pauli: str, left: int, right: int) -> object:
        return tensor([named(pauli, [left]), named(pauli, [right])])

    if circuit.name == "mera":
        coefficients.append(1.0)
        terms.append(named("PauliZ", [qubits - 1]))
    elif circuit.name == "equivariant-qnn":
        coefficients = [1.0 / qubits] * qubits
        terms = [named("PauliX", [wire]) for wire in range(qubits)]
    elif circuit.name == "data-reuploading":
        coefficients = [1.0]
        terms = [named("PauliZ", [0])]
    elif circuit.name in {"qaoa", "qaoa-ns"}:
        for wire in range(qubits):
            coefficients.extend((0.5, -0.5))
            terms.extend(
                (
                    pauli_product("PauliZ", wire, (wire + 1) % qubits),
                    named("Identity", [wire]),
                )
            )
    elif circuit.name == "xxz-hva":
        for wire in range(qubits):
            next_wire = (wire + 1) % qubits
            coefficients.extend((1.0, 1.0, 0.5))
            terms.extend(
                (
                    pauli_product("PauliX", wire, next_wire),
                    pauli_product("PauliY", wire, next_wire),
                    pauli_product("PauliZ", wire, next_wire),
                )
            )
    else:
        for wire in range(qubits):
            coefficients.append(-1.0)
            terms.append(pauli_product("PauliZ", wire, (wire + 1) % qubits))
        for wire in range(qubits):
            coefficients.append(-1.0)
            terms.append(named("PauliX", [wire]))
    return bindings.hamiltonian(coefficients, terms)


def _build_ops_data(
    bindings: _NativeBindings,
    precision: Precision,
    operations: tuple[NativeOperation, ...],
) -> object:
    count = len(operations)
    empty_matrices = [np.empty(0, dtype=precision.complex_dtype) for _ in range(count)]
    empty_controls = [[] for _ in range(count)]
    return bindings.create_ops_list(
        [operation.name for operation in operations],
        [list(operation.parameters) for operation in operations],
        [list(operation.wires) for operation in operations],
        [False] * count,
        empty_matrices,
        empty_controls,
        [[] for _ in range(count)],
    )


def _measure_phase(function: Callable[[], object]) -> tuple[float, object]:
    _synchronize()
    started = time.perf_counter()
    value = function()
    _synchronize()
    return time.perf_counter() - started, value


def native_energy_and_grad(
    circuit: str | CircuitSpec = "su2-hea",
    random_seed: int = 42,
    scalability: tuple[int, int] = (16, 8),
    batches: int = 1,
    precision: str = "float64",
    steps: int = 5,
    *,
    warmup_steps: int = 1,
    device_name: str = "lightning.gpu.ops",
) -> NativeEnergyGradResult:
    """Evaluate energy and full adjoint gradient through ``lightning_gpu_ops``."""

    if batches != 1:
        raise ValueError(f"batches must be 1, got {batches!r}")
    if device_name != "lightning.gpu.ops":
        raise ValueError(
            f"device_name must be 'lightning.gpu.ops', got {device_name!r}"
        )
    _validate_positive_integer(steps, "steps")
    _validate_positive_integer(warmup_steps, "warmup_steps", allow_zero=True)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    if (
        not isinstance(scalability, tuple)
        or len(scalability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in scalability)
    ):
        raise ValueError("scalability must be a (qubits, layers) integer tuple")

    qubits, layers = scalability
    circuit_spec = get_circuit(circuit)
    parameter_count = circuit_spec.parameter_count(qubits, layers)
    precision_spec = _resolve_precision(precision)
    bindings = _native_bindings(precision_spec.name)

    params = np.ascontiguousarray(
        np.random.default_rng(random_seed).uniform(-math.pi, math.pi, parameter_count),
        dtype=precision_spec.real_dtype,
    )
    if circuit_spec.name == "data-reuploading":
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits):
                feature = 2.0 * (wire + 1) / (qubits + 1) - 1.0
                params[base + wire] += feature
                params[base + qubits + wire] += 0.5 * feature
                params[base + 2 * qubits + wire] -= 0.5 * feature
    operations = _build_operations(circuit_spec, qubits, layers, params)
    parameter_sources = np.asarray(
        [
            operation.source_parameter
            for operation in operations
            if operation.source_parameter is not None
        ],
        dtype=np.intp,
    )
    ops_data = _build_ops_data(bindings, precision_spec, operations)
    trainable_params = list(range(len(parameter_sources)))
    hamiltonian = _build_hamiltonian(bindings, circuit_spec, qubits)

    memory_before = take_memory_snapshot()
    state_vector = bindings.state_vector(qubits)
    measurements = bindings.measurements(state_vector)
    adjoint = bindings.adjoint_jacobian()
    forward_calls = [
        (
            getattr(state_vector, operation.name),
            list(operation.wires),
            list(operation.parameters),
        )
        for operation in operations
    ]

    def forward() -> None:
        state_vector.resetStateVector()
        for apply_operation, wires, operation_params in forward_calls:
            apply_operation(wires, False, operation_params)

    def expectation() -> float:
        return float(measurements.expval(hamiltonian))

    def gradient() -> np.ndarray:
        raw = np.asarray(
            adjoint(state_vector, [hamiltonian], ops_data, trainable_params),
            dtype=precision_spec.real_dtype,
        ).reshape(-1)
        if raw.shape != parameter_sources.shape:
            raise RuntimeError(
                f"native adjoint returned shape {raw.shape}; expected {parameter_sources.shape}"
            )
        reduced = np.zeros(parameter_count, dtype=precision_spec.real_dtype)
        np.add.at(reduced, parameter_sources, raw)
        return reduced

    def run_step() -> tuple[float, np.ndarray, float, float, float]:
        forward_time, _ = _measure_phase(forward)
        hamiltonian_time, energy_value = _measure_phase(expectation)
        backward_time, gradient_value = _measure_phase(gradient)
        return (
            float(energy_value),
            np.asarray(gradient_value),
            forward_time,
            hamiltonian_time,
            backward_time,
        )

    energy_value: float | None = None
    gradient_value: np.ndarray | None = None
    for _ in range(warmup_steps):
        energy_value, gradient_value, _, _, _ = run_step()

    memory_after_warmup = take_memory_snapshot()
    forward_times: list[float] = []
    hamiltonian_times: list[float] = []
    backward_times: list[float] = []
    memory_samples: list[MemorySnapshot] = []
    for _ in range(steps):
        energy_value, gradient_value, forward_time, hamiltonian_time, backward_time = (
            run_step()
        )
        forward_times.append(forward_time)
        hamiltonian_times.append(hamiltonian_time)
        backward_times.append(backward_time)
        memory_samples.append(take_memory_snapshot())

    if energy_value is None or gradient_value is None:
        raise RuntimeError("native Lightning runner returned no energy/gradient result")
    if gradient_value.shape != (parameter_count,):
        raise RuntimeError(
            f"unexpected native gradient shape {gradient_value.shape}; "
            f"expected {(parameter_count,)}"
        )
    if not np.isfinite(energy_value) or not np.all(np.isfinite(gradient_value)):
        raise FloatingPointError("energy or gradient contains a non-finite value")

    step_times = tuple(
        forward + hamiltonian + backward
        for forward, hamiltonian, backward in zip(
            forward_times, hamiltonian_times, backward_times, strict=True
        )
    )
    return NativeEnergyGradResult(
        energy=energy_value,
        grad=gradient_value.copy(),
        step_times_s=step_times,
        forward_times_s=tuple(forward_times),
        hamiltonian_times_s=tuple(hamiltonian_times),
        backward_times_s=tuple(backward_times),
        memory=_build_memory_usage(
            memory_before, memory_after_warmup, memory_samples
        ),
        circuit=circuit_spec.name,
        qubits=qubits,
        layers=layers,
        parameter_count=parameter_count,
        precision=precision_spec.name,
        random_seed=random_seed,
        batches=batches,
        warmup_steps=warmup_steps,
        device_name=device_name,
        execution_scope="prebuilt-lightning-gpu-ops-synchronized-phase-sum",
    )


__all__ = [
    "NativeEnergyGradResult",
    "NativeOperation",
    "native_energy_and_grad",
]
