"""Public energy-and-adjoint-gradient benchmark API."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .circuits import CircuitSpec, build_hamiltonian, get_circuit
from .memory import MemorySnapshot, take_memory_snapshot


@dataclass(frozen=True)
class Precision:
    name: str
    real_dtype: type[np.floating]
    complex_dtype: type[np.complexfloating]


_PRECISIONS = {
    "float32": Precision("float32", np.float32, np.complex64),
    "fp32": Precision("float32", np.float32, np.complex64),
    "single": Precision("float32", np.float32, np.complex64),
    "complex64": Precision("float32", np.float32, np.complex64),
    "float64": Precision("float64", np.float64, np.complex128),
    "fp64": Precision("float64", np.float64, np.complex128),
    "double": Precision("float64", np.float64, np.complex128),
    "complex128": Precision("float64", np.float64, np.complex128),
}


@dataclass(frozen=True)
class MemoryUsage:
    """Boundary-sampled memory information in MiB."""

    gpu_before_device_mib: float | None
    gpu_after_warmup_mib: float | None
    gpu_peak_observed_mib: float | None
    gpu_delta_observed_mib: float | None
    host_rss_before_mib: float
    host_rss_after_mib: float
    host_peak_rss_mib: float


@dataclass(frozen=True)
class EnergyGradResult:
    """Result of repeated energy + adjoint-gradient evaluations."""

    energy: float
    grad: np.ndarray
    step_times_s: tuple[float, ...]
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

    def __iter__(self) -> Iterator[object]:
        """Allow ``energy, grad, times, memory = result``."""

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


def _resolve_precision(value: str) -> Precision:
    if not isinstance(value, str):
        raise TypeError("precision must be a string")
    try:
        return _PRECISIONS[value.strip().lower()]
    except KeyError as exc:
        choices = "float32/fp32/single/complex64 or float64/fp64/double/complex128"
        raise ValueError(f"unsupported precision {value!r}; use {choices}") from exc


def _validate_positive_integer(
    value: int, name: str, *, allow_zero: bool = False
) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"{name} must be an integer {comparator}, got {value!r}")


def _build_memory_usage(
    before: MemorySnapshot,
    after_warmup: MemorySnapshot,
    samples: list[MemorySnapshot],
) -> MemoryUsage:
    gpu_values = [
        sample.gpu_process_used_mib
        for sample in (before, after_warmup, *samples)
        if sample.gpu_process_used_mib is not None
    ]
    gpu_peak = max(gpu_values) if gpu_values else None
    if gpu_peak is None or before.gpu_process_used_mib is None:
        gpu_delta = None
    else:
        gpu_delta = gpu_peak - before.gpu_process_used_mib

    final = samples[-1] if samples else after_warmup
    return MemoryUsage(
        gpu_before_device_mib=before.gpu_process_used_mib,
        gpu_after_warmup_mib=after_warmup.gpu_process_used_mib,
        gpu_peak_observed_mib=gpu_peak,
        gpu_delta_observed_mib=gpu_delta,
        host_rss_before_mib=before.host_rss_mib,
        host_rss_after_mib=final.host_rss_mib,
        host_peak_rss_mib=max(
            sample.host_peak_rss_mib for sample in (before, after_warmup, *samples)
        ),
    )


def energy_and_grad(
    circuit: str | CircuitSpec = "su2-hea",
    random_seed: int = 42,
    scalability: tuple[int, int] = (16, 8),
    batches: int = 1,
    precision: str = "float64",
    steps: int = 5,
    *,
    warmup_steps: int = 1,
    device_name: str = "lightning.gpu",
) -> EnergyGradResult:
    """Evaluate a fixed HEA energy and its full gradient with adjoint diff.

    ``steps`` counts measured repetitions. Warmups are additional calls and are not
    included in ``step_times_s``. Parameters are sampled once from ``U(-pi, pi)``
    and are intentionally not updated between repetitions.
    """

    if batches != 1:
        raise ValueError(
            f"batches is reserved for future use and must be 1, got {batches!r}"
        )
    _validate_positive_integer(steps, "steps")
    _validate_positive_integer(warmup_steps, "warmup_steps", allow_zero=True)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    if (
        not isinstance(scalability, tuple)
        or len(scalability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in scalability
        )
    ):
        raise ValueError("scalability must be a (qubits, layers) integer tuple")

    qubits, layers = scalability
    circuit_spec = get_circuit(circuit)
    parameter_count = circuit_spec.parameter_count(qubits, layers)
    precision_spec = _resolve_precision(precision)

    rng = np.random.default_rng(random_seed)
    raw_params = rng.uniform(-math.pi, math.pi, parameter_count).astype(
        precision_spec.real_dtype, copy=False
    )
    params = pnp.array(raw_params, requires_grad=True)

    memory_before = take_memory_snapshot()
    try:
        device = qml.device(
            device_name,
            wires=qubits,
            shots=None,
            c_dtype=precision_spec.complex_dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not create PennyLane device {device_name!r}; install a compatible "
            "pennylane-lightning-gpu/cuQuantum stack and verify CUDA availability"
        ) from exc

    hamiltonian = build_hamiltonian(qubits)

    @qml.qnode(device, interface="autograd", diff_method="adjoint")
    def qnode(values: object) -> object:
        # The omitted state-preparation operation is exactly |0>**qubits.
        circuit_spec.apply(values, qubits, layers)
        return qml.expval(hamiltonian)

    gradient_fn = qml.grad(qnode)
    energy_value: object | None = None
    gradient_value: object | None = None

    for _ in range(warmup_steps):
        gradient_value = gradient_fn(params)
        energy_value = gradient_fn.forward

    memory_after_warmup = take_memory_snapshot()
    step_times: list[float] = []
    memory_samples: list[MemorySnapshot] = []
    for _ in range(steps):
        started = time.perf_counter()
        gradient_value = gradient_fn(params)
        energy_value = gradient_fn.forward
        step_times.append(time.perf_counter() - started)
        # Memory instrumentation is deliberately outside the timed region.
        memory_samples.append(take_memory_snapshot())

    if energy_value is None or gradient_value is None:
        raise RuntimeError("PennyLane returned no energy/gradient result")

    grad_array = np.asarray(gradient_value, dtype=precision_spec.real_dtype)
    if grad_array.shape != (parameter_count,):
        raise RuntimeError(
            f"unexpected gradient shape {grad_array.shape}; expected {(parameter_count,)}"
        )
    energy_float = float(np.asarray(energy_value))
    if not np.isfinite(energy_float) or not np.all(np.isfinite(grad_array)):
        raise FloatingPointError("energy or gradient contains a non-finite value")

    return EnergyGradResult(
        energy=energy_float,
        grad=grad_array.copy(),
        step_times_s=tuple(step_times),
        memory=_build_memory_usage(memory_before, memory_after_warmup, memory_samples),
        circuit=circuit_spec.name,
        qubits=qubits,
        layers=layers,
        parameter_count=parameter_count,
        precision=precision_spec.name,
        random_seed=random_seed,
        batches=batches,
        warmup_steps=warmup_steps,
        device_name=device_name,
    )
