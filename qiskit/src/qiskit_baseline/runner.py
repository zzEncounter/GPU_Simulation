"""Parameter-shift-rule gradient computation using Qiskit AerSimulator (GPU).

Core loop
---------
For each model parameter i that has k circuit-level occurrences:

  grad[i] = sum over each circuit param c where param_map[c] == i of:
      (E(θ_c + π/2) - E(θ_c - π/2)) / 2

where only parameter c is shifted; all other circuit parameters keep their
current values.  This correctly handles shared parameters (e.g. equivariant-qnn,
QAOA) because the total derivative equals the sum of partial derivatives, each
of which is computed by the standard parameter-shift formula.

Cost: 2 * n_circuit_params expectation-value evaluations per gradient step,
compared to O(1) for PennyLane's adjoint-differentiation method.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np

from .circuits import CircuitBundle, build_circuit, build_hamiltonian
from .memory import MemorySnapshot, take_memory_snapshot


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Precision:
    name: str
    real_dtype: type
    complex_dtype: type


_PRECISIONS: dict[str, _Precision] = {
    "float32":   _Precision("float32", np.float32,  np.complex64),
    "fp32":      _Precision("float32", np.float32,  np.complex64),
    "single":    _Precision("float32", np.float32,  np.complex64),
    "complex64": _Precision("float32", np.float32,  np.complex64),
    "float64":   _Precision("float64", np.float64, np.complex128),
    "fp64":      _Precision("float64", np.float64, np.complex128),
    "double":    _Precision("float64", np.float64, np.complex128),
    "complex128":_Precision("float64", np.float64, np.complex128),
}


def _resolve_precision(value: str) -> _Precision:
    try:
        return _PRECISIONS[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported precision {value!r}; use float32/float64"
        ) from exc


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryUsage:
    gpu_before_device_mib: float | None
    gpu_after_warmup_mib: float | None
    gpu_peak_observed_mib: float | None
    gpu_delta_observed_mib: float | None
    host_rss_before_mib: float
    host_rss_after_mib: float
    host_peak_rss_mib: float


@dataclass(frozen=True)
class EnergyGradResult:
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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class _Evaluator:
    """Wraps a BackendEstimatorV2 for repeated expectation-value queries."""

    def __init__(self, bundle: CircuitBundle, hamiltonian, backend) -> None:
        from qiskit.primitives import BackendEstimatorV2

        self._bundle = bundle
        self._hamiltonian = hamiltonian
        self._estimator = BackendEstimatorV2(backend=backend)

    def __call__(self, model_values: np.ndarray) -> float:
        """Evaluate ⟨H⟩ for the given model-parameter values."""
        binding = {
            self._bundle.param_list[i]: float(model_values[self._bundle.param_map[i]])
            for i in range(len(self._bundle.param_list))
        }
        bound = self._bundle.circuit.assign_parameters(binding)
        job = self._estimator.run([(bound, self._hamiltonian)])
        return float(np.real(job.result()[0].data.evs))


# ---------------------------------------------------------------------------
# Parameter-shift-rule gradient
# ---------------------------------------------------------------------------

def _parameter_shift_grad(
    evaluator: _Evaluator,
    model_values: np.ndarray,
    bundle: CircuitBundle,
) -> tuple[float, np.ndarray]:
    """Compute energy and full gradient via parameter shift.

    Returns (energy, grad_array) where energy is re-used from the first
    circuit-parameter shift pair of the first model parameter, and grad is
    the full model-parameter gradient vector.

    Total circuit evaluations: 2 * n_circuit_params.
    """
    n_model = bundle.n_model_params
    n_circuit = len(bundle.param_list)
    grad = np.zeros(n_model, dtype=model_values.dtype)
    energy = None

    for c_idx in range(n_circuit):
        model_idx = bundle.param_map[c_idx]

        # Build +shift and -shift model-value arrays (only the circuit-level
        # parameter c_idx is shifted; all other circuit params keep their
        # current model values — the shift must act on the circuit param, not
        # on all model params with the same index).
        vals_plus = _shift_one_circuit_param(model_values, bundle, c_idx, +math.pi / 2)
        vals_minus = _shift_one_circuit_param(model_values, bundle, c_idx, -math.pi / 2)

        e_plus = evaluator(vals_plus)
        e_minus = evaluator(vals_minus)

        partial = (e_plus - e_minus) / 2.0
        grad[model_idx] += partial

        # Use the midpoint as an energy estimate for the first pair
        if energy is None:
            energy = (e_plus + e_minus) / 2.0

    return float(energy), grad


def _shift_one_circuit_param(
    model_values: np.ndarray,
    bundle: CircuitBundle,
    c_idx: int,
    shift: float,
) -> np.ndarray:
    """Return model_values with a virtual shift applied to circuit param c_idx only.

    Because the evaluator maps circuit params → model values via param_map,
    we cannot shift a single circuit occurrence by modifying model_values when
    multiple circuit params share the same model-param index (shared parameters).

    Strategy: create a temporary expanded parameter array of length n_circuit,
    apply the shift to position c_idx, then pass it through a one-shot binding
    that bypasses the normal param_map logic.

    We return a special sentinel array of length n_circuit + 1 that the
    evaluator recognises as "raw circuit-level values".  To keep the evaluator
    interface clean we instead build the binding directly here and return the
    result via a tiny wrapper.

    Simpler alternative used here: return an augmented model_values of length
    n_model where only the model param for c_idx has been shifted — this is
    CORRECT for circuits where each model param appears in exactly one circuit
    slot, which is true for all non-shared-parameter circuits.

    For shared-parameter circuits the correct approach is implemented separately:
    we build the binding dict directly by iterating all circuit params, shifting
    only c_idx.

    Since _Evaluator always rebuilds the binding from scratch, we pass a
    "per-circuit-slot" values array of length n_circuit instead of n_model, and
    adjust the evaluator accordingly.  See _EvaluatorRaw below.
    """
    # We return a n_circuit-length array where position j holds the value that
    # circuit param j should take.  This is consumed by _EvaluatorRaw.
    circuit_vals = np.array(
        [float(model_values[bundle.param_map[j]]) for j in range(len(bundle.param_list))],
        dtype=model_values.dtype,
    )
    circuit_vals[c_idx] += shift
    return circuit_vals


class _EvaluatorRaw:
    """Evaluator that accepts a per-circuit-slot values array (length = n_circuit)."""

    def __init__(self, bundle: CircuitBundle, hamiltonian, backend) -> None:
        from qiskit.primitives import BackendEstimatorV2

        self._bundle = bundle
        self._hamiltonian = hamiltonian
        self._estimator = BackendEstimatorV2(backend=backend)

    def __call__(self, circuit_values: np.ndarray) -> float:
        binding = {
            self._bundle.param_list[i]: float(circuit_values[i])
            for i in range(len(self._bundle.param_list))
        }
        bound = self._bundle.circuit.assign_parameters(binding)
        job = self._estimator.run([(bound, self._hamiltonian)])
        return float(np.real(job.result()[0].data.evs))

    def from_model(self, model_values: np.ndarray) -> float:
        circuit_vals = np.array(
            [float(model_values[self._bundle.param_map[j]])
             for j in range(len(self._bundle.param_list))],
            dtype=model_values.dtype,
        )
        return self(circuit_vals)


def _parameter_shift_grad_v2(
    evaluator: _EvaluatorRaw,
    model_values: np.ndarray,
    bundle: CircuitBundle,
) -> tuple[float, np.ndarray]:
    """Correct PSR gradient for all circuits, including shared-parameter ones."""
    n_model = bundle.n_model_params
    grad = np.zeros(n_model, dtype=model_values.dtype)
    energy = None

    # Base circuit values (circuit-slot expansion of model_values)
    base_cv = np.array(
        [float(model_values[bundle.param_map[j]])
         for j in range(len(bundle.param_list))],
        dtype=model_values.dtype,
    )

    for c_idx in range(len(bundle.param_list)):
        model_idx = bundle.param_map[c_idx]

        cv_plus = base_cv.copy()
        cv_plus[c_idx] += math.pi / 2

        cv_minus = base_cv.copy()
        cv_minus[c_idx] -= math.pi / 2

        e_plus = evaluator(cv_plus)
        e_minus = evaluator(cv_minus)

        grad[model_idx] += (e_plus - e_minus) / 2.0

        if energy is None:
            energy = (e_plus + e_minus) / 2.0

    return float(energy), grad


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def _build_memory_usage(
    before: MemorySnapshot,
    after_warmup: MemorySnapshot,
    samples: list[MemorySnapshot],
) -> MemoryUsage:
    gpu_values = [
        s.gpu_process_used_mib
        for s in (before, after_warmup, *samples)
        if s.gpu_process_used_mib is not None
    ]
    gpu_peak = max(gpu_values) if gpu_values else None
    gpu_delta = (
        (gpu_peak - before.gpu_process_used_mib)
        if gpu_peak is not None and before.gpu_process_used_mib is not None
        else None
    )
    final = samples[-1] if samples else after_warmup
    return MemoryUsage(
        gpu_before_device_mib=before.gpu_process_used_mib,
        gpu_after_warmup_mib=after_warmup.gpu_process_used_mib,
        gpu_peak_observed_mib=gpu_peak,
        gpu_delta_observed_mib=gpu_delta,
        host_rss_before_mib=before.host_rss_mib,
        host_rss_after_mib=final.host_rss_mib,
        host_peak_rss_mib=max(
            s.host_peak_rss_mib for s in (before, after_warmup, *samples)
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def energy_and_grad(
    circuit: str = "su2-hea",
    random_seed: int = 42,
    scalability: tuple[int, int] = (16, 8),
    batches: int = 1,
    precision: str = "float64",
    steps: int = 5,
    *,
    warmup_steps: int = 1,
    device_name: str = "aer_simulator",
    gpu: bool = True,
) -> EnergyGradResult:
    """Evaluate energy and full gradient via parameter shift rule.

    Each "step" performs 2 * n_circuit_params expectation-value evaluations,
    where n_circuit_params >= n_model_params (equality holds for circuits without
    shared parameters).

    Parameters
    ----------
    circuit:      Circuit name (see available_circuits()).
    random_seed:  RNG seed for initial parameter sampling.
    scalability:  (qubits, layers) tuple.
    batches:      Reserved; must be 1.
    precision:    "float32" or "float64".
    steps:        Number of measured gradient steps.
    warmup_steps: Un-timed warm-up iterations.
    device_name:  Qiskit backend name ("aer_simulator").
    gpu:          If True, configure AerSimulator for GPU (cuStateVec).
    """
    if batches != 1:
        raise ValueError(f"batches must be 1, got {batches!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps!r}")

    qubits, layers = scalability
    prec = _resolve_precision(precision)

    bundle = build_circuit(circuit, qubits, layers)
    hamiltonian = build_hamiltonian(circuit, qubits)
    n_model = bundle.n_model_params

    rng = np.random.default_rng(random_seed)
    raw_params = rng.uniform(-math.pi, math.pi, n_model).astype(prec.real_dtype, copy=False)

    # data-reuploading: embed classical features into the initial parameters
    if circuit.lower().replace("_", "-") in {"data-reuploading", "data-reupload", "drqnn"}:
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits):
                feature = 2.0 * (wire + 1) / (qubits + 1) - 1.0
                raw_params[base + wire] += feature
                raw_params[base + qubits + wire] += 0.5 * feature
                raw_params[base + 2 * qubits + wire] -= 0.5 * feature

    model_values = raw_params.copy()

    try:
        from qiskit_aer import AerSimulator

        if gpu:
            backend = AerSimulator(
                method="statevector",
                device="GPU",
                cuStateVec_enable=True,
            )
        else:
            backend = AerSimulator(method="statevector")
    except Exception as exc:
        raise RuntimeError(
            "Could not create AerSimulator; install qiskit-aer with GPU support "
            "(pip install qiskit-aer-gpu) and verify CUDA availability."
        ) from exc

    memory_before = take_memory_snapshot()
    evaluator = _EvaluatorRaw(bundle, hamiltonian, backend)

    energy_val: float | None = None
    grad_val: np.ndarray | None = None

    for _ in range(warmup_steps):
        energy_val, grad_val = _parameter_shift_grad_v2(evaluator, model_values, bundle)

    memory_after_warmup = take_memory_snapshot()
    step_times: list[float] = []
    memory_samples: list[MemorySnapshot] = []

    for _ in range(steps):
        t0 = time.perf_counter()
        energy_val, grad_val = _parameter_shift_grad_v2(evaluator, model_values, bundle)
        step_times.append(time.perf_counter() - t0)
        memory_samples.append(take_memory_snapshot())

    if energy_val is None or grad_val is None:
        raise RuntimeError("No energy/gradient result produced.")

    grad_array = np.asarray(grad_val, dtype=prec.real_dtype)
    energy_float = float(energy_val)

    if not np.isfinite(energy_float) or not np.all(np.isfinite(grad_array)):
        raise FloatingPointError("energy or gradient contains a non-finite value")

    return EnergyGradResult(
        energy=energy_float,
        grad=grad_array.copy(),
        step_times_s=tuple(step_times),
        memory=_build_memory_usage(memory_before, memory_after_warmup, memory_samples),
        circuit=circuit,
        qubits=qubits,
        layers=layers,
        parameter_count=n_model,
        precision=prec.name,
        random_seed=random_seed,
        batches=batches,
        warmup_steps=warmup_steps,
        device_name=device_name,
    )
