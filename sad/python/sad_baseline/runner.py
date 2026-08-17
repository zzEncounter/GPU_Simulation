"""Python-facing API for the custom CUDA/C++ implementation."""

from __future__ import annotations

import ctypes
import math
import os
import resource
import statistics
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_SAD_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LIBRARY = _SAD_ROOT / "build" / "libsad_cuda.so"
_VARIANT_FLAGS = {
    "f32r2_b32r2": (
        "-DSAD_FORWARD_BLOCK_THREADS=32",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=2",
    ),
    "f64r2_b64r2": (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=2",
    ),
    "f64r4_b64r4": (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
    "f64r3_b64r4": (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
    "f128r3_b64r4": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
    "f64r4_b128r3": (
        "-DSAD_FORWARD_BLOCK_THREADS=64",
        "-DSAD_FORWARD_REGISTER_BITS=4",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=3",
    ),
    "f128r3_b32r3": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=3",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=3",
    ),
    # Direction-independent finalists.  These keep the conservative forward
    # geometry when the larger/smaller joint variant only helps backward.
    "f128r2_b64r4": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=64",
        "-DSAD_REGISTER_BITS=4",
    ),
    "f128r2_b128r3": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=3",
    ),
    "f128r2_b32r2": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=2",
    ),
    # The ordinary RZZ diagonal lookup has two narrow, E2E-confirmed
    # exceptions to the otherwise shared k8 policy.
    "f32r2_b32r2_d6": (
        "-DSAD_FORWARD_BLOCK_THREADS=32",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=2",
        "-DSAD_DIAGONAL_LOOKUP_BITS=6",
    ),
    "f128r2_b128r2_d10": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=2",
        "-DSAD_DIAGONAL_LOOKUP_BITS=10",
    ),
    "f128r2_b128r2_d6": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=2",
        "-DSAD_DIAGONAL_LOOKUP_BITS=6",
    ),
    "f32r2_b32r2_xsep": (
        "-DSAD_FORWARD_BLOCK_THREADS=32",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=32",
        "-DSAD_REGISTER_BITS=2",
        "-DSAD_XXZ_CROSS_MATCHING=0",
    ),
    "f128r2_b128r2_xsep": (
        "-DSAD_FORWARD_BLOCK_THREADS=128",
        "-DSAD_FORWARD_REGISTER_BITS=2",
        "-DSAD_BLOCK_THREADS=128",
        "-DSAD_REGISTER_BITS=2",
        "-DSAD_XXZ_CROSS_MATCHING=0",
    ),
}
_VARIANT_LIBRARIES = {
    name: _SAD_ROOT / "build" / f"libsad_{name}.so"
    for name in _VARIANT_FLAGS
}
_MIB = 1024 * 1024


def _load_dotenv() -> None:
    dotenv = _SAD_ROOT / ".env"
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


class _NativeMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("state_vector_bytes", ctypes.c_uint64),
        ("total_workspace_bytes", ctypes.c_uint64),
        ("device_free_before_bytes", ctypes.c_uint64),
        ("device_free_after_alloc_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class MemoryUsage:
    """Static CUDA allocations and host RSS, all reported in MiB."""

    gpu_before_device_mib: float | None
    gpu_after_warmup_mib: float | None
    gpu_peak_observed_mib: float | None
    gpu_delta_observed_mib: float | None
    host_rss_before_mib: float
    host_rss_after_mib: float
    host_peak_rss_mib: float
    state_vector_mib: float
    total_workspace_mib: float
    device_total_mib: float


@dataclass(frozen=True)
class EnergyGradResult:
    energy: float
    grad: np.ndarray
    step_times_s: tuple[float, ...]
    memory: MemoryUsage
    forward_times_s: tuple[float, ...]
    hamiltonian_times_s: tuple[float, ...]
    backward_times_s: tuple[float, ...]
    circuit: str
    qubits: int
    layers: int
    parameter_count: int
    precision: str
    random_seed: int
    batches: int
    warmup_steps: int
    device_name: str
    execution_mode: str
    kernel_variant: str
    forward_phase_plan: str
    backward_phase_plan: str

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


_CIRCUITS = {
    "ra-hea": (0, 1),
    "ra": (0, 1),
    "su2-hea": (1, 2),
    "su2": (1, 2),
    "rzz-hea": (2, 3),
    "rzz": (2, 3),
    "qaoa": (3, 2),
    "qaoa-ns": (8, 0),
    "xxz-hva": (4, 3),
    "xxz": (4, 3),
    "mera": (5, 0),
    "equivariant-qnn": (6, 0),
    "eqnn": (6, 0),
    "data-reuploading": (7, 3),
    "data-reupload": (7, 3),
    "drqnn": (7, 3),
}

_PRECISIONS = {
    "float32": (0, np.dtype(np.float32)),
    "fp32": (0, np.dtype(np.float32)),
    "single": (0, np.dtype(np.float32)),
    "complex64": (0, np.dtype(np.float32)),
    "float64": (1, np.dtype(np.float64)),
    "fp64": (1, np.dtype(np.float64)),
    "double": (1, np.dtype(np.float64)),
    "complex128": (1, np.dtype(np.float64)),
}


def _normalise_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _truthy_environment(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _build_library(library_path: Path) -> None:
    variant = next(
        (name for name, path in _VARIANT_LIBRARIES.items()
         if path == library_path),
        None,
    )
    if library_path != _DEFAULT_LIBRARY and variant is None:
        raise RuntimeError(
            f"SAD_LIBRARY_PATH points to missing file {library_path}; custom paths "
            "cannot be auto-built"
        )
    nvcc = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
    cuda_arch = os.environ.get("SAD_CUDA_ARCH", "native")
    command = [
        "make",
        "-C",
        str(_SAD_ROOT),
        f"NVCC={nvcc}",
        f"CUDA_ARCH={cuda_arch}",
    ]
    if variant is not None:
        command.extend(
            (
                f"TARGET=build/libsad_{variant}.so",
                f"EXTRA_NVCCFLAGS={' '.join(_VARIANT_FLAGS[variant])}",
            )
        )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to build SAD CUDA library:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )


def _library_is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    source_mtime = max(
        file.stat().st_mtime
        for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h")
        for file in _SAD_ROOT.glob(pattern)
    )
    return path.stat().st_mtime < source_mtime


def _select_library(
    circuit_id: int, qubits: int, execution_mode: str
) -> tuple[str, Path]:
    explicit = os.environ.get("SAD_LIBRARY_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        return f"custom:{path.name}", path
    if execution_mode != "optimized":
        return "f128r2_b128r2", _DEFAULT_LIBRARY
    if _truthy_environment("SAD_DISABLE_VARIANT_DISPATCH", False):
        # Keep structural/diagonal policy identical in the conservative A/B;
        # only the execution geometry and phase plan are held conservative.
        if circuit_id == 2 and qubits == 10:
            return (
                "f128r2_b128r2_d6",
                _VARIANT_LIBRARIES["f128r2_b128r2_d6"],
            )
        if circuit_id == 2 and qubits == 20:
            return (
                "f128r2_b128r2_d10",
                _VARIANT_LIBRARIES["f128r2_b128r2_d10"],
            )
        if circuit_id == 4 and qubits == 6:
            return (
                "f128r2_b128r2_xsep",
                _VARIANT_LIBRARIES["f128r2_b128r2_xsep"],
            )
        return "f128r2_b128r2", _DEFAULT_LIBRARY
    if circuit_id == 2 and qubits == 10:
        return "f32r2_b32r2_d6", _VARIANT_LIBRARIES["f32r2_b32r2_d6"]
    if circuit_id == 2 and qubits == 20:
        return "f128r2_b128r2_d10", _VARIANT_LIBRARIES["f128r2_b128r2_d10"]
    if circuit_id == 4 and qubits == 6:
        return "f32r2_b32r2_xsep", _VARIANT_LIBRARIES["f32r2_b32r2_xsep"]
    # RX/RY geometry is selected per direction.  The direction-only layer
    # benchmark found no forward win for the joint shapes in these cases, so
    # retain F128r2 while keeping the independently faster backward shape.
    if (circuit_id, qubits) in {(0, 22), (1, 22), (1, 24), (2, 24)}:
        return "f128r2_b64r4", _VARIANT_LIBRARIES["f128r2_b64r4"]
    if (circuit_id, qubits) in {(2, 26), (3, 26)}:
        return "f128r2_b128r3", _VARIANT_LIBRARIES["f128r2_b128r3"]
    if (circuit_id, qubits) == (3, 18):
        return "f128r2_b32r2", _VARIANT_LIBRARIES["f128r2_b32r2"]
    if qubits in (4, 6, 10, 12, 14):
        return "f32r2_b32r2", _VARIANT_LIBRARIES["f32r2_b32r2"]
    if qubits in (8, 16):
        return "f64r2_b64r2", _VARIANT_LIBRARIES["f64r2_b64r2"]
    if qubits == 18 and circuit_id in (3, 4):
        return "f32r2_b32r2", _VARIANT_LIBRARIES["f32r2_b32r2"]
    if circuit_id == 0:
        if qubits >= 28:
            return "f64r3_b64r4", _VARIANT_LIBRARIES["f64r3_b64r4"]
        if qubits >= 20:
            return "f64r4_b64r4", _VARIANT_LIBRARIES["f64r4_b64r4"]
    elif circuit_id == 1 and qubits >= 22:
        return "f128r3_b64r4", _VARIANT_LIBRARIES["f128r3_b64r4"]
    elif circuit_id == 2:
        if qubits >= 26:
            return "f64r4_b128r3", _VARIANT_LIBRARIES["f64r4_b128r3"]
        if qubits == 24:
            return "f64r3_b64r4", _VARIANT_LIBRARIES["f64r3_b64r4"]
    elif circuit_id == 3 and qubits >= 26:
        return "f64r4_b128r3", _VARIANT_LIBRARIES["f64r4_b128r3"]
    elif circuit_id == 4 and qubits >= 22:
        return "f128r3_b32r3", _VARIANT_LIBRARIES["f128r3_b32r3"]
    return "f128r2_b128r2", _DEFAULT_LIBRARY


def _select_phase_plans(
    circuit_id: int, qubits: int, execution_mode: str
) -> tuple[str, str]:
    """Return E2E-confirmed runtime phase maps for the selected shape."""

    if execution_mode != "optimized" or _truthy_environment(
        "SAD_DISABLE_VARIANT_DISPATCH", False
    ):
        return "", ""
    if qubits == 10:
        if circuit_id == 0:
            return "", "compact:L2R2W0-L4R2W0"
        if circuit_id == 1:
            return (
                "compact:L4R0W0-L5R1W0",
                "compact:L2R2W0-L4R2W0",
            )
        if circuit_id in (2, 3):
            return (
                "compact:L2R2W0-L4R2W0",
                "compact:L4R2W0-L2R2W0",
            )
    if qubits == 12 and circuit_id == 0:
        return (
            "compact:L5R1W0-L5R1W0",
            "compact:L4R2W0-L4R2W0",
        )
    return "", ""


@lru_cache(maxsize=8)
def _native_library(library_path: str) -> ctypes.CDLL:
    path = Path(library_path)
    if _library_is_stale(path):
        if not _truthy_environment("SAD_AUTO_BUILD", True):
            raise RuntimeError(f"SAD CUDA library not found at {path}")
        _build_library(path)
    library = ctypes.CDLL(str(path))
    library.sad_energy_and_grad.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(_NativeMemoryInfo),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sad_energy_and_grad.restype = ctypes.c_int
    library.sad_version.argtypes = []
    library.sad_version.restype = ctypes.c_char_p
    return library


def _host_rss_mib() -> float:
    try:
        with Path("/proc/self/statm").open(encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / _MIB
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _host_peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / _MIB if value > 10 * _MIB else value / 1024.0


def energy_and_grad(
    circuit: str = "su2-hea",
    random_seed: int = 42,
    scalability: tuple[int, int] = (16, 8),
    batches: int = 1,
    precision: str = "float64",
    steps: int = 5,
    *,
    warmup_steps: int = 1,
    device_name: str = "sad.cuda",
    forward_phase_plan: str | None = None,
    backward_phase_plan: str | None = None,
) -> EnergyGradResult:
    """Run the custom CUDA forward/Hamiltonian/adjoint implementation."""

    if batches != 1:
        raise ValueError(
            f"batches is reserved for future use and must be 1, got {batches!r}"
        )
    if device_name != "sad.cuda":
        raise ValueError(f"device_name must be 'sad.cuda', got {device_name!r}")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an integer >= 1")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be an integer >= 0")
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
    if qubits < 2 or qubits > 30 or layers < 1:
        raise ValueError("qubits must be in [2, 30] and layers must be >= 1")

    if not isinstance(circuit, str):
        raise TypeError("circuit must be a registered circuit name")
    circuit_key = _normalise_name(circuit)
    try:
        circuit_id, parameters_per_qubit_layer = _CIRCUITS[circuit_key]
    except KeyError as exc:
        raise ValueError(f"unsupported circuit {circuit!r}") from exc
    if circuit_id in (2, 3, 4, 7, 8) and qubits % 2:
        raise ValueError(f"{circuit_key} requires an even number of qubits")
    if circuit_id in (3, 4, 7, 8) and qubits < 4:
        raise ValueError(f"{circuit_key} requires at least four qubits")
    if circuit_id == 5 and layers != (qubits - 1).bit_length():
        raise ValueError("MERA layers must equal ceil(log2(qubits))")

    if not isinstance(precision, str):
        raise TypeError("precision must be a string")
    try:
        precision_id, real_dtype = _PRECISIONS[precision.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported precision {precision!r}") from exc

    if circuit_id == 3:
        parameter_count = 2 * layers
    elif circuit_id == 8:
        parameter_count = 2 * qubits * layers
    elif circuit_id == 5:
        parameter_count = 4 * (qubits - 1) - 2 * (qubits - 1).bit_count()
    elif circuit_id == 6:
        parameter_count = 3 * layers
    else:
        parameter_count = parameters_per_qubit_layer * qubits * layers
    rng = np.random.default_rng(random_seed)
    params = np.ascontiguousarray(
        rng.uniform(-math.pi, math.pi, parameter_count), dtype=real_dtype
    )
    if circuit_id == 7:
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits):
                feature = 2.0 * (wire + 1) / (qubits + 1) - 1.0
                params[base + wire] += feature
                params[base + qubits + wire] += 0.5 * feature
                params[base + 2 * qubits + wire] -= 0.5 * feature
    gradient = np.empty(parameter_count, dtype=real_dtype)
    forward_times = np.empty(steps, dtype=np.float64)
    hamiltonian_times = np.empty(steps, dtype=np.float64)
    backward_times = np.empty(steps, dtype=np.float64)
    native_memory = _NativeMemoryInfo()
    native_energy = ctypes.c_double()
    error_buffer = ctypes.create_string_buffer(4096)
    device = int(os.environ.get("SAD_DEVICE", "0"))
    execution_mode = os.environ.get("SAD_EXECUTION_MODE", "optimized").strip().lower()
    kernel_variant, library_path = _select_library(
        circuit_id, qubits, execution_mode
    )
    automatic_phase_plans = (
        ("", "")
        if kernel_variant.startswith("custom:")
        else _select_phase_plans(circuit_id, qubits, execution_mode)
    )
    phase_plans: list[str] = []
    for index, (name, value) in enumerate((
        ("forward_phase_plan", forward_phase_plan),
        ("backward_phase_plan", backward_phase_plan),
    )):
        if value is None:
            value = automatic_phase_plans[index]
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None")
        phase_plans.append(value.strip())
    selected_forward_plan, selected_backward_plan = phase_plans
    host_before = _host_rss_mib()

    status = _native_library(str(library_path)).sad_energy_and_grad(
        precision_id,
        circuit_id,
        qubits,
        layers,
        steps,
        warmup_steps,
        device,
        params.ctypes.data_as(ctypes.c_void_p),
        parameter_count,
        selected_forward_plan.encode(),
        selected_backward_plan.encode(),
        ctypes.byref(native_energy),
        gradient.ctypes.data_as(ctypes.c_void_p),
        forward_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hamiltonian_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        backward_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(native_memory),
        error_buffer,
        ctypes.sizeof(error_buffer),
    )
    if status != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"SAD CUDA error ({status}): {message}")

    host_after = _host_rss_mib()
    allocated_mib = (
        max(
            0,
            native_memory.device_free_before_bytes
            - native_memory.device_free_after_alloc_bytes,
        )
        / _MIB
    )
    workspace_mib = native_memory.total_workspace_bytes / _MIB
    memory = MemoryUsage(
        gpu_before_device_mib=None,
        gpu_after_warmup_mib=allocated_mib,
        gpu_peak_observed_mib=allocated_mib,
        gpu_delta_observed_mib=allocated_mib,
        host_rss_before_mib=host_before,
        host_rss_after_mib=host_after,
        host_peak_rss_mib=max(host_before, host_after, _host_peak_rss_mib()),
        state_vector_mib=native_memory.state_vector_bytes / _MIB,
        total_workspace_mib=workspace_mib,
        device_total_mib=native_memory.device_total_bytes / _MIB,
    )
    step_times = forward_times + hamiltonian_times + backward_times
    canonical_name = (
        "ra-hea",
        "su2-hea",
        "rzz-hea",
        "qaoa",
        "xxz-hva",
        "mera",
        "equivariant-qnn",
        "data-reuploading",
        "qaoa-ns",
    )[circuit_id]
    return EnergyGradResult(
        energy=native_energy.value,
        grad=gradient,
        step_times_s=tuple(float(value) for value in step_times),
        memory=memory,
        forward_times_s=tuple(float(value) for value in forward_times),
        hamiltonian_times_s=tuple(float(value) for value in hamiltonian_times),
        backward_times_s=tuple(float(value) for value in backward_times),
        circuit=canonical_name,
        qubits=qubits,
        layers=layers,
        parameter_count=parameter_count,
        precision=real_dtype.name,
        random_seed=random_seed,
        batches=batches,
        warmup_steps=warmup_steps,
        device_name=device_name,
        execution_mode=execution_mode,
        kernel_variant=kernel_variant,
        forward_phase_plan=selected_forward_plan,
        backward_phase_plan=selected_backward_plan,
    )
