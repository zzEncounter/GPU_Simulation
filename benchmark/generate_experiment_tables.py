"""Generate the refreshed main experiment, execution A/B CSV, and main report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
DEFAULT_OPTIMIZED = RESULTS / "sad_optimized_gpu.csv"
DEFAULT_FIXED = RESULTS / "sad_fixed_parameters_gpu.csv"
DEFAULT_LIGHTNING = RESULTS / "lightning_gpu_native.csv"
DEFAULT_PAIRED = RESULTS / "parameter_policy_paired_raw.csv"
DEFAULT_DIRECTIONAL_SHAPES = RESULTS / "directional_rotation_shape_raw.csv"
DEFAULT_MAIN = RESULTS / "main_experiment.csv"
DEFAULT_PARAMETERS = RESULTS / "parameter_search_experiment.csv"
DEFAULT_REPORT = ROOT / "docs" / "experiments" / "主实验.md"
HARDWARE = "NVIDIA RTX 6000 Ada"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
QUBITS = tuple(range(4, 29, 2))
VARIANT = re.compile(r"f(\d+)r(\d+)_b(\d+)r(\d+)(?:_d\d+|_xsep)?$")

MAIN_FIELDS = (
    "experiment_group",
    "timestamp_utc",
    "hardware",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "measured_steps",
    "warmup_steps",
    "sad_kernel_variant",
    "sad_execution_parameters",
    "sad_structural_parameters",
    "sad_selected_parameters",
    "sad_median_ms",
    "sad_forward_mean_ms",
    "sad_hamiltonian_mean_ms",
    "sad_backward_mean_ms",
    "lightning_native_median_ms",
    "speedup_vs_lightning_native",
    "energy_abs_error_vs_lightning_native",
    "gradient_max_abs_error_vs_lightning_native",
    "source_sad",
    "source_lightning_native",
)

PARAMETER_FIELDS = (
    "experiment_group",
    "timestamp_utc",
    "hardware",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "default_policy",
    "default_execution_parameters",
    "default_parameters",
    "default_median_ms",
    "selected_policy",
    "selected_execution_parameters",
    "shared_structural_parameters",
    "selected_parameters",
    "selected_median_ms",
    "speedup_vs_default",
    "effect_estimator",
    "paired_samples",
    "paired_ratio_mad",
    "improvement_percent",
    "forward_speedup_vs_default",
    "forward_selection_speedup_vs_default",
    "forward_selection_estimator",
    "forward_selection_source",
    "hamiltonian_speedup_vs_default",
    "backward_speedup_vs_default",
    "backward_selection_speedup_vs_default",
    "backward_selection_estimator",
    "backward_selection_source",
    "selected_variant",
    "default_variant",
    "correct",
    "energy_abs_error",
    "gradient_max_abs_error",
    "source_selected",
    "source_default",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def directional_shape_index(
    path: Path | None,
) -> dict[tuple[str, int, str], float]:
    if path is None or not path.exists():
        return {}
    samples: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("candidate") != "joint" or row.get("correct") != "1":
            continue
        samples[(row["circuit"], int(row["qubits"]), row["direction"])].append(
            float(row["baseline_over_candidate"])
        )
    return {key: statistics.median(values) for key, values in samples.items()}


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["circuit"], int(row["qubits"]), int(row["layers"])


def index_complete(
    path: Path, *, label: str
) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, int, int], dict[str, str]] = {}
    failures: list[str] = []
    for row in rows:
        if row.get("status") != "ok":
            failures.append(f"{key(row)}: {row.get('error', 'unknown error')}")
            continue
        row_key = key(row)
        if row_key in result:
            raise ValueError(f"duplicate {label} row {row_key}")
        result[row_key] = row
    expected = {(circuit, qubits, 8) for circuit in CIRCUITS for qubits in QUBITS}
    missing = sorted(expected - set(result))
    extra = sorted(set(result) - expected)
    if failures or missing or extra:
        raise ValueError(
            f"incomplete {label}: failures={failures or 'none'}, "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    return result


def write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_name(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def paired_index(path: Path) -> dict[tuple[str, int, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[key(row)].append(row)
    return grouped


def gradient(row: dict[str, str]) -> np.ndarray:
    return np.asarray(json.loads(row["grad_json"]), dtype=np.float64)


def tile_bits(threads: int, register_bits: int) -> int:
    """Return address bits covered by one CTA tile."""
    if threads < 32 or threads % 32:
        raise ValueError(f"threads must be a positive warp multiple, got {threads}")
    warps = threads // 32
    if warps & (warps - 1):
        raise ValueError(f"warp count must be a power of two, got {warps}")
    return 5 + register_bits + int(math.log2(warps))


def phase_target_counts(
    qubits: int, bits_per_tile: int, *, fixed_low_5: bool
) -> tuple[int, ...]:
    """Describe how many new logical targets each phase introduces."""
    if qubits <= 0 or bits_per_tile < 5:
        raise ValueError((qubits, bits_per_tile))
    remaining = qubits
    counts: list[int] = []
    first_capacity = bits_per_tile
    later_capacity = bits_per_tile - 5 if fixed_low_5 else bits_per_tile
    if remaining > first_capacity and later_capacity <= 0:
        raise ValueError("fixed-low-5 needs more than five tile bits")
    while remaining:
        capacity = first_capacity if not counts else later_capacity
        count = min(remaining, capacity)
        counts.append(count)
        remaining -= count
    return tuple(counts)


def cross_matching_phase_count(qubits: int, bits_per_tile: int) -> int:
    """Return the production XXZ cross-matching phase count."""
    matching_targets = min(qubits, bits_per_tile) & ~1
    if matching_targets == 0:
        raise ValueError((qubits, bits_per_tile))
    return math.ceil(qubits / matching_targets) + 1


def strategy_descriptor(circuit: str, qubits: int) -> str:
    if circuit == "ra-hea":
        return "RA-real;RY+CNOT-fused;CNOT=Fscatter/Bgather;D=none"
    if circuit == "su2-hea":
        forward = "phased" if qubits == 10 else "RY+RZ+CNOT-lookup-fused"
        backward = (
            "phased" if qubits == 4
            else "RY+RZ+CNOT-lookup-fused" if qubits in (6, 16)
            else "split"
        )
        return (
            f"F={forward};B={backward};"
            "CNOT=Fscatter/Bgather;D=RZ-k8-t64"
        )
    if circuit == "rzz-hea":
        backward = (
            "RZ+RZZ-combined" if qubits <= 14
            else "RX+RZ+RZZ-fused" if qubits == 18
            else "split"
        )
        lookup = "k6" if qubits == 10 else "k10" if qubits == 20 else "k8"
        return (
            f"F=RX+RZ+RZZ-fused;B={backward};"
            f"D=RZ/RZZ-{lookup}-t64"
        )
    if circuit == "qaoa":
        return (
            "init+cost=combined;F=cost+RX-fused;B=cost+RX-fused;"
            "D=domain-wall-t128"
        )
    if circuit == "xxz-hva":
        matching = "separate-matching" if qubits == 6 else "cross-matching"
        return f"XX+YY+ZZ-fused;{matching};D=bond-fused"
    raise ValueError(circuit)


def execution_descriptor(
    circuit: str,
    qubits: int,
    variant: str,
    forward_phase_plan: str = "",
    backward_phase_plan: str = "",
) -> str:
    match = VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError(f"invalid SAD variant {variant!r}")
    ft, fr, bt, br = (int(value) for value in match.groups()[:4])
    forward_tile = tile_bits(ft, fr)
    backward_tile = tile_bits(bt, br)
    if circuit == "xxz-hva":
        if qubits == 6:
            return f"F:t{ft}r{fr}m1-Zsep/B:t{bt}r{br}m1-Zsep"
        forward = (
            f"F:t{ft}r{fr}m1-Zx"
            f"{cross_matching_phase_count(qubits, forward_tile)}"
        )
        backward = (
            f"B:t{bt}r{br}m1-Zx"
            f"{cross_matching_phase_count(qubits, backward_tile)}"
        )
        return f"{forward}/{backward}"
    backward_fixed = circuit in {"ra-hea", "su2-hea"} and qubits >= 18
    forward_counts = phase_target_counts(
        qubits, forward_tile, fixed_low_5=False
    )
    backward_counts = phase_target_counts(
        qubits, backward_tile, fixed_low_5=backward_fixed
    )
    forward_phases = "+".join(str(value) for value in forward_counts)
    backward_phases = "+".join(str(value) for value in backward_counts)
    backward_family = "X" if backward_fixed else "C"
    forward_layout = _phase_layout_descriptor(
        forward_phase_plan, "C", forward_phases
    )
    backward_layout = _phase_layout_descriptor(
        backward_phase_plan, backward_family, backward_phases
    )
    return f"F:t{ft}r{fr}m1-{forward_layout}/B:t{bt}r{br}m1-{backward_layout}"


def _phase_layout_descriptor(plan: str, family: str, canonical: str) -> str:
    if not plan:
        return f"{family}[{canonical}]"
    try:
        plan_family, schedule = plan.split(":", 1)
        code = {"compact": "C", "fixed": "X", "pairs": "P"}[plan_family]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"invalid recorded phase plan {plan!r}") from exc
    return f"{code}[{schedule}]"


def parameter_descriptor(
    circuit: str,
    qubits: int,
    variant: str,
    forward_phase_plan: str = "",
    backward_phase_plan: str = "",
) -> str:
    return (
        f"{execution_descriptor(circuit, qubits, variant, forward_phase_plan, backward_phase_plan)};"
        f"{strategy_descriptor(circuit, qubits)}"
    )


def build_tables(
    optimized_path: Path,
    fixed_path: Path,
    lightning_path: Path,
    paired_path: Path,
    directional_shapes_path: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    optimized = index_complete(optimized_path, label="optimized SAD")
    fixed = index_complete(fixed_path, label="fixed-parameter SAD")
    lightning = index_complete(lightning_path, label="Lightning native")
    paired = paired_index(paired_path)
    directional_shapes = directional_shape_index(directional_shapes_path)
    main_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for circuit in CIRCUITS:
        for qubits in QUBITS:
            row_key = (circuit, qubits, 8)
            selected = optimized[row_key]
            default = fixed[row_key]
            reference = lightning[row_key]
            default_match = VARIANT.fullmatch(default["kernel_variant"])
            selected_match = VARIANT.fullmatch(selected["kernel_variant"])
            if selected_match is None:
                raise ValueError(
                    f"unrecognized selected variant {selected['kernel_variant']}"
                )
            if default_match is None or tuple(
                int(value) for value in default_match.groups()[:4]
            ) != (128, 2, 128, 2):
                raise ValueError(
                    f"fixed policy dispatched {default['kernel_variant']} for {row_key}"
                )
            selected_grad = gradient(selected)
            default_grad = gradient(default)
            reference_grad = gradient(reference)
            if selected_grad.shape != default_grad.shape or selected_grad.shape != reference_grad.shape:
                raise ValueError(f"gradient shape mismatch for {row_key}")
            selected_ms = 1000 * float(selected["time_median_s"])
            default_ms = 1000 * float(default["time_median_s"])
            lightning_ms = 1000 * float(reference["time_median_s"])
            if min(selected_ms, default_ms, lightning_ms) <= 0 or not all(
                math.isfinite(value) for value in (selected_ms, default_ms, lightning_ms)
            ):
                raise ValueError(f"invalid timing for {row_key}")
            selected_execution = execution_descriptor(
                circuit,
                qubits,
                selected["kernel_variant"],
                selected.get("forward_phase_plan", ""),
                selected.get("backward_phase_plan", ""),
            )
            default_execution = execution_descriptor(
                circuit,
                qubits,
                default["kernel_variant"],
                default.get("forward_phase_plan", ""),
                default.get("backward_phase_plan", ""),
            )
            shared_structure = strategy_descriptor(circuit, qubits)
            selected_parameters = f"{selected_execution};{shared_structure}"
            default_parameters = f"{default_execution};{shared_structure}"
            selected_energy_error = abs(
                float(selected["energy"]) - float(reference["energy"])
            )
            selected_gradient_error = float(
                np.max(np.abs(selected_grad - reference_grad))
            )
            ab_energy_error = abs(float(selected["energy"]) - float(default["energy"]))
            ab_gradient_error = float(np.max(np.abs(selected_grad - default_grad)))
            main_rows.append(
                {
                    "experiment_group": "main_experiment",
                    "timestamp_utc": selected["timestamp_utc"],
                    "hardware": HARDWARE,
                    "circuit": circuit,
                    "qubits": qubits,
                    "layers": selected["layers"],
                    "precision": selected["precision"],
                    "measured_steps": selected["steps"],
                    "warmup_steps": selected["warmup_steps"],
                    "sad_kernel_variant": selected["kernel_variant"],
                    "sad_execution_parameters": selected_execution,
                    "sad_structural_parameters": shared_structure,
                    "sad_selected_parameters": selected_parameters,
                    "sad_median_ms": selected_ms,
                    "sad_forward_mean_ms": 1000 * float(selected["forward_mean_s"]),
                    "sad_hamiltonian_mean_ms": 1000 * float(selected["hamiltonian_mean_s"]),
                    "sad_backward_mean_ms": 1000 * float(selected["backward_mean_s"]),
                    "lightning_native_median_ms": lightning_ms,
                    "speedup_vs_lightning_native": lightning_ms / selected_ms,
                    "energy_abs_error_vs_lightning_native": selected_energy_error,
                    "gradient_max_abs_error_vs_lightning_native": selected_gradient_error,
                    "source_sad": source_name(optimized_path),
                    "source_lightning_native": source_name(lightning_path),
                }
            )
            same_execution_parameters = (
                selected["kernel_variant"] == default["kernel_variant"]
                and selected.get("forward_phase_plan", "")
                == default.get("forward_phase_plan", "")
                and selected.get("backward_phase_plan", "")
                == default.get("backward_phase_plan", "")
            )
            selected_geometry = tuple(int(value) for value in selected_match.groups()[:4])
            default_geometry = tuple(int(value) for value in default_match.groups()[:4])
            same_forward_parameters = (
                selected_geometry[:2] == default_geometry[:2]
                and selected.get("forward_phase_plan", "")
                == default.get("forward_phase_plan", "")
            )
            same_backward_parameters = (
                selected_geometry[2:] == default_geometry[2:]
                and selected.get("backward_phase_plan", "")
                == default.get("backward_phase_plan", "")
            )
            if same_execution_parameters:
                parameter_selected_ms = selected_ms
                parameter_default_ms = selected_ms
                speedup = 1.0
                forward_speedup = 1.0
                hamiltonian_speedup = 1.0
                backward_speedup = 1.0
                effect_estimator = "identical_configuration"
                paired_samples = 0
                paired_ratio_mad: object = ""
                ab_energy_error = 0.0
                ab_gradient_error = 0.0
                parameter_source = optimized_path
            else:
                samples = paired.get(row_key, [])
                expected_samples = 5 if qubits <= 24 else 3
                if len(samples) != expected_samples:
                    raise ValueError(
                        f"expected {expected_samples} paired samples for {row_key}, "
                        f"found {len(samples)}"
                    )
                if any(
                    row["selected_variant"] != selected["kernel_variant"]
                    or row["default_variant"] != default["kernel_variant"]
                    or row.get("selected_forward_phase_plan", "")
                    != selected.get("forward_phase_plan", "")
                    or row.get("default_forward_phase_plan", "")
                    != default.get("forward_phase_plan", "")
                    or row.get("selected_backward_phase_plan", "")
                    != selected.get("backward_phase_plan", "")
                    or row.get("default_backward_phase_plan", "")
                    != default.get("backward_phase_plan", "")
                    for row in samples
                ):
                    raise ValueError(f"paired variant mismatch for {row_key}")
                ratios = [float(row["default_over_selected"]) for row in samples]
                speedup = statistics.median(ratios)
                paired_ratio_mad = statistics.median(
                    abs(value - speedup) for value in ratios
                )
                parameter_selected_ms = statistics.median(
                    float(row["selected_median_ms"]) for row in samples
                )
                parameter_default_ms = statistics.median(
                    float(row["default_median_ms"]) for row in samples
                )
                forward_speedup = statistics.median(
                    float(row["default_over_selected_forward"]) for row in samples
                )
                hamiltonian_speedup = statistics.median(
                    float(row["default_over_selected_hamiltonian"]) for row in samples
                )
                backward_speedup = statistics.median(
                    float(row["default_over_selected_backward"]) for row in samples
                )
                # A direction with identical geometry has no treatment.  Its
                # component ratio is clock/order noise from the surrounding
                # whole-run pair and is therefore exactly a tie by design.
                if same_forward_parameters:
                    forward_speedup = 1.0
                if same_backward_parameters:
                    backward_speedup = 1.0
                # Hamiltonian code and launch geometry are shared in this A/B.
                hamiltonian_speedup = 1.0
                effect_estimator = "median_of_adjacent_paired_ratios"
                paired_samples = len(samples)
                ab_energy_error = max(float(row["energy_abs_error"]) for row in samples)
                ab_gradient_error = max(
                    float(row["gradient_max_abs_error"]) for row in samples
                )
                parameter_source = paired_path
            forward_directional_key = (circuit, qubits, "forward")
            if same_forward_parameters:
                forward_selection_speedup = 1.0
                forward_selection_estimator = "identical_forward_configuration"
                forward_selection_source = source_name(parameter_source)
            elif forward_directional_key in directional_shapes:
                forward_selection_speedup = directional_shapes[
                    forward_directional_key
                ]
                forward_selection_estimator = "direction_only_rotation_layer_median"
                forward_selection_source = source_name(directional_shapes_path)
            else:
                forward_selection_speedup = forward_speedup
                forward_selection_estimator = "paired_full_forward_component_median"
                forward_selection_source = source_name(parameter_source)
            backward_directional_key = (circuit, qubits, "backward")
            if same_backward_parameters:
                backward_selection_speedup = 1.0
                backward_selection_estimator = "identical_backward_configuration"
                backward_selection_source = source_name(parameter_source)
            elif backward_directional_key in directional_shapes:
                backward_selection_speedup = directional_shapes[
                    backward_directional_key
                ]
                backward_selection_estimator = "direction_only_rotation_layer_median"
                backward_selection_source = source_name(directional_shapes_path)
            else:
                backward_selection_speedup = backward_speedup
                backward_selection_estimator = "paired_full_backward_component_median"
                backward_selection_source = source_name(parameter_source)
            parameter_rows.append(
                {
                    "experiment_group": "execution_parameter_dispatch_experiment",
                    "timestamp_utc": selected["timestamp_utc"],
                    "hardware": HARDWARE,
                    "circuit": circuit,
                    "qubits": qubits,
                    "layers": selected["layers"],
                    "precision": selected["precision"],
                    "default_policy": "uniform_conservative_execution_same_structure",
                    "default_execution_parameters": default_execution,
                    "default_parameters": default_parameters,
                    "default_median_ms": parameter_default_ms,
                    "selected_policy": "circuit_size_execution_dispatch",
                    "selected_execution_parameters": selected_execution,
                    "shared_structural_parameters": shared_structure,
                    "selected_parameters": selected_parameters,
                    "selected_median_ms": parameter_selected_ms,
                    "speedup_vs_default": speedup,
                    "effect_estimator": effect_estimator,
                    "paired_samples": paired_samples,
                    "paired_ratio_mad": paired_ratio_mad,
                    "improvement_percent": 100 * (1 - 1 / speedup),
                    "forward_speedup_vs_default": forward_speedup,
                    "forward_selection_speedup_vs_default": forward_selection_speedup,
                    "forward_selection_estimator": forward_selection_estimator,
                    "forward_selection_source": forward_selection_source,
                    "hamiltonian_speedup_vs_default": hamiltonian_speedup,
                    "backward_speedup_vs_default": backward_speedup,
                    "backward_selection_speedup_vs_default": backward_selection_speedup,
                    "backward_selection_estimator": backward_selection_estimator,
                    "backward_selection_source": backward_selection_source,
                    "selected_variant": selected["kernel_variant"],
                    "default_variant": default["kernel_variant"],
                    "correct": int(ab_energy_error <= 1e-10 and ab_gradient_error <= 1e-9),
                    "energy_abs_error": ab_energy_error,
                    "gradient_max_abs_error": ab_gradient_error,
                    "source_selected": source_name(parameter_source),
                    "source_default": source_name(parameter_source),
                }
            )
    return main_rows, parameter_rows


def format_ms(value: object) -> str:
    number = float(value)
    if number < 0.01:
        return f"{number:.4f}"
    if number < 1:
        return f"{number:.3f}"
    return f"{number:.2f}"


def write_report(
    path: Path,
    main_rows: list[dict[str, object]],
) -> None:
    by_circuit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in main_rows:
        by_circuit[str(row["circuit"])].append(row)

    all_speedups = [float(row["speedup_vs_lightning_native"]) for row in main_rows]
    best_main = max(main_rows, key=lambda row: float(row["speedup_vs_lightning_native"]))
    large_rows = [row for row in main_rows if int(row["qubits"]) >= 20]
    backward_fractions = [
        float(row["sad_backward_mean_ms"])
        / (
            float(row["sad_forward_mean_ms"])
            + float(row["sad_hamiltonian_mean_ms"])
            + float(row["sad_backward_mean_ms"])
        )
        for row in large_rows
    ]
    max_energy_error = max(
        float(row["energy_abs_error_vs_lightning_native"]) for row in main_rows
    )
    max_gradient_error = max(
        float(row["gradient_max_abs_error_vs_lightning_native"]) for row in main_rows
    )
    variants = Counter(str(row["sad_kernel_variant"]) for row in main_rows)

    lines = [
        "# 主实验",
        "",
        "本报告由 `benchmark/generate_experiment_tables.py` 根据重新实测的原始 CSV "
        "生成。所有 wall time 均为同步后的完整一次 energy + 全梯度计算；表中只保留"
        "一个外部加速比：SAD 相对直接调用低层 `lightning_gpu_ops` 的 Lightning "
        "GPU native 基线。",
        "",
        "## 一、电路介绍",
        "",
        "实验统一使用 NVIDIA RTX 6000 Ada、float64、8 层、随机种子 42，规模为"
        " 4--28 个 qubit（偶数）。4--20q 预热 5 次，22--28q 预热 1 次；正式"
        "测量次数随规模采用 20/10/5/3/2。SAD 使用当前生产执行路径和按电路/规模"
        "选择的编译参数与少量运行时 phase plan。",
        "",
        "- **RA-HEA**：每层 `RY → ring CNOT`。SAD 使用实振幅融合快速路径。",
        "- **SU2-HEA**：每层 `RY → RZ → ring CNOT`；每个 qubit、每层各有"
        "一组 RY/RZ 参数。",
        "- **RZZ-HEA**：每层 `RX → RZ → even RZZ → odd RZZ`，在环形偶/奇"
        "bond 上作用。",
        "- **QAOA**：从 `|+⟩^n` 出发，每层为环形 MaxCut cost 的 even/odd "
        "RZZ，再施加共享角 RX mixer；每层两个参数。",
        "- **XXZ-HVA**：从 Néel 态出发，每层依次处理 even/odd matching 上的 "
        "RXX、RYY、RZZ；目标 Hamiltonian 为 `Σ(XX + YY + 0.5 ZZ)`。",
        "",
        "前三类 HEA 使用固定 TFIM 目标 Hamiltonian "
        "`H = -Σ Z_i Z_(i+1) - Σ X_i`；QAOA 使用环形 MaxCut cost。",
        "",
        "## 二、主实验",
        "",
        f"共 {len(main_rows)} 个场景。SAD/Lightning-native 加速比范围为 "
        f"**{min(all_speedups):.2f}×--{max(all_speedups):.2f}×**；最大值出现在 "
        f"{best_main['circuit']} {best_main['qubits']}q。能量最大绝对误差为 "
        f"{max_energy_error:.2e}，完整梯度最大绝对误差为 {max_gradient_error:.2e}。",
        "",
        "### 参数字符串说明",
        "",
        "主表把参数拆成‘执行参数’和‘结构/对角策略’，避免把不同层次的选择压成"
        "一个难以核对的标签。执行参数示例为 "
        "`F:t64r4m1-C[10+10+4]/B:t64r4m1-X[10+5+5+4]`：",
        "",
        "- `F`/`B` 分别表示 forward/backward；`t64` 是每个 CTA 的线程数；"
        "`r4` 表示每线程保存 `2^4=16` 个 amplitude。由此 tile 地址位数 "
        "`L = 5 + r + log2(t/32)`；`L` 不是固定的 5，固定的只是一个 warp 的"
        "低 5 个 lane 位。",
        "- `m1` 是 mailbox 切分数；`m1` 表示不切分的 full mailbox。active CTA/SM "
        "由寄存器、共享内存和 shape 推导，不是另一个独立选择参数。",
        "- `C[...]` 是 compact phase，`X[...]` 是 fixed-low-5，`P[...]` 是 "
        "pair-low-6。纯数字如 `C[7+3]` 表示 canonical map 中各 phase 的新 target "
        "数；`C[L2R2W0-L4R2W0]` 则是已经上线的非均匀 plan，逐 phase 给出放到"
        " lane/register/warp slot 的新 target 数。",
        "- `LxRyWz` 中的 `R` 是该 phase 放入 register slot 的 target 数，不是"
        "另一套逐 phase `r` 模板。当前 `t/r` 仍是每个方向一个编译期模板；phase "
        "division 与 slot map 在运行时选择。",
        "- `ZxN` 表示 XXZ 使用保持 bond 依赖关系的 cross-matching，共 `N` 个"
        "phase；它不是 rotation 的 compact/fixed-low-5 partition。",
        "- 当前主表全部使用 ordinary per-phase launch；persistent 未启用，因此不在"
        "每行重复。",
        "- 结构字符串中的 `fused/split`、`Fscatter/Bgather` 和 `D=...` 分别说明"
        "融合边界、CNOT 数据移动方向和 diagonal/lookup 实现；`k8` 是 8-bit lookup。",
    ]
    for circuit in CIRCUITS:
        lines.extend(
            [
                "",
                f"### {circuit}",
                "",
                "| q | SAD ms | F/H/B ms | Lightning native ms | 加速比 | 执行参数 | 结构/对角策略 |",
                "|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in by_circuit[circuit]:
            lines.append(
                f"| {row['qubits']} | {format_ms(row['sad_median_ms'])} | "
                f"{format_ms(row['sad_forward_mean_ms'])}/"
                f"{format_ms(row['sad_hamiltonian_mean_ms'])}/"
                f"{format_ms(row['sad_backward_mean_ms'])} | "
                f"{format_ms(row['lightning_native_median_ms'])} | "
                f"{float(row['speedup_vs_lightning_native']):.2f}× | "
                f"`{row['sad_execution_parameters']}` | "
                f"`{row['sad_structural_parameters']}` |"
            )
    lines.extend(
        [
            "",
            "### 基本观察",
            "",
            f"- 20q 以上，backward 占 SAD 三阶段时间的 "
            f"{100 * min(backward_fractions):.1f}%--"
            f"{100 * max(backward_fractions):.1f}%，仍是主要 wall-time 瓶颈。",
            "- 小规模结果同时受 launch、Python/native 边界和通用 adjoint 固定开销"
            "影响；大规模结果更能反映状态遍历、融合次数与显存带宽。",
            "- Lightning 基线直接调用预构造的低层 GPU Ops，排除了 QNode、Autograd "
            "和 transform 开销；但 forward 仍逐 gate 跨越 Python/nanobind 边界。",
            f"- 本次实际用到的 SAD 编译 variant 分布为 `{dict(variants)}`。参数选择"
            "不是单一大 shape 规则，而是由电路、规模和 forward/backward 资源压力"
            "共同决定。",
            "",
            "## 三、参数选择",
            "",
            "全固定基线、结构开关、执行参数、方向独立 shape、异构 phase 与 mailbox "
            "refinement 已从主实验正文分离，统一见 [参数选择](参数选择.md)。主实验"
            "不再重复参数搜索过程或内部基线加速比。",
            "",
            "机器可读数据：",
            "",
            "- [主实验 CSV](../../benchmark/results/main_experiment.csv)",
            "- [参数选择三级汇总 CSV](../../benchmark/results/parameter_selection_stages.csv)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--lightning", type=Path, default=DEFAULT_LIGHTNING)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument(
        "--directional-shapes", type=Path, default=DEFAULT_DIRECTIONAL_SHAPES
    )
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--parameter-output", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_rows, parameter_rows = build_tables(
        args.optimized,
        args.fixed,
        args.lightning,
        args.paired,
        args.directional_shapes,
    )
    write_csv(args.main_output, MAIN_FIELDS, main_rows)
    write_csv(args.parameter_output, PARAMETER_FIELDS, parameter_rows)
    write_report(args.report, main_rows)
    print(f"wrote {len(main_rows)} main rows to {args.main_output}")
    print(f"wrote {len(parameter_rows)} parameter rows to {args.parameter_output}")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
