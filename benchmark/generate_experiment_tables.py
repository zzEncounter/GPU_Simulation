"""Generate the refreshed main experiment, parameter A/B table, and report."""

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
DEFAULT_MAIN = RESULTS / "main_experiment.csv"
DEFAULT_PARAMETERS = RESULTS / "parameter_search_experiment.csv"
DEFAULT_REPORT = ROOT / "docs" / "experiments" / "主实验.md"
HARDWARE = "NVIDIA RTX 6000 Ada"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
QUBITS = tuple(range(4, 29, 2))
VARIANT = re.compile(r"f(\d+)r(\d+)_b(\d+)r(\d+)$")

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
    "default_parameters",
    "default_median_ms",
    "selected_policy",
    "selected_parameters",
    "selected_median_ms",
    "speedup_vs_default",
    "effect_estimator",
    "paired_samples",
    "paired_ratio_mad",
    "improvement_percent",
    "forward_speedup_vs_default",
    "hamiltonian_speedup_vs_default",
    "backward_speedup_vs_default",
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


def strategy_descriptor(circuit: str, qubits: int) -> str:
    if circuit == "ra-hea":
        return "real-fused"
    if circuit == "su2-hea":
        return "F:lookup-fused/B:split"
    if circuit == "rzz-hea":
        backward = "split" if qubits >= 20 else "fused"
        return f"F:RZ+RZZ-fused/B:{backward}"
    if circuit == "qaoa":
        forward = "cost+RX-fused" if qubits >= 22 else "split"
        backward = "fused" if qubits >= 20 else "split"
        return f"F:{forward}/B:{backward}"
    if circuit == "xxz-hva":
        return "cross-matching"
    raise ValueError(circuit)


def parameter_descriptor(circuit: str, qubits: int, variant: str) -> str:
    match = VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError(f"invalid SAD variant {variant!r}")
    ft, fr, bt, br = match.groups()
    return (
        f"F{ft}r{fr}/B{bt}r{br};compact;m1;"
        f"{strategy_descriptor(circuit, qubits)}"
    )


def build_tables(
    optimized_path: Path,
    fixed_path: Path,
    lightning_path: Path,
    paired_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    optimized = index_complete(optimized_path, label="optimized SAD")
    fixed = index_complete(fixed_path, label="fixed-parameter SAD")
    lightning = index_complete(lightning_path, label="Lightning native")
    paired = paired_index(paired_path)
    main_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for circuit in CIRCUITS:
        for qubits in QUBITS:
            row_key = (circuit, qubits, 8)
            selected = optimized[row_key]
            default = fixed[row_key]
            reference = lightning[row_key]
            if default["kernel_variant"] != "f128r2_b128r2":
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
            selected_parameters = parameter_descriptor(
                circuit, qubits, selected["kernel_variant"]
            )
            default_parameters = parameter_descriptor(
                circuit, qubits, default["kernel_variant"]
            )
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
            if selected["kernel_variant"] == default["kernel_variant"]:
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
                effect_estimator = "median_of_adjacent_paired_ratios"
                paired_samples = len(samples)
                ab_energy_error = max(float(row["energy_abs_error"]) for row in samples)
                ab_gradient_error = max(
                    float(row["gradient_max_abs_error"]) for row in samples
                )
                parameter_source = paired_path
            parameter_rows.append(
                {
                    "experiment_group": "parameter_selection_experiment",
                    "timestamp_utc": selected["timestamp_utc"],
                    "hardware": HARDWARE,
                    "circuit": circuit,
                    "qubits": qubits,
                    "layers": selected["layers"],
                    "precision": selected["precision"],
                    "default_policy": "uniform_fixed_shape",
                    "default_parameters": default_parameters,
                    "default_median_ms": parameter_default_ms,
                    "selected_policy": "circuit_and_size_dispatch",
                    "selected_parameters": selected_parameters,
                    "selected_median_ms": parameter_selected_ms,
                    "speedup_vs_default": speedup,
                    "effect_estimator": effect_estimator,
                    "paired_samples": paired_samples,
                    "paired_ratio_mad": paired_ratio_mad,
                    "improvement_percent": 100 * (1 - 1 / speedup),
                    "forward_speedup_vs_default": forward_speedup,
                    "hamiltonian_speedup_vs_default": hamiltonian_speedup,
                    "backward_speedup_vs_default": backward_speedup,
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
    parameter_rows: list[dict[str, object]],
) -> None:
    by_circuit: dict[str, list[dict[str, object]]] = defaultdict(list)
    parameter_by_circuit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in main_rows:
        by_circuit[str(row["circuit"])].append(row)
    for row in parameter_rows:
        parameter_by_circuit[str(row["circuit"])].append(row)

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
    parameter_speedups = [float(row["speedup_vs_default"]) for row in parameter_rows]
    selected_wins = sum(value > 1.02 for value in parameter_speedups)
    fixed_wins = sum(value < 1 / 1.02 for value in parameter_speedups)
    tied = len(parameter_speedups) - selected_wins - fixed_wins
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
        "选择的编译参数。",
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
    ]
    for circuit in CIRCUITS:
        lines.extend(
            [
                "",
                f"### {circuit}",
                "",
                "| q | SAD ms | F/H/B ms | Lightning native ms | 加速比 | 选用参数 |",
                "|---:|---:|---:|---:|---:|---|",
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
                f"`{row['sad_selected_parameters']}` |"
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
            "## 三、参数选择实验",
            "",
            "### 选择因素与备选方案",
            "",
            "| 因素 | 搜索过的方案 | 为什么依赖场景 |",
            "|---|---|---|",
            "| kernel shape | 32--512 threads；每线程 2²--2⁶ amplitudes；F/B 可分开 | "
            "forward 与 backward 的寄存器活跃范围和 reduction 压力不同 |",
            "| phase 调度 | compact、fixed-low、pair；target 分配到 L/R/W slot | "
            "phase 数、lane shuffle、寄存器访问和 warp mailbox 成本互相制约 |",
            "| mailbox/occupancy | full mailbox、2--64 份切分、逐 phase 省略 | "
            "共享内存下降可能提高 CTA/SM，也可能被额外 barrier 抵消 |",
            "| diagonal/lookup | 64--512 threads、不同 lookup bit、split/fused | "
            "RZ/RZZ 的查表流量和与 rotation 融合后的寄存器压力不同 |",
            "| 电路融合 | split、forward fused、backward fused、双向 fused | "
            "减少 full-state pass 不一定抵消更长 live range；方向和 q 会改变边界 |",
            "| CNOT/XXZ layout | gather/scatter；separate/cross matching | "
            "读写连续性、双缓冲方向和 bond ownership 不同 |",
            "",
            "L2 hit 已从选择因素中移除：隔离复测没有得到足以改善 wall time 的稳定"
            "收益。最终选择始终依据端到端 wall time，而不是单个硬件 counter。",
            "",
            "### 缺省组定义",
            "",
            "缺省组仍使用同一份新版代码和同一 optimized 电路算法，只关闭按"
            " `(circuit, q)` 的编译 variant dispatch：所有场景固定 "
            "`F128r2/B128r2`、compact phase 和 full mailbox。这样不会把旧算法"
            "退化混进参数收益，A/B 只回答“是否值得按场景选择 kernel 参数”。",
            "",
            "21 个实际改变 variant 的场景采用相邻 A/B，并逐组交替先测 selected "
            "或 default；20--24q 测 5 对，26--28q 测 3 对，效果取配对比值中位数。"
            "其余 44 个场景的两个策略会选择完全相同的二进制，直接记为 1.000×，"
            "避免把不同轮次的频率漂移误算成参数收益。配对 MAD 保存在机器可读 CSV。",
            "",
            f"在 {len(parameter_rows)} 个端到端场景中，按 2% 阈值计，选择策略胜出 "
            f"{selected_wins} 个、固定缺省胜出 {fixed_wins} 个、近似持平 {tied} 个。"
            "这也说明调优 dispatch 应允许回退到缺省 variant，而不是强制每个场景"
            "使用不同配置。",
        ]
    )
    for circuit in CIRCUITS:
        lines.extend(
            [
                "",
                f"### {circuit}：选择策略 vs 统一缺省",
                "",
                "| q | 缺省 ms | 选择后 ms | wall 加速 | F 加速 | B 加速 | 选用参数 |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in parameter_by_circuit[circuit]:
            lines.append(
                f"| {row['qubits']} | {format_ms(row['default_median_ms'])} | "
                f"{format_ms(row['selected_median_ms'])} | "
                f"{float(row['speedup_vs_default']):.3f}× | "
                f"{float(row['forward_speedup_vs_default']):.3f}× | "
                f"{float(row['backward_speedup_vs_default']):.3f}× | "
                f"`{row['selected_parameters']}` |"
            )
    lines.extend(
        [
            "",
            "机器可读数据：",
            "",
            "- `benchmark/results/main_experiment.csv`",
            "- `benchmark/results/parameter_search_experiment.csv`",
            "- `benchmark/results/parameter_policy_paired_raw.csv`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--lightning", type=Path, default=DEFAULT_LIGHTNING)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--parameter-output", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_rows, parameter_rows = build_tables(
        args.optimized, args.fixed, args.lightning, args.paired
    )
    write_csv(args.main_output, MAIN_FIELDS, main_rows)
    write_csv(args.parameter_output, PARAMETER_FIELDS, parameter_rows)
    write_report(args.report, main_rows, parameter_rows)
    print(f"wrote {len(main_rows)} main rows to {args.main_output}")
    print(f"wrote {len(parameter_rows)} parameter rows to {args.parameter_output}")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
