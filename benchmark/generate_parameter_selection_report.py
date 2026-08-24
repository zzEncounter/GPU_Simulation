"""Generate the single Chinese parameter-selection report."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
DEFAULT_STAGES = RESULTS / "parameter_selection_stages.csv"
DEFAULT_PARAMETERS = RESULTS / "parameter_search_experiment.csv"
DEFAULT_STRUCTURE = RESULTS / "structure_strategy_experiment.csv"
DEFAULT_POSITION = RESULTS / "rotation_position_group_summary.csv"
DEFAULT_HETERO = RESULTS / "heterogeneous_phase_paired_raw.csv"
DEFAULT_MAILBOX = RESULTS / "mailbox_phase_refinement_summary.csv"
DEFAULT_REPORT = ROOT / "docs" / "experiments" / "参数选择.md"
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
QUBITS = tuple(range(4, 29, 2))

DECISION_LABELS = {
    "ra_real_amplitude_fused_path": ("RA real-fused / complex", "SAD_REAL_AMPLITUDE"),
    "su2_forward": ("SU2 forward split / lookup / phased", "SAD_SU2_FORWARD_STRATEGY"),
    "su2_backward": ("SU2 backward split / lookup / phased", "SAD_SU2_BACKWARD_STRATEGY"),
    "rzz_forward": ("RZZ forward fused / split", "SAD_RZZ_FORWARD_FUSED"),
    "rzz_backward": ("RZZ backward split / diagonal / fused", "SAD_RZZ_BACKWARD_STRATEGY"),
    "qaoa_forward": ("QAOA forward fused / split", "SAD_QAOA_FUSE_COST_RX"),
    "qaoa_backward": ("QAOA backward fused / split", "SAD_QAOA_FUSED_BACKWARD"),
    "qaoa_initial_layer": ("QAOA initial 三种策略", "SAD_QAOA_INITIAL_STRATEGY"),
    "qaoa_domain_wall_lookup": ("QAOA compact / chunk lookup", "SAD_QAOA_COMPACT_LOOKUP"),
    "qaoa_shared_diagonal_threads": ("QAOA diagonal threads", "SAD_SHARED_DIAGONAL_BLOCK_THREADS"),
    "diagonal_lookup_bits": ("普通 diagonal lookup bits", "SAD_DIAGONAL_LOOKUP_BITS"),
    "ordinary_diagonal_threads": ("普通 diagonal threads", "SAD_DIAGONAL_BLOCK_THREADS"),
    "diagonal_reduction": ("diagonal CTA / warp-atomic", "SAD_DIAGONAL_WARP_ATOMIC"),
    "xxz_matching": ("XXZ separate / cross matching", "SAD_XXZ_CROSS_MATCHING"),
}

FIXED_POLICY = {
    "ra-hea": "complex RA；RY/CNOT split；CNOT F/B gather",
    "su2-hea": "RY/RZ/CNOT split；RZ k8/t64；CNOT F/B gather",
    "rzz-hea": "RX/RZ/RZZ split；RZ/RZZ k8/t64",
    "qaoa": "split init/cost/RX；chunk lookup；diagonal t128",
    "xxz-hva": "XX/YY/ZZ fused algebra；separate matching",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def format_ms(value: object) -> str:
    number = float(value)
    if number < 1:
        return f"{number:.3f}"
    return f"{number:.2f}"


def geometric_mean(values: list[float]) -> float:
    return statistics.geometric_mean(values)


def _validate_stage_rows(rows: list[dict[str, str]]) -> None:
    keys = {(row["circuit"], int(row["qubits"])) for row in rows}
    expected = {(circuit, q) for circuit in CIRCUITS for q in QUBITS}
    if keys != expected or len(rows) != len(expected):
        raise ValueError(
            f"stage result coverage mismatch: missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    if any(row["correct"] != "1" for row in rows):
        raise ValueError("stage results contain a correctness failure")


def write_report(
    path: Path,
    stage_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
    structure_rows: list[dict[str, str]],
    position_rows: list[dict[str, str]],
    hetero_rows: list[dict[str, str]],
    mailbox_rows: list[dict[str, str]],
) -> None:
    _validate_stage_rows(stage_rows)
    parameter_index = {
        (row["circuit"], int(row["qubits"])): row for row in parameter_rows
    }
    if set(parameter_index) != {
        (circuit, q) for circuit in CIRCUITS for q in QUBITS
    }:
        raise ValueError("execution parameter table is incomplete")

    by_circuit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stage_rows:
        by_circuit[row["circuit"]].append(row)
    for rows in by_circuit.values():
        rows.sort(key=lambda row: int(row["qubits"]))

    all_total = [float(row["fully_selected_speedup_vs_all_fixed"]) for row in stage_rows]
    all_structure = [float(row["structure_speedup_vs_all_fixed"]) for row in stage_rows]
    all_execution = [float(row["execution_speedup_vs_structure_selected"]) for row in stage_rows]
    small = [
        float(row["fully_selected_speedup_vs_all_fixed"])
        for row in stage_rows
        if int(row["qubits"]) <= 16
    ]
    max_energy = max(float(row["max_energy_abs_error"]) for row in stage_rows)
    max_gradient = max(float(row["max_gradient_abs_error"]) for row in stage_rows)

    lines = [
        "# 参数选择",
        "",
        "本文是唯一的参数选择报告；主实验只保留最终 SAD 相对 Lightning native 的结果。"
        "这里把优化拆成三个相邻阶段，避免再把‘同结构的固定执行参数’误称为‘所有参数固定’。",
        "",
        "## 一、结论",
        "",
        f"在 NVIDIA RTX 6000 Ada、float64、8 层、4--28q 的 65 个完整 energy + 全梯度场景中，"
        f"最终策略相对全固定基线的逐场景加速范围为 **{min(all_total):.3f}×--{max(all_total):.3f}×**，"
        f"几何平均为 **{geometric_mean(all_total):.3f}×**。最大能量/梯度误差为 "
        f"{max_energy:.2e}/{max_gradient:.2e}。",
        "",
        f"小规模并未跳过：4--16q 的全固定→最佳加速范围为 "
        f"**{min(small):.3f}×--{max(small):.3f}×**。结构阶段的 65 场景几何平均为 "
        f"{geometric_mean(all_structure):.3f}×，执行阶段为 {geometric_mean(all_execution):.3f}×。"
        "两阶段贡献依电路和 q 而变，不能用一个比例拆分所有场景。",
        "",
        "三个阶段的定义如下：",
        "",
        "1. **全固定（原始固定基线）**：所有已暴露的搜索轴都固定；所有 q 共用 `F128r2/B128r2`、canonical phase、`m1`；"
        "结构开关固定为 complex RA、split SU2/RZZ/QAOA、普通 diagonal、XXZ separate matching、CNOT gather。",
        "2. **结构已选、执行固定（可选中间列）**：打开生产结构/diagonal 规则，但仍使用 "
        "`F128r2/B128r2` 和 canonical phase。",
        "3. **最佳**：再打开按 `(circuit,q,direction)` 选择的 F/B shape 和少量 phase plan。",
        "",
        "‘原始’在这里是当前代码上可重复的保守操作基线，不声称等同于某个历史 commit。"
        "除运行时 phase map 外，候选 shape、mailbox 与结构策略都以预编译模板存在；运行时 dispatch 只选择已编译 variant。",
        "",
        "中间阶段只是归因工具。若只关心结果，可直接看‘全固定→最佳’和最后一列；"
        "CSV 保留了中间阶段，便于决定是否启用结构或执行 dispatch。",
        "",
        "### 各电路汇总",
        "",
        "| 电路 | 全固定策略（相关部分） | 结构阶段几何均值 | 执行阶段几何均值 | 总加速几何均值 | 总加速范围 | 28q 全固定→最佳 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for circuit in CIRCUITS:
        rows = by_circuit[circuit]
        structure = [float(row["structure_speedup_vs_all_fixed"]) for row in rows]
        execution = [
            float(row["execution_speedup_vs_structure_selected"]) for row in rows
        ]
        total = [float(row["fully_selected_speedup_vs_all_fixed"]) for row in rows]
        q28 = next(row for row in rows if int(row["qubits"]) == 28)
        lines.append(
            f"| {circuit} | {FIXED_POLICY[circuit]} | {geometric_mean(structure):.3f}× | "
            f"{geometric_mean(execution):.3f}× | {geometric_mean(total):.3f}× | "
            f"{min(total):.3f}--{max(total):.3f}× | "
            f"{format_ms(q28['all_fixed_median_ms'])}→{format_ms(q28['fully_selected_median_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## 二、全固定 → 部分选择 → 最佳",
            "",
            "以下每行来自同一轮中的三级配对；4--24q 三轮，26--28q 两轮，测量顺序循环轮换。"
            "`结构` 是全固定/结构已选，`执行` 是结构已选/最佳，`总计` 是全固定/最佳。",
            "表中的 ms 和加速比各自取配对样本中位数，所以显示值相除可能与比值末位略有不同。",
        ]
    )
    for circuit in CIRCUITS:
        lines.extend(
            [
                "",
                f"### {circuit}",
                "",
                "| q | 全固定 ms | +结构 ms | 最佳 ms | 结构 | 执行 | 总计 | 最佳执行参数 | 最佳结构/diagonal |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in by_circuit[circuit]:
            parameter = parameter_index[(circuit, int(row["qubits"]))]
            lines.append(
                f"| {row['qubits']} | {format_ms(row['all_fixed_median_ms'])} | "
                f"{format_ms(row['structure_selected_median_ms'])} | "
                f"{format_ms(row['fully_selected_median_ms'])} | "
                f"{float(row['structure_speedup_vs_all_fixed']):.3f}× | "
                f"{float(row['execution_speedup_vs_structure_selected']):.3f}× | "
                f"{float(row['fully_selected_speedup_vs_all_fixed']):.3f}× | "
                f"`{parameter['selected_execution_parameters']}` | "
                f"`{parameter['shared_structural_parameters']}` |"
            )

    evidence_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in structure_rows:
        evidence_groups[row["decision"]].append(row)
    lines.extend(
        [
            "",
            "## 三、快速结构与 diagonal 开关",
            "",
            "第二节的全固定→结构阶段已在当前源码上重新做聚合配对；下面复用此前边界 sweep 中的稀疏单变量 "
            "yes/no 或少量枚举，各行 CSV 保留各自 source fingerprint。它们不做参数直积，每次保持生产 "
            "shape、phase、输入和层数不变。"
            "2% 内保留较简单或当前生产选择；因此‘0 个反例’表示当前候选集没有稳定反例，不代表任意 GPU 上全局最优。",
            "",
            "| 开关 | 编译宏 | 比较数 | 生产更快 | 2% 内持平 | 候选更快 | 候选/生产范围 | 当前生产选择 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for decision, rows in sorted(evidence_groups.items()):
        label, macro = DECISION_LABELS[decision]
        verdicts = Counter(row["verdict_at_2_percent"] for row in rows)
        ratios = [float(row["candidate_over_production"]) for row in rows]
        choices = "/".join(sorted({row["production_choice"] for row in rows}))
        lines.append(
            f"| {label} | `{macro}` | {len(rows)} | {verdicts['production_faster']} | "
            f"{verdicts['near_tie_keep_production']} | {verdicts['candidate_faster']} | "
            f"{min(ratios):.3f}--{max(ratios):.3f}× | {choices} |"
        )
    lines.extend(
        [
            "",
            "CNOT 融合与 CNOT 数据方向不是同一个开关。融合在 RA/SU2 电路结构中有端到端收益；"
            "但 legacy standalone CNOT 的 F-scatter/B-gather 相对 gather-both 只在 2% 内近似持平，"
            "所以不能概括成所有 CNOT 相关改动都稳赚。RZZ/QAOA 融合也保留 q-dependent 回退。",
            "",
            "## 四、RX/RY：方向、position、phase 与 mailbox",
            "",
            "forward 和 backward 独立选择。forward 只传播 state，通常偏好 compact 和较轻 shape；"
            "backward 还传播 adjoint、归约梯度并承担 mailbox/共享内存压力，因此可使用不同 `t/r` 与 phase map。"
            "`L = 5 + r + log2(t/32)`，不可变的是 warp 的 32 lane（低 5 lane bits），不是 `L=5`。",
            "",
        ]
    )
    families: dict[tuple[int, str], list[float]] = defaultdict(list)
    family_max: dict[tuple[int, str], float] = defaultdict(float)
    family_noise: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in position_rows:
        key = (int(row["qubits"]), row["family"])
        families[key].append(float(row["position_spread_percent"]))
        family_max[key] = max(family_max[key], float(row["position_spread_percent"]))
        family_noise[key].append(float(row["median_relative_mad_percent"]))
    lines.extend(
        [
            "bit-position 扫描证明读取连续性不能只用 compact/fixed 一个标签概括：",
            "",
            "| q | continuity | position spread 中位数/最大值 | 计时相对 MAD 中位数 |",
            "|---:|---|---:|---:|",
        ]
    )
    family_names = {"compact": "compact", "fixed": "fixed-low-5", "pairs": "pair-low-6"}
    for key in sorted(families):
        q, family = key
        lines.append(
            f"| {q} | {family_names[family]} | {statistics.median(families[key]):.1f}%/"
            f"{family_max[key]:.1f}% | {statistics.median(family_noise[key]):.2f}% |"
        )

    hetero_grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    hetero_schedule: dict[tuple[str, str, int, str], str] = {}
    for row in hetero_rows:
        key = (row["gate"], row["direction"], int(row["qubits"]), row["candidate"])
        hetero_grouped[key].append(float(row["uniform_over_candidate"]))
        hetero_schedule[key] = row["schedule"]
    best_hetero: dict[tuple[str, str, int], tuple[str, float, str]] = {}
    for (gate, direction, q, candidate), values in hetero_grouped.items():
        scenario = (gate, direction, q)
        item = (candidate, statistics.median(values), hetero_schedule[(gate, direction, q, candidate)])
        if scenario not in best_hetero or item[1] > best_hetero[scenario][1]:
            best_hetero[scenario] = item
    lines.extend(
        [
            "",
            "异构 phase 不是把孤立 `ms/gate` 相加：finalist 在同一 state 上逐 phase 执行并配对。"
            "各场景最佳实测候选如下；RX forward 28q 有稳定约 8% layer 收益，RY backward 24q 有约 3% 的边界候选；"
            "两者都尚未通过各目标电路的最终 E2E，因而没有直接部署。",
            "",
            "| gate/direction/q | 最佳候选 | uniform/heterogeneous | schedule |",
            "|---|---|---:|---|",
        ]
    )
    for scenario, (candidate, speedup, schedule) in sorted(best_hetero.items()):
        gate, direction, q = scenario
        lines.append(
            f"| {gate.upper()} {direction} {q}q | {candidate} | {speedup:.3f}× | `{schedule}` |"
        )

    lines.extend(
        [
            "",
            "固定 phase 后又独立扫描了 mailbox cut；下表给出组合验证。相对全 `m1` 的收益不能替代相对原 phase winner 的比较。",
            "",
            "| gate/direction/q | selected cuts | vs 全 m1 | vs 原 phase winner | 决策 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in mailbox_rows:
        lines.append(
            f"| {row['gate'].upper()} {row['direction']} {row['qubits']}q | "
            f"{row['selected_cuts'] or '无'} | {float(row['actual_speedup']):.3f}× | "
            f"{float(row['refined_speedup_vs_source']):.3f}× | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "结论是：mailbox cut 值得保留为便宜的 refinement 测试，但本轮没有一个新 cut 同时胜过全 `m1` 和"
            "原 phase winner，因此不增加 production dispatch。active CTA 是 shape/mailbox 的资源结果，不单独作为开关。",
            "",
            "phase 数通常是一级因素，但不是唯一因素。同 phase 数下，CTA 大小、寄存器压力、position/continuity 仍能显著改变 wall time；"
            "compact phase 也可能因完整电路组合入选，即使其隔离 `ms/gate` 不是最低。24q 的旧观察不能外推到所有 q。",
            "",
            "## 五、L2 cache",
            "",
            "L2 hit 已从选择目标和产品功能中移除。现有隔离复测没有证明‘提高 L2 hit’能稳定改善完整 wall time；"
            "保留相关 CSV 只作历史研究证据，不再生成 L2-aware dispatch、额外开关或主表列。",
            "",
            "## 六、机器可读数据与复现",
            "",
            "- [三级参数选择汇总](../../benchmark/results/parameter_selection_stages.csv)",
            "- [三级参数选择原始配对](../../benchmark/results/parameter_selection_stages_raw.csv)",
            "- [同结构执行参数 A/B](../../benchmark/results/parameter_search_experiment.csv)",
            "- [结构与 diagonal 单变量汇总](../../benchmark/results/structure_strategy_experiment.csv)",
            "- [结构与 diagonal 原始配对](../../benchmark/results/structure_policy_paired_raw.csv)",
            "- [方向-only shape 配对](../../benchmark/results/directional_rotation_shape_raw.csv)",
            "- [bit-position 汇总](../../benchmark/results/rotation_position_group_summary.csv)",
            "- [异构 phase 完整 layer 配对](../../benchmark/results/heterogeneous_phase_paired_raw.csv)",
            "- [逐 phase mailbox 汇总](../../benchmark/results/mailbox_phase_refinement_summary.csv)",
            "- [三级配对程序](../../benchmark/benchmark_parameter_selection_stages.py)",
            "- [结构开关程序](../../benchmark/benchmark_structure_policy.py)",
            "- [异构 phase 程序](../../benchmark/benchmark_heterogeneous_phases.py)",
            "- [mailbox refinement 程序](../../benchmark/benchmark_phase_mailbox_refinement.py)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path, default=DEFAULT_STAGES)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--position", type=Path, default=DEFAULT_POSITION)
    parser.add_argument("--heterogeneous", type=Path, default=DEFAULT_HETERO)
    parser.add_argument("--mailbox", type=Path, default=DEFAULT_MAILBOX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        args.report,
        read_csv(args.stages),
        read_csv(args.parameters),
        read_csv(args.structure),
        read_csv(args.position),
        read_csv(args.heterogeneous),
        read_csv(args.mailbox),
    )
    print(f"wrote parameter-selection report to {args.report}")


if __name__ == "__main__":
    main()
