"""生成执行策略搜索综合研究报告。"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_execution_search import (
    MODEL_FEATURES,
    confirmed_rows,
    fit_model,
    load_aggregates,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
DEFAULT_ROTATION = RESULTS / "execution_search_exhaustive.csv"
DEFAULT_HELDOUT = RESULTS / "execution_search_final.csv"
DEFAULT_ADAPTIVE = RESULTS / "adaptive_mailbox_summary.csv"
DEFAULT_XXZ = RESULTS / "xxz_search_summary.csv"
DEFAULT_FUSION = RESULTS / "fusion_search_summary.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "research" / "EXECUTION_STRATEGY_REPORT.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sample_coverage(path: Path) -> tuple[int, int, Counter[int]]:
    rows = read_csv(path)
    grouped: dict[tuple[str, ...], int] = Counter()
    identity = (
        "stage", "variant", "family", "candidate", "gate", "direction",
        "qubits", "layout",
    )
    for row in rows:
        grouped[tuple(row[field] for field in identity)] += 1
    return len(rows), len(grouped), Counter(grouped.values())


def heldout_metrics(
    path: Path,
) -> list[tuple[str, str, str, str, float, float]]:
    rows = load_aggregates(path)
    result: list[tuple[str, str, str, str, float, float]] = []
    for gate in ("rx", "ry"):
        for direction in ("forward", "backward"):
            train = [
                row for row in rows
                if row.stage in {"shape", "schedule"}
                and row.qubits < 28
                and row.gate == gate
                and row.direction == direction
            ]
            test = [
                row for row in rows
                if row.stage == "shape"
                and row.qubits == 28
                and row.gate == gate
                and row.direction == direction
            ]
            prediction = fit_model(train).predict(test)
            predicted = min(range(len(test)), key=lambda index: prediction[index])
            measured = min(range(len(test)), key=lambda index: test[index].median_ms)
            regret = test[predicted].median_ms / test[measured].median_ms - 1
            median_ape = statistics.median(
                abs(prediction[index] - test[index].median_ms)
                / test[index].median_ms
                for index in range(len(test))
            )
            result.append(
                (
                    gate.upper(), "正向" if direction == "forward" else "反向",
                    f"{test[predicted].variant}/{test[predicted].family}",
                    f"{test[measured].variant}/{test[measured].family}",
                    regret,
                    median_ape,
                )
            )
    return result


def fusion_factors(
    circuit: str, variant: str
) -> tuple[str, str] | None:
    """Return independently controlled forward/backward factor labels."""

    if circuit == "ra-hea":
        if not variant.startswith("complex-"):
            return None
        return (
            "fused" if "fused-forward" in variant or variant.endswith("both")
            else "split",
            "fused" if "fused-backward" in variant or variant.endswith("both")
            else "split",
        )
    if circuit in {"su2-hea", "rzz-hea"}:
        if variant == "auto":
            return None
        forward, backward = variant.split("-forward_", 1)
        return forward, backward.removesuffix("-backward")
    if circuit == "qaoa":
        return {
            "split-both": ("split", "split"),
            "fused-forward": ("fused", "split"),
            "fused-backward": ("split", "fused"),
            "fused-both": ("fused", "fused"),
        }.get(variant)
    if circuit == "xxz-hva":
        return variant, variant
    raise ValueError(circuit)


def factor_winner(
    rows: list[dict[str, str]], circuit: str, direction: str
) -> tuple[str, float, float]:
    """Median away the orthogonal factor before ranking a fusion choice."""

    field = f"{direction}_ms"
    grouped: dict[str, list[float]] = defaultdict(list)
    factor_index = 0 if direction == "forward" else 1
    for row in rows:
        factors = fusion_factors(circuit, row["variant"])
        if factors is not None:
            grouped[factors[factor_index]].append(float(row[field]))
    medians = {
        factor: statistics.median(samples)
        for factor, samples in grouped.items()
    }
    winner = min(medians, key=lambda factor: (medians[factor], factor))
    ordered = sorted(medians.values())
    margin = math.inf if len(ordered) == 1 else ordered[1] / ordered[0] - 1
    return winner, medians[winner], margin


def same_shape_full_mailbox(
    rows: list[object], selected: object
) -> object | None:
    candidates = [
        row for row in rows
        if row.threads == selected.threads
        and row.register_amplitudes == selected.register_amplitudes
        and row.family == selected.family
        and row.candidate == selected.candidate
        and row.mailbox_chunks == 1
    ]
    return min(candidates, key=lambda row: row.median_ms) if candidates else None


def best_partition_comparison(
    rows: list[object],
) -> tuple[object, object] | None:
    """Best same-geometry chunked/full pair, ranked by measured speedup."""

    pairs: list[tuple[float, object, object]] = []
    for selected in rows:
        if selected.mailbox_chunks == 1:
            continue
        full = same_shape_full_mailbox(rows, selected)
        if full is not None:
            pairs.append((selected.median_ms / full.median_ms, full, selected))
    if not pairs:
        return None
    _, full, selected = min(pairs, key=lambda item: item[0])
    return full, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=Path, default=DEFAULT_ROTATION)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--xxz", type=Path, default=DEFAULT_XXZ)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rotation_rows, rotation_candidates, repetitions = sample_coverage(
        args.rotation
    )
    rotation = confirmed_rows(load_aggregates(args.rotation), 3)
    schedule = [row for row in rotation if row.stage == "schedule"]
    fusion = read_csv(args.fusion)
    xxz = read_csv(args.xxz)
    heldout = heldout_metrics(args.heldout)
    adaptive = read_csv(args.adaptive) if args.adaptive.exists() else []
    effective_adaptive = [
        row for row in adaptive
        if int(row["adaptive_changes_execution"])
    ]
    adaptive_by_scenario: dict[
        tuple[str, str, int], list[dict[str, str]]
    ] = defaultdict(list)
    for row in effective_adaptive:
        adaptive_by_scenario[
            row["gate"], row["direction"], int(row["qubits"])
        ].append(row)
    adaptive_winners = [
        min(rows, key=lambda row: float(row["adaptive_ms"]))
        for rows in adaptive_by_scenario.values()
    ]
    correct = sum(int(row["correct"]) * int(row["samples"]) for row in fusion)
    fusion_samples = sum(int(row["samples"]) for row in fusion)
    xxz_samples = sum(int(row["samples"]) for row in xxz)

    lines = [
        "# 执行策略搜索",
        "",
        "由 `benchmark/generate_strategy_report.py` 生成。计时均取中位数；"
        "差距在 2% 以内的配置视为近似持平。",
        "",
        "## 硬件与搜索覆盖范围",
        "",
        "目标硬件：NVIDIA RTX 6000 Ada，计算能力 8.9，142 个 SM，"
        "每个 warp 32 个线程，每个 SM 有 65,536 个寄存器和 100 KiB "
        "共享内存，L2 缓存为 96 MiB。",
        "",
        f"RX/RY 主搜索包含 {rotation_rows:,} 行原始结果和 "
        f"{rotation_candidates:,} 个硬件等价候选；各候选重复次数的分布为 "
        f"`{dict(sorted(repetitions.items()))}`。shape 阶段枚举所有能够编译且"
        "合法的 7--12 bit tile，以及 mailbox 从不切分到 64 份之间所有 2 的"
        "幂次切分；已知会超过 CUDA 资源限制的 launch 会被明确剔除。随后，"
        "针对每个场景留下的候选，在所有可达的 (lane, register, warp) 类别上"
        "用动态规划生成 phase 调度。每个生成的调度先测量一次；每个场景的前"
        "五名或距最优值 5% 以内的调度，再以打乱顺序补足三次测量。下文排名"
        "只采用拥有三次样本的调度。",
        "",
        f"融合搜索包含 {fusion_samples:,} 个端到端样本；其中 "
        f"{correct:,}/{fusion_samples:,} 通过了与旧实现的能量和梯度校验。",
        "",
        f"XXZ shape/分区搜索包含 {xxz_samples:,} 个样本，每个候选均重复三次。"
        "canonical 与非均匀分区的直接对比检查通过了两种 parity：状态最大误差"
        "为 0，梯度最大误差为 1.137e-13。",
        *(
            [
                "",
                f"自适应 mailbox 确认实验包含 {len(adaptive):,} 个确实会改变"
                "执行路径的候选。每个候选按交替顺序测量九组相邻的 "
                "full/adaptive 配对；效果取配对比值的中位数，MAD 用于表示抖动。",
            ]
            if adaptive else []
        ),
        "",
        "## RX/RY phase 最优配置",
        "",
        "| 门 | 方向 | q | variant/family | 各 phase 的 L/R/W | "
        "mailbox / CTA | 毫秒 |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    grouped_schedule: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for row in schedule:
        grouped_schedule[row.gate, row.direction, row.qubits].append(row)
    for scenario, rows in sorted(grouped_schedule.items()):
        best = min(rows, key=lambda row: row.median_ms)
        phase_classes = "/".join(
            f"{lane},{register},{warp}"
            for lane, register, warp in zip(
                best.phase_lane_targets,
                best.phase_register_targets,
                best.phase_warp_targets,
                strict=True,
            )
        )
        lines.append(
            f"| {scenario[0].upper()} | "
            f"{'正向' if scenario[1] == 'forward' else '反向'} | "
            f"{scenario[2]} | `{best.variant}/{best.family}` | "
            f"`{phase_classes}` | "
            f"{best.mailbox_bytes} B / {best.active_cta_per_sm} | "
            f"{best.median_ms:.6f} |"
        )
    lines.extend(
        [
            "",
            "低 target phase 的最优选择并不单调：正向路径通常偏好 phase 数最少"
            "的 compact 调度；反向路径有时则以较小的首尾 phase 或固定的低 lane "
            "数量取胜。切分 mailbox 会增加 barrier，但能缩小共享内存和活跃变量"
            "范围，因此必须按门类型和方向分别评估。",
            "",
            "## Mailbox 与 active CTA 的权衡",
            "",
            "下表列出每个场景中提升最大的同 shape、同调度 chunked/full-mailbox "
            "对比，从而把 mailbox 大小的影响与 phase 数量及 target 位置隔离开。",
            "",
            "| 门 | 方向 | q | shape/family | 完整 mailbox | 分块 mailbox | "
            "CTA/SM | 耗时变化 |",
            "|---|---:|---:|---|---:|---:|---|---:|",
        ]
    )
    grouped_shape: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for row in rotation:
        if row.stage == "shape":
            grouped_shape[row.gate, row.direction, row.qubits].append(row)
    for scenario, rows in sorted(grouped_shape.items()):
        comparison = best_partition_comparison(rows)
        if comparison is None:
            continue
        full, chunked = comparison
        change = chunked.median_ms / full.median_ms - 1
        lines.append(
            f"| {scenario[0].upper()} | "
            f"{'正向' if scenario[1] == 'forward' else '反向'} | "
            f"{scenario[2]} | `t{full.threads}r{full.register_bits}/"
            f"{full.family}` | {full.mailbox_bytes} B | "
            f"{chunked.mailbox_bytes} B (m{chunked.mailbox_chunks}) | "
            f"{full.active_cta_per_sm}→{chunked.active_cta_per_sm} | "
            f"{100 * change:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "单 warp block 不需要 amplitude mailbox；这是一个真正的零 mailbox "
            "方案，而不是 `m=infinity`。不过它的 tile 较小，可能增加遍历完整状态"
            "所需的 phase 数。反过来，切分多 warp tile 的 mailbox 能保持 phase map "
            "不变，但每次 warp-target 交换大约会额外增加正向 `2m` 或反向 `4m` 个 "
            "CTA barrier。上表揭示了共享内存下降何时足以跨过 occupancy 台阶，"
            "从而补偿这些额外 barrier。",
        ]
    )
    if adaptive:
        lines.extend(
            [
                "",
                "下面是逐 phase 自适应的配对结果（仅 W=0 的 phase 省略 mailbox）。"
                "单 warp/无 mailbox 以及所有 phase 均满足 W>0 的调度，已被识别为"
                "执行路径相同的对照组，因此不纳入最终配对实验。下表每行是在生成"
                "的 kernel 确有差异的候选中，自适应执行耗时最短者：",
                "",
                "| 门 | 方向 | q | 候选 | 各 phase 的 W | CTA/SM "
                "完整→省略 | 配对耗时变化 | 配对 MAD |",
                "|---|---:|---:|---|---|---|---:|---:|",
            ]
        )
        for scenario, rows in sorted(adaptive_by_scenario.items()):
            best = min(rows, key=lambda row: float(row["adaptive_ms"]))
            lines.append(
                f"| {scenario[0].upper()} | "
                f"{'正向' if scenario[1] == 'forward' else '反向'} | "
                f"{scenario[2]} | `{best['variant']}/{best['family']}` | "
                f"`{best['phase_warp_targets']}` | "
                f"{best['full_active_cta_per_sm']}→"
                f"{best['no_mailbox_active_cta_per_sm']} | "
                f"{100 * (float(best['adaptive_relative']) - 1):+.2f}% | "
                f"{100 * float(best['paired_ratio_mad']):.2f}% |"
            )
    lines.extend(
        [
            "",
            "## XX 与 YY 的差异及耦合边分区",
            "",
            "| 方向 | q | 奇偶组 | 融合 XX+YY+ZZ 最优 shape | 耦合边分区 | 毫秒 |",
            "|---|---:|---:|---|---|---:|",
        ]
    )
    for row in xxz:
        if row["stage"] == "partition" and row["rank"] == "1":
            lines.append(
                f"| {'正向' if row['direction'] == 'forward' else '反向'} | "
                f"{row['qubits']} | "
                f"{row['parity']} | `{row['variant']}` | "
                f"`{row['candidate']}` | {float(row['median_ms']):.6f} |"
            )
    shape_xxz = [row for row in xxz if row["stage"] == "shape"]
    ratios: list[float] = []
    for direction in ("forward", "backward"):
        for q in (20, 24, 26):
            for parity in (0, 1):
                xx = min(
                    float(row["median_ms"]) for row in shape_xxz
                    if row["component"] == "xx"
                    and row["direction"] == direction
                    and int(row["qubits"]) == q
                    and int(row["parity"]) == parity
                )
                yy = min(
                    float(row["median_ms"]) for row in shape_xxz
                    if row["component"] == "yy"
                    and row["direction"] == direction
                    and int(row["qubits"]) == q
                    and int(row["parity"]) == parity
                )
                ratios.append(yy / xx - 1)
    lines.extend(
        [
            "",
            f"在各场景独立选择最优 shape 后，YY 比 XX 慢 "
            f"{100 * min(ratios):.1f}%--{100 * max(ratios):.1f}%。YY 中依赖 "
            "`Z_i Z_j` 的 partner 符号会改变指令和寄存器压力；即使二者使用相同"
            "的 pair map，也不具备相同的资源特征。",
            "",
            "RX/RY 调度的每个 slot 只计算一个 qubit target；XX/YY 则必须把每条"
            "互不相交 bond 的两个端点放入同一个 tile。因此，含 `b` 个 bit 的 tile "
            "至多容纳 `floor(b/2)` 条 bond。XXZ 还需携带三个相互对易的系数，并在"
            "反向传播中为每条 bond 做三次梯度 reduction。因此，其 bond 组合搜索"
            "与 RX/RY 的 lane/register/warp 动态规划相互独立。",
            "",
            "在 q=26 的 separate-matching 反向路径上，canonical 紧凑分区比非均匀"
            "分区慢 2.18%--2.94%。生产路径目前使用更快的 cross-matching kernel，"
            "所以这些 separate-kernel 分区结果用于说明分区成本，而不会直接改变"
            "生产 dispatch。",
            "",
            "## 电路融合边界",
            "",
            "下表的正向/反向选择采用因子化中位数：改变另一个方向的策略后将其"
            "影响取中位数消除，避免无关因素的计时噪声决定融合边界。`≈` 表示差距"
            "在 2% 以内。",
            "",
            "| 电路 | q | 正向因子 | 反向因子 | 端到端最优方案 |",
            "|---|---:|---|---|---|",
        ]
    )
    for circuit in ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva"):
        for q in (20, 22, 24, 26, 28):
            rows = [
                row for row in fusion
                if row["circuit"] == circuit and int(row["qubits"]) == q
            ]
            if not rows:
                continue
            forward, _, forward_margin = factor_winner(
                rows, circuit, "forward"
            )
            backward, _, backward_margin = factor_winner(
                rows, circuit, "backward"
            )
            total = min(rows, key=lambda row: float(row["total_ms"]))
            lines.append(
                f"| {circuit} | {q} | "
                f"`{forward}{' ≈' if forward_margin <= 0.02 else ''}` | "
                f"`{backward}{' ≈' if backward_margin <= 0.02 else ''}` | "
                f"`{total['variant']}` |"
            )
    lines.extend(
        [
            "",
            "对实测融合边界的因子化解释如下：",
            "",
            "- RA 保留实振幅快速路径。对于复振幅研究路径，正向 CNOT 融合在 22q "
            "及以下近似持平，从 24q 起胜出；反向则从 22q 起胜出。",
            "- SU2 正向使用 lookup 融合的 RY+RZ+CNOT；从 20q 起，反向拆分 "
            "CNOT/RZ/RY。分 phase 路径在较大 q 下明显落后。",
            "- RZZ 正向保持将 RZ/RZZ 融入 RX，但从 20q 起将反向的四次 pass "
            "全部拆开；全融合反向路径扩大的活跃变量范围，超过了减少状态 pass "
            "所带来的收益。",
            "- QAOA 从 22q 起在正向将 cost 融入 RX，并从 20q 起采用 compact "
            "融合反向路径。XXZ 保留 cross-matching。",
            "",
            "## 白盒模型与留出集验证",
            "",
            "校准后的非负成本模型为：",
            "",
            "`T = c_mem*state_pass_GiB + c_lane*lane_gate_G + "
            "c_reg*register_gate_G + c_warp*warp_gate_G + "
            "c_smem*mailbox_GiB + c_barrier*barrier_M + "
            "c_atomic*gradient_atomic_M + c_wave*CTA_waves + "
            "c_occ*occupancy_pressure_GiB + c_launch*phase_count`.",
            "",
            "在 q=20/24/26 上训练，并在独立的 q=28 shape 测量中进行选择，"
            "结果如下：",
            "",
            "| 门 | 方向 | 模型预测 | 实测最优 | APE 中位数 | 选择遗憾 |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for gate, direction, predicted, measured, regret, median_ape in heldout:
        lines.append(
            f"| {gate} | {direction} | `{predicted}` | `{measured}` | "
            f"{100 * median_ape:.2f}% | {100 * regret:.2f}% |"
        )
    lines.extend(
        [
            "",
            "因此，该模型适合作为搜索先验，而不是可移植的闭式最优选择公式："
            "留出集上的选择遗憾接近 9%，按 family 外推 RY 时还可能更差。资源台阶、"
            "编译器寄存器分配以及 launch 合法性仍是不连续因素。",
            "",
            "下表给出主实验数据上拟合的非负系数，特征顺序与上式一致。系数为零"
            "表示该相关项无法被独立辨识，并不表示其硬件成本真的为零。",
            "",
            "| 门 | 方向 | " + " | ".join(MODEL_FEATURES) + " |",
            "|---|---:|" + "---:|" * len(MODEL_FEATURES),
        ]
    )
    model_rows = [
        row for row in rotation if row.stage in {"shape", "schedule"}
    ]
    for gate, direction in itertools.product(
        ("rx", "ry"), ("forward", "backward")
    ):
        subset = [
            row for row in model_rows
            if row.gate == gate and row.direction == direction
        ]
        coefficients = fit_model(subset).coefficients
        lines.append(
            f"| {gate.upper()} | {direction[0].upper()} | "
            + " | ".join(f"{value:.5g}" for value in coefficients)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 可移植调优流程",
            "",
            "1. 探测 SM 数量以及寄存器/共享内存上限；编译 7--12 bit tile "
            "及所有 mailbox 切分因子，并剔除 launch 不合法的配置。",
            "2. 对每个 `(gate,direction,q)`，以打乱顺序的重复轮次测量 "
            "compact/fixed/pair 完整 layer；保留五个 shape 候选，对其 DP 调度"
            "各筛选一次，再将前五名或 5% frontier 内的候选补足三次测量。",
            "3. 校准 empty/lane/register/warp phase，拟合白盒模型，以动态规划"
            "遍历可达 phase 类别，再实测预测 frontier 以及多一个 phase 的方案。",
            "4. 以旧实现的能量和梯度为正确性基准，重新进行端到端融合边界测量；"
            "绝不能只依据 ms/gate 选择融合方式。",
            "5. 使用一个留出的 q 进行验证，并保留所有差距在 2% 以内的方案；"
            "当差距小于该阈值时，优先采用更简单的规则。",
            "",
            "在这块 GPU 上，逐 phase 省略 mailbox 并不是普适规则：在各场景"
            "自适应执行最快且确实会改变执行路径的候选中，相邻配对耗时变化的"
            f"中位数范围为 {100 * (min(float(row['adaptive_relative']) for row in adaptive_winners) - 1):+.2f}% "
            f"至 {100 * (max(float(row['adaptive_relative']) for row in adaptive_winners) - 1):+.2f}%；"
            f"另有 {12 - len(adaptive_winners)} 个场景在其 2% frontier 上没有"
            "实际执行差异。因此，它应保留为需要实测后启用的可选开关；生产路径"
            "默认仍使用 full-mailbox。",
            "",
            "任意 pure-rotation phase map 在语义上是正确的，但融合 RZZ 反向路径"
            "目前仍根据 canonical phase 公式推导 edge ownership。非 canonical "
            "调度必须先加入显式 owner map，才能安全地成为通用生产 dispatch。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {args.output}")


if __name__ == "__main__":
    main()
