"""Generate a Markdown report from the four parameter-search JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "benchmark/results/native_baseline_comparison_merged_cuquantum.csv"
DEFAULT_OUTPUT = ROOT / "docs/experiments/PARAMETER_SEARCH_SUMMARY.md"
SEARCH_FILES = {
    "mera": ROOT / "benchmark/results/mera_parameter_search.json",
    "equivariant-qnn": ROOT / "benchmark/results/equivariant_qnn_parameter_search.json",
    "data-reuploading": ROOT / "benchmark/results/data_reuploading_parameter_search.json",
    "qaoa-ns": ROOT / "benchmark/results/qaoa_ns_parameter_search.json",
}
DISPLAY_NAMES = {
    "mera": "MERA",
    "equivariant-qnn": "Equivariant-QNN",
    "data-reuploading": "Data Re-uploading",
    "qaoa-ns": "QAOA-NS",
}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _speedup(reference_ms: float | None, search_ms: float | None) -> float | None:
    if reference_ms is None or search_ms is None or search_ms <= 0:
        return None
    return reference_ms / search_ms


def _gmean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    return math.exp(sum(math.log(value) for value in positive) / len(positive)) if positive else None


def load_baselines(path: Path) -> dict[tuple[str, int], dict[str, float | str]]:
    result: dict[tuple[str, int], dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("cuquantum_status", "ok") not in {"", "ok"}:
                continue
            result[(row["circuit"], int(row["qubits"]))] = {
                "pennylane_ms": float(row["pennylane_qnode_time_median_s"]) * 1000.0,
                "cuquantum_ms": float(row["cuquantum_time_median_s"]) * 1000.0
                if row.get("cuquantum_time_median_s") else None,
            }
    return result


def load_search(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render(search_data: dict[str, dict], baselines: dict[tuple[str, int], dict]) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# 四个电路参数搜索与基准对比",
        "",
        f"> 生成时间：`{generated}`",
        "",
        "## 口径",
        "",
        "本报告使用四个参数搜索 JSON 中每个 qubit 的**当前最快已完成组合**。搜索时间为完整 `forward + Hamiltonian + backward` 调用的 `median_ms`。",
        "每个 qubit 同时保留该 JSON 的最快、最慢和平均搜索时间。",
        "",
        "加速比定义为：",
        "",
        "```text",
        "PennyLane 加速比 = PennyLane median time / 搜索最优 SAD median time",
        "cuQuantum 加速比 = cuQuantum median time / 搜索最优 SAD median time",
        "```",
        "",
        "PennyLane 和 cuQuantum 时间来自 `benchmark/results/native_baseline_comparison_merged_cuquantum.csv`，时间单位统一为 ms。搜索 JSON 中的最快组合是候选筛选结果，不等同于已经写入生产 dispatch 的最终配置。",
        "",
        "## 总览",
        "",
        "| 电路 | 已完成 | 失败 | 搜索最优几何均值(ms) | PennyLane/SAD 几何均值 | cuQuantum/SAD 几何均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for circuit, data in search_data.items():
        rows = data["by_qubits"]
        best = [float(item["fastest"]["median_ms"]) for item in rows.values()]
        pl = []
        cu = []
        for q, item in rows.items():
            baseline = baselines.get((circuit, int(q)), {})
            search_ms = float(item["fastest"]["median_ms"])
            pl_speed = _speedup(baseline.get("pennylane_ms"), search_ms)
            cu_speed = _speedup(baseline.get("cuquantum_ms"), search_ms)
            if pl_speed is not None:
                pl.append(pl_speed)
            if cu_speed is not None:
                cu.append(cu_speed)
        lines.append(
            f"| {DISPLAY_NAMES[circuit]} | {data['completed_rows']} | {data['failed_rows']} | "
            f"{_fmt(_gmean(best))} | {_fmt(_gmean(pl))}x | {_fmt(_gmean(cu))}x |"
        )

    for circuit, data in search_data.items():
        lines.extend([
            "",
            f"## {DISPLAY_NAMES[circuit]}",
            "",
            "| qubits | 最快 ms | 平均 ms | 最慢 ms | 最快阶段 | PennyLane ms | PL/SAD | cuQuantum ms | cuQ/SAD |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ])
        for q, item in sorted(data["by_qubits"].items(), key=lambda pair: int(pair[0])):
            fastest = item["fastest"]
            slowest = item["slowest"]
            baseline = baselines.get((circuit, int(q)), {})
            search_ms = float(fastest["median_ms"])
            pl_ms = baseline.get("pennylane_ms")
            cu_ms = baseline.get("cuquantum_ms")
            lines.append(
                f"| {q} | {_fmt(search_ms)} | {_fmt(float(item['average_median_ms']))} | "
                f"{_fmt(float(slowest['median_ms']))} | `{fastest['stage']}` | "
                f"{_fmt(pl_ms)} | {_fmt(_speedup(pl_ms, search_ms))}x | "
                f"{_fmt(cu_ms)} | {_fmt(_speedup(cu_ms, search_ms))}x |"
            )
        lines.extend([
            "",
            "最优配置（JSON 中的当前最快候选）：",
            "",
            "| qubits | candidate key | 配置 |",
            "|---:|---|---|",
        ])
        for q, item in sorted(data["by_qubits"].items(), key=lambda pair: int(pair[0])):
            fastest = item["fastest"]
            config = json.loads(fastest["config"]) if isinstance(fastest["config"], str) else fastest["config"]
            compact = ", ".join(f"{key}={value}" for key, value in sorted(config.items()))
            lines.append(f"| {q} | `{fastest['candidate_key']}` | `{compact}` |")

    lines.extend([
        "",
        "## 注意事项",
        "",
        "- 搜索 JSON 中的 `failed_rows` 是编译资源限制或运行失败的候选，不参与最快、最慢和平均时间统计。",
        "- PennyLane、cuQuantum 与 SAD 的 benchmark 均使用 float64 和相同 qubit 列表；除 MERA 使用其拓扑要求的 `ceil(log2(qubits))` layers 外，其余电路使用 8 layers。具体 warmup/steps 以各自 CSV 字段为准。",
        "- 加速比大于 `1x` 表示搜索最优 SAD 更快；小于 `1x` 表示对应基准更快。",
        "- 本报告只做时间比较，不重新验证能量/梯度正确性；正确性字段仍保留在原始 benchmark CSV 中。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = {circuit: load_search(path) for circuit, path in SEARCH_FILES.items()}
    baselines = load_baselines(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data, baselines), encoding="utf-8")
    print(f"Markdown written to {args.output}")


if __name__ == "__main__":
    main()
