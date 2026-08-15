"""Join SAD, low-level Lightning, and QNode Lightning benchmark results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS / "native_baseline_comparison.csv"
OUTPUT_REPORT = ROOT / "docs" / "experiments" / "NATIVE_BASELINE_COMPARISON.md"

FIELDS = (
    "circuit",
    "qubits",
    "layers",
    "parameter_count",
    "precision",
    "sad_time_median_s",
    "lightning_native_time_median_s",
    "pennylane_qnode_time_median_s",
    "speedup_vs_lightning_native",
    "speedup_vs_pennylane_qnode",
    "qnode_over_native",
    "lightning_native_forward_mean_s",
    "lightning_native_hamiltonian_mean_s",
    "lightning_native_backward_mean_s",
    "native_energy_abs_error_vs_qnode",
    "native_gradient_max_abs_error_vs_qnode",
    "sad_energy_abs_error_vs_native",
    "sad_gradient_max_abs_error_vs_native",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["circuit"], int(row["qubits"]), int(row["layers"])


def _index(rows: list[dict[str, str]], label: str) -> dict[tuple[str, int, int], dict[str, str]]:
    result: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        row_key = _key(row)
        if row_key in result:
            raise ValueError(f"duplicate {label} row: {row_key}")
        result[row_key] = row
    return result


def _pennylane_rows() -> list[dict[str, str]]:
    rows = _read(RESULTS / "pennylane_lightning_gpu.csv")
    extension = RESULTS / "qaoa_xxz_pennylane_gpu.csv"
    if extension.exists():
        rows += _read(extension)
    else:
        rows += _read(RESULTS / "qaoa_pennylane_gpu.csv")
    return rows


def _sad_rows() -> list[dict[str, str]]:
    optimized = _read(RESULTS / "sad_optimized_gpu.csv")
    rows = [row for row in optimized if row["circuit"] not in {"qaoa", "xxz-hva"}]

    qaoa = RESULTS / "qaoa_shared_sad_gpu.csv"
    if qaoa.exists():
        rows += _read(qaoa)
    else:
        rows += [row for row in optimized if row["circuit"] == "qaoa"]

    extension = RESULTS / "qaoa_xxz_sad_gpu.csv"
    if extension.exists():
        rows += [row for row in _read(extension) if row["circuit"] == "xxz-hva"]
    return rows


def _gradient(row: dict[str, str]) -> np.ndarray:
    return np.asarray(json.loads(row["grad_json"]), dtype=np.float64)


def _joined_rows() -> list[dict[str, object]]:
    native = _index(_read(RESULTS / "lightning_gpu_native.csv"), "native")
    qnode = _index(_pennylane_rows(), "QNode")
    sad = _index(_sad_rows(), "SAD")
    expected = set(native)
    missing_qnode = sorted(expected - set(qnode))
    missing_sad = sorted(expected - set(sad))
    if missing_qnode or missing_sad:
        raise ValueError(
            f"missing rows: QNode={missing_qnode or 'none'}, SAD={missing_sad or 'none'}"
        )

    rows: list[dict[str, object]] = []
    for row_key in sorted(expected, key=lambda key: (key[0], key[1], key[2])):
        native_row = native[row_key]
        qnode_row = qnode[row_key]
        sad_row = sad[row_key]
        native_time = float(native_row["time_median_s"])
        qnode_time = float(qnode_row["time_median_s"])
        sad_time = float(sad_row["time_median_s"])
        native_gradient = _gradient(native_row)
        qnode_gradient = _gradient(qnode_row)
        sad_gradient = _gradient(sad_row)
        if native_gradient.shape != qnode_gradient.shape or native_gradient.shape != sad_gradient.shape:
            raise ValueError(
                f"gradient shape mismatch for {row_key}: "
                f"native={native_gradient.shape}, QNode={qnode_gradient.shape}, "
                f"SAD={sad_gradient.shape}"
            )
        rows.append(
            {
                "circuit": row_key[0],
                "qubits": row_key[1],
                "layers": row_key[2],
                "parameter_count": native_row["parameter_count"],
                "precision": native_row["precision"],
                "sad_time_median_s": sad_time,
                "lightning_native_time_median_s": native_time,
                "pennylane_qnode_time_median_s": qnode_time,
                "speedup_vs_lightning_native": native_time / sad_time,
                "speedup_vs_pennylane_qnode": qnode_time / sad_time,
                "qnode_over_native": qnode_time / native_time,
                "lightning_native_forward_mean_s": native_row["forward_mean_s"],
                "lightning_native_hamiltonian_mean_s": native_row[
                    "hamiltonian_mean_s"
                ],
                "lightning_native_backward_mean_s": native_row["backward_mean_s"],
                "native_energy_abs_error_vs_qnode": abs(
                    float(native_row["energy"]) - float(qnode_row["energy"])
                ),
                "native_gradient_max_abs_error_vs_qnode": float(
                    np.max(np.abs(native_gradient - qnode_gradient))
                ),
                "sad_energy_abs_error_vs_native": abs(
                    float(sad_row["energy"]) - float(native_row["energy"])
                ),
                "sad_gradient_max_abs_error_vs_native": float(
                    np.max(np.abs(sad_gradient - native_gradient))
                ),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_range(values: list[float]) -> str:
    return f"{min(values):.2f}–{max(values):.2f}×"


def _write_report(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["circuit"])].append(row)

    circuit_order = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
    lines = [
        "# Lightning-GPU native 基线对比",
        "",
        "## 方法",
        "",
        "所有结果使用 RTX 6000 Ada、float64、8 layers、seed 42，以及完全相同的参数和 "
        "Hamiltonian。`SAD native` 是优化后 CUDA/C++ 的三阶段同步时间之和；"
        "`Lightning native` 预构造 circuit、observable、OpsData、state vector 和 adjoint "
        "对象，再通过 `lightning_gpu_ops` 测量同步后的 reset + forward + Hamiltonian + "
        "adjoint-gradient；`PennyLane QNode` 保留原 `qml.grad(qnode)` 端到端 wall time。",
        "",
        "为消除大规模配置释放显存后回到小规模时的瞬态抖动，4–20q 使用 5 次 warmup，"
        "22–28q 使用 1 次 warmup；正式测量次数仍采用 20/10/5/3/2 的规模调度。",
        "",
        "这个 packaged native binding 的 forward 仍然每个 gate 跨越一次 Python/nanobind "
        "边界；它移除了 QNode、Autograd、transforms 和逐 step 序列化，但不是独立的纯 "
        "C++ Lightning 可执行程序。",
        "",
        "## 汇总",
        "",
        "| Circuit | SAD/native 范围 | SAD/QNode 范围 | 4q QNode/native | 28q QNode/native | native 最大梯度误差 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for circuit in circuit_order:
        circuit_rows = sorted(grouped[circuit], key=lambda row: int(row["qubits"]))
        native_speedups = [float(row["speedup_vs_lightning_native"]) for row in circuit_rows]
        qnode_speedups = [float(row["speedup_vs_pennylane_qnode"]) for row in circuit_rows]
        max_gradient_error = max(
            float(row["native_gradient_max_abs_error_vs_qnode"])
            for row in circuit_rows
        )
        lines.append(
            f"| {circuit} | {_fmt_range(native_speedups)} | "
            f"{_fmt_range(qnode_speedups)} | "
            f"{float(circuit_rows[0]['qnode_over_native']):.2f}× | "
            f"{float(circuit_rows[-1]['qnode_over_native']):.2f}× | "
            f"{max_gradient_error:.2e} |"
        )

    four_qubit_rows = [row for row in rows if int(row["qubits"]) == 4]
    large_rows = [row for row in rows if int(row["qubits"]) >= 22]
    four_qubit_native_speedups = [
        float(row["speedup_vs_lightning_native"]) for row in four_qubit_rows
    ]
    four_qubit_qnode_ratios = [float(row["qnode_over_native"]) for row in four_qubit_rows]
    four_qubit_backward_fractions = [
        float(row["lightning_native_backward_mean_s"])
        / (
            float(row["lightning_native_forward_mean_s"])
            + float(row["lightning_native_hamiltonian_mean_s"])
            + float(row["lightning_native_backward_mean_s"])
        )
        for row in four_qubit_rows
    ]
    large_qnode_ratios = [float(row["qnode_over_native"]) for row in large_rows]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 4q 去掉 QNode/Autograd/序列化后，Lightning 缩短了 "
            f"`{min(four_qubit_qnode_ratios):.2f}–{max(four_qubit_qnode_ratios):.2f}×`；"
            f"但 SAD 相对低层 native 仍有 "
            f"`{min(four_qubit_native_speedups):.2f}–{max(four_qubit_native_speedups):.2f}×`。"
            "因此原小规模高加速比只有一部分来自 PennyLane 框架。",
            f"- 4q 的 Lightning native backward 占总时间 "
            f"`{100 * min(four_qubit_backward_fractions):.1f}%–"
            f"{100 * max(four_qubit_backward_fractions):.1f}%`，固定开销主要位于通用 "
            "adjoint 路径，而不是 forward 或 Hamiltonian。",
            f"- 22q 以后 QNode/native 为 "
            f"`{min(large_qnode_ratios):.2f}–{max(large_qnode_ratios):.2f}×`；"
            "两者已经同量级，少量反转来自分批运行、三阶段额外同步和测量波动，"
            "不应解释为低层接口在大规模上系统性更慢。",
        ]
    )

    for circuit in circuit_order:
        lines.extend(
            [
                "",
                f"## {circuit}",
                "",
                "| q | SAD ms | Lightning native ms | native fwd ms | native H ms | native bwd ms | vs native | QNode ms | vs QNode | QNode/native |",
                "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for row in sorted(grouped[circuit], key=lambda item: int(item["qubits"])):
            lines.append(
                f"| {row['qubits']} | "
                f"{1000 * float(row['sad_time_median_s']):.3f} | "
                f"{1000 * float(row['lightning_native_time_median_s']):.3f} | "
                f"{1000 * float(row['lightning_native_forward_mean_s']):.3f} | "
                f"{1000 * float(row['lightning_native_hamiltonian_mean_s']):.3f} | "
                f"{1000 * float(row['lightning_native_backward_mean_s']):.3f} | "
                f"{float(row['speedup_vs_lightning_native']):.2f}× | "
                f"{1000 * float(row['pennylane_qnode_time_median_s']):.3f} | "
                f"{float(row['speedup_vs_pennylane_qnode']):.2f}× | "
                f"{float(row['qnode_over_native']):.2f}× |"
            )

    max_energy_error = max(float(row["native_energy_abs_error_vs_qnode"]) for row in rows)
    max_gradient_error = max(
        float(row["native_gradient_max_abs_error_vs_qnode"]) for row in rows
    )
    max_sad_energy_error = max(float(row["sad_energy_abs_error_vs_native"]) for row in rows)
    max_sad_gradient_error = max(
        float(row["sad_gradient_max_abs_error_vs_native"]) for row in rows
    )
    lines.extend(
        [
            "",
            "## 数值校验",
            "",
            f"- Native/QNode 最大 energy absolute error：`{max_energy_error:.3e}`。",
            f"- Native/QNode 最大 gradient-element absolute error：`{max_gradient_error:.3e}`。",
            f"- SAD/native 最大 energy absolute error：`{max_sad_energy_error:.3e}`。",
            f"- SAD/native 最大 gradient-element absolute error：`{max_sad_gradient_error:.3e}`。",
            "",
            f"机器可读数据：`benchmark/results/{OUTPUT_CSV.name}`。",
            "",
        ]
    )
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = _joined_rows()
    _write_csv(rows)
    _write_report(rows)
    print(f"CSV written to {OUTPUT_CSV}")
    print(f"Markdown written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
