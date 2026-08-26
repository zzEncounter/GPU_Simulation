"""Generate the SAD-best versus Qiskit runtime comparison report."""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs" / "search" / "JOINT_PHASE_SEARCH_RUNTIME_SUMMARY.md"
QISKIT_RESULTS = ROOT / "qiskit" / "results" / "qiskit_aer_gpu_merged.csv"
OUTPUT = ROOT / "qiskit" / "results" / "SAD_VS_QISKIT_SPEEDUP.md"
CIRCUITS = (
    "ra-hea",
    "su2-hea",
    "rzz-hea",
    "qaoa",
    "qaoa-ns",
    "equivariant-qnn",
    "data-reuploading",
    "xxz-hva",
    "mera",
)


def read_sad_best() -> dict[tuple[str, int], float]:
    lines = SUMMARY.read_text(encoding="utf-8").splitlines()
    result: dict[tuple[str, int], float] = {}
    circuit: str | None = None
    in_runtime_table = False

    for line in lines:
        heading = re.fullmatch(r"## ([a-z0-9-]+)", line)
        if heading:
            circuit = heading.group(1)
            in_runtime_table = False
            continue
        if circuit not in CIRCUITS:
            continue
        if line.startswith("| qubits | 有效数 | SAD 最好 |"):
            in_runtime_table = True
            continue
        if not in_runtime_table or line.startswith("|---"):
            continue
        if not line.startswith("|"):
            in_runtime_table = False
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        result[circuit, int(cells[0])] = float(cells[2])

    expected = {(circuit, qubits) for circuit in CIRCUITS for qubits in range(4, 29, 2)}
    missing = sorted(expected - result.keys())
    if missing:
        raise ValueError(f"missing SAD best values: {missing}")
    return result


def read_qiskit() -> dict[tuple[str, int], dict[str, str]]:
    with QISKIT_RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["circuit"], int(row["qubits"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate circuit/qubits rows in merged Qiskit results")
    return result


def main() -> None:
    sad = read_sad_best()
    qiskit = read_qiskit()
    details: dict[str, list[tuple[int, float, float | None, float | None]]] = {}

    for circuit in CIRCUITS:
        rows = []
        for qubits in range(4, 29, 2):
            sad_ms = sad[circuit, qubits]
            qiskit_row = qiskit[circuit, qubits]
            if qiskit_row["status"] == "ok":
                qiskit_ms = float(qiskit_row["time_median_s"]) * 1000
                speedup = qiskit_ms / sad_ms
            else:
                qiskit_ms = None
                speedup = None
            rows.append((qubits, sad_ms, qiskit_ms, speedup))
        details[circuit] = rows

    lines = [
        "# SAD 最优配置相对 Qiskit 的加速比",
        "",
        "加速比定义为 `Qiskit time_median_s × 1000 / SAD 最好 median_ms`。数值大于 1 表示 SAD 更快。SAD 最优时间来自 `../../docs/search/JOINT_PHASE_SEARCH_RUNTIME_SUMMARY.md`，Qiskit 时间来自 `qiskit_aer_gpu_merged.csv`。Qiskit 使用 8 layers、float64 和 random seed 42；除 MERA 外，SAD 汇总也使用 8 layers。",
        "",
        "Qiskit 未完成的测试显示为“超时”，由于没有有限的运行时间，不计算加速比。",
        "",
        "> MERA 的 SAD 独立参数搜索包含 2–5 layers，而 Qiskit MERA 记录为 8 layers，因此 MERA 加速比仅表示当前两个结果文件的运行时间之比，不是同深度受控对比。",
        "",
        "## 汇总",
        "",
        "| 电路 | 完成点数 | 最小加速比 | 最大加速比 | Qiskit 超时 |",
        "|---|---:|---:|---:|---:|",
    ]
    for circuit in CIRCUITS:
        finite = [(q, speedup) for q, _, _, speedup in details[circuit] if speedup is not None]
        min_q, min_speedup = min(finite, key=lambda item: item[1])
        max_q, max_speedup = max(finite, key=lambda item: item[1])
        timeout_count = len(details[circuit]) - len(finite)
        lines.append(
            f"| {circuit} | {len(finite)}/13 | {min_speedup:.6f}× ({min_q}q) | "
            f"{max_speedup:.6f}× ({max_q}q) | {timeout_count} |"
        )

    for circuit in CIRCUITS:
        lines += [
            "",
            f"## {circuit}",
            "",
            "| qubits | SAD 最优 ms | Qiskit 中位数 ms | Qiskit/SAD 加速比 |",
            "|---:|---:|---:|---:|",
        ]
        for qubits, sad_ms, qiskit_ms, speedup in details[circuit]:
            if qiskit_ms is None:
                lines.append(f"| {qubits} | {sad_ms:.6f} | 超时 | 超时 |")
            else:
                lines.append(f"| {qubits} | {sad_ms:.6f} | {qiskit_ms:.6f} | {speedup:.6f}× |")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
