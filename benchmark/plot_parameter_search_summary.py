"""Plot parameter-search timing and speedup summaries as standalone SVGs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "benchmark/results/native_baseline_comparison_merged_cuquantum.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs/experiments/figures/parameter_search"
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
    "qaoa-bd": "QAOA-BD",
    "qaoa-ns-bd": "QAOA-NS-BD",
    "xxz-hva-bd": "XXV-HEV-BD",
}
BD_FILES = {
    "qaoa-bd": ("qaoa", ROOT / "benchmark/results/parameter_selection_stages_raw.csv", ROOT / "benchmark/results/pennylane_lightning_gpu_bd.csv", ROOT / "benchmark/results/cuquantum_bd.csv"),
    "qaoa-ns-bd": ("qaoa-ns", ROOT / "benchmark/results/qaoa_ns_parameter_search.csv", ROOT / "benchmark/results/pennylane-cuQuantum-qaoa-ns-bd.csv", ROOT / "benchmark/results/pennylane-cuQuantum-qaoa-ns-bd.csv"),
    "xxz-hva-bd": ("xxz-hva", ROOT / "benchmark/results/parameter_selection_stages_raw.csv", ROOT / "benchmark/results/pennylane-cuQuantum-xxz-hva-bd.csv", ROOT / "benchmark/results/pennylane-cuQuantum-xxz-hva-bd.csv"),
}

WIDTH = 1280
HEIGHT = 760
LEFT = 90
RIGHT = 1235
PLOT_WIDTH = RIGHT - LEFT
TOP_Y0, TOP_Y1 = 92, 370
BOTTOM_Y0, BOTTOM_Y1 = 448, 690


def load_baselines(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("cuquantum_status", "ok") not in {"", "ok"}:
                continue
            result[(row["circuit"], int(row["qubits"]))] = (
                float(row["pennylane_qnode_time_median_s"]) * 1000.0,
                float(row["cuquantum_time_median_s"]) * 1000.0,
            )
    return result


def load_bd_data(name: str) -> tuple[dict, dict[tuple[str, int], tuple[float, float]]]:
    source, search_path, pl_path, cu_path = BD_FILES[name]
    with search_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_qubits = {}
    for qubit in range(4, 29, 2):
        candidates = []
        for row in rows:
            if row.get("qubits") != str(qubit) or row.get("status", "ok") != "ok":
                continue
            if source != "qaoa-ns" and (row.get("circuit") != source or row.get("correct") != "1"):
                continue
            candidates.append(float(row["median_ms"]))
        by_qubits[str(qubit)] = {
            "fastest": {"median_ms": min(candidates)},
            "average_median_ms": sum(candidates) / len(candidates),
        }

    def read_backend(path: Path, backend: str) -> dict[int, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {
                int(row["qubits"]): float(row["time_median_s"]) * 1000.0
                for row in csv.DictReader(stream)
                if row.get("status", "ok") == "ok" and row.get("backend", backend) == backend
            }

    if name == "qaoa-bd":
        pl = read_backend(pl_path, "lightning.gpu")
        cu = read_backend(cu_path, "custatevec-inverse-walk")
    else:
        pl = read_backend(pl_path, "pennylane")
        cu = read_backend(cu_path, "cuQuantum")
    baselines = {(name, q): (pl[q], cu[q]) for q in range(4, 29, 2)}
    return {"by_qubits": by_qubits}, baselines


def linear_bounds(values: list[float]) -> tuple[float, float]:
    """Return linear-axis bounds with a small headroom above the maximum."""
    high = max(values)
    if high <= 0:
        return 0.0, 1.0
    return 0.0, high * 1.05


def x_position(index: int, count: int) -> float:
    return LEFT + index * PLOT_WIDTH / (count - 1)


def y_position(value: float, bounds: tuple[float, float], top: int, bottom: int) -> float:
    low, high = bounds
    ratio = (value - low) / (high - low)
    return bottom - ratio * (bottom - top)


def format_tick(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def svg_text(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(value)}</text>'


def render_panel(
    lines: list[str],
    qubits: list[int],
    series: list[tuple[str, list[float], str, str]],
    bounds: tuple[float, float],
    top: int,
    bottom: int,
    ylabel: str,
) -> None:
    lines.append(f'<rect x="{LEFT}" y="{top}" width="{PLOT_WIDTH}" height="{bottom-top}" fill="#ffffff"/>')
    for index in range(6):
        value = bounds[0] + (bounds[1] - bounds[0]) * index / 5.0
        y = y_position(value, bounds, top, bottom)
        lines.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" stroke="#d9dee5"/>')
        lines.append(svg_text(LEFT - 12, y + 5, format_tick(value), text_anchor="end", font_size="13", fill="#52606d"))
    for index, qubit in enumerate(qubits):
        x = x_position(index, len(qubits))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#eef1f4"/>')
        lines.append(svg_text(x, bottom + 24, str(qubit), text_anchor="middle", font_size="13", fill="#52606d"))
    lines.append(f'<line x1="{LEFT}" y1="{top}" x2="{LEFT}" y2="{bottom}" stroke="#3d4752" stroke-width="1.5"/>')
    lines.append(f'<line x1="{LEFT}" y1="{bottom}" x2="{RIGHT}" y2="{bottom}" stroke="#3d4752" stroke-width="1.5"/>')
    lines.append(svg_text(25, (top + bottom) / 2, ylabel, text_anchor="middle", font_size="14", fill="#20262d", transform=f"rotate(-90 25 {(top + bottom) / 2:.1f})"))

    for _, values, color, dash in series:
        points = " ".join(
            f"{x_position(i, len(qubits)):.1f},{y_position(value, bounds, top, bottom):.1f}"
            for i, value in enumerate(values)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>' )
        for i, value in enumerate(values):
            x = x_position(i, len(qubits))
            y = y_position(value, bounds, top, bottom)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')


def render_legend(lines: list[str], series: list[tuple[str, list[float], str, str]], y: int) -> None:
    item_width = PLOT_WIDTH / len(series)
    for index, (label, _, color, dash) in enumerate(series):
        x = LEFT + index * item_width
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x+30:.1f}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>' )
        lines.append(svg_text(x + 38, y + 5, label, font_size="13", fill="#303841"))


def render_circuit(circuit: str, data: dict, baselines: dict[tuple[str, int], tuple[float, float]]) -> str:
    rows = sorted(data["by_qubits"].items(), key=lambda item: int(item[0]))
    qubits = [int(q) for q, _ in rows]
    fastest = [float(item["fastest"]["median_ms"]) for _, item in rows]
    average = [float(item["average_median_ms"]) for _, item in rows]
    pennylane = [baselines[(circuit, q)][0] for q in qubits]
    cuquantum = [baselines[(circuit, q)][1] for q in qubits]

    timing_series = [
        ("SAD fastest", fastest, "#00796b", ""),
        ("SAD average", average, "#d97706", ""),
        ("PennyLane", pennylane, "#6d28d9", "8 5"),
        ("cuQuantum", cuquantum, "#2563eb", "8 5"),
    ]
    speedup_series = [
        ("PL / SAD fastest", [p / s for p, s in zip(pennylane, fastest)], "#6d28d9", ""),
        ("PL / SAD average", [p / s for p, s in zip(pennylane, average)], "#c026d3", "7 4"),
        ("cuQ / SAD fastest", [c / s for c, s in zip(cuquantum, fastest)], "#2563eb", ""),
        ("cuQ / SAD average", [c / s for c, s in zip(cuquantum, average)], "#0891b2", "7 4"),
    ]
    timing_bounds = linear_bounds([value for _, values, _, _ in timing_series for value in values])
    speedup_bounds = linear_bounds([value for _, values, _, _ in speedup_series for value in values] + [1.0])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f7f8fa"/>',
        svg_text(LEFT, 36, f"{DISPLAY_NAMES[circuit]} parameter-search performance", font_size="24", font_weight="700", fill="#171b20"),
        svg_text(LEFT, 62, "Absolute runtime and speedup across qubit counts", font_size="14", fill="#5d6772"),
        svg_text(LEFT, TOP_Y0 - 14, "Absolute time", font_size="16", font_weight="700", fill="#20262d"),
    ]
    render_panel(lines, qubits, timing_series, timing_bounds, TOP_Y0, TOP_Y1, "Time (ms)")
    render_legend(lines, timing_series, TOP_Y1 + 52)
    lines.append(svg_text(LEFT, BOTTOM_Y0 - 14, "Speedup", font_size="16", font_weight="700", fill="#20262d"))
    render_panel(lines, qubits, speedup_series, speedup_bounds, BOTTOM_Y0, BOTTOM_Y1, "Speedup (x)")
    one_y = y_position(1.0, speedup_bounds, BOTTOM_Y0, BOTTOM_Y1)
    lines.append(f'<line x1="{LEFT}" y1="{one_y:.1f}" x2="{RIGHT}" y2="{one_y:.1f}" stroke="#7b8794" stroke-width="1.5" stroke-dasharray="3 4"/>')
    render_legend(lines, speedup_series, HEIGHT - 34)
    lines.append(svg_text((LEFT + RIGHT) / 2, HEIGHT - 8, "Qubits", text_anchor="middle", font_size="14", fill="#20262d"))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    baselines = load_baselines(args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for circuit, path in SEARCH_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        output = args.output_dir / f"{circuit}.svg"
        output.write_text(render_circuit(circuit, data, baselines), encoding="utf-8")
        print(output.relative_to(ROOT))
    for circuit in BD_FILES:
        data, bd_baselines = load_bd_data(circuit)
        output = args.output_dir / f"{circuit}.svg"
        output.write_text(render_circuit(circuit, data, bd_baselines), encoding="utf-8")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
