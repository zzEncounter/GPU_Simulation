#!/usr/bin/env python3
"""Plot SAD speedups over PennyLane and cuQuantum baselines."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


WIDTH = 960
HEIGHT = 600
MARGIN = {"left": 92, "right": 38, "top": 70, "bottom": 78}
COLORS = {"sad_pennylane": "#176B87", "sad_cuquantum": "#C44A35", "cuquantum_pennylane": "#5B7F36"}
DISPLAY_NAMES = {
    "equivariant-qnn": "Equivariant QNN",
    "mera": "MERA",
    "data-reuploading": "Data Re-uploading",
    "qaoa-ns": "Non-shared-angle QAOA",
}
CIRCUITS = tuple(DISPLAY_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results/native_baseline_comparison_merged_cuquantum.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results/figures",
    )
    parser.add_argument("--circuit", choices=CIRCUITS)
    return parser.parse_args()


def nice_upper_bound(value: float) -> int:
    step = 10 ** max(0, math.floor(math.log10(value)) - 1)
    return max(1, int(math.ceil(value / step) * step))


def load_rows(path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    grouped: dict[str, list[tuple[int, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            pennylane = float(row["pennylane_qnode_time_median_s"])
            sad = float(row["sad_time_median_s"])
            cuquantum = float(row["cuquantum_time_median_s"])
            if sad <= 0 or cuquantum <= 0:
                raise ValueError(f"Non-positive timing for {row['circuit']} / {row['qubits']} qubits")
            grouped.setdefault(row["circuit"], []).append(
                (int(row["qubits"]), pennylane / sad, cuquantum / sad, pennylane / cuquantum)
            )
    return {name: sorted(values) for name, values in grouped.items()}


def render_svg(circuit: str, values: list[tuple[int, float, float, float]]) -> str:
    left, right = MARGIN["left"], WIDTH - MARGIN["right"]
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    x_min, x_max = min(x for x, _, _, _ in values), max(x for x, _, _, _ in values)
    y_max = nice_upper_bound(max(max(sad_pl, sad_cu, cu_pl) for _, sad_pl, sad_cu, cu_pl in values) * 1.1)
    x_span = max(1, x_max - x_min)
    x_scale = lambda value: left + (value - x_min) / x_span * (right - left)
    y_scale = lambda value: bottom - value / y_max * (bottom - top)
    name = html.escape(DISPLAY_NAMES.get(circuit, circuit))
    sad_pl_points = " ".join(f"{x_scale(x):.1f},{y_scale(sad_pl):.1f}" for x, sad_pl, _, _ in values)
    sad_cu_points = " ".join(f"{x_scale(x):.1f},{y_scale(sad_cu):.1f}" for x, _, sad_cu, _ in values)
    cu_pl_points = " ".join(f"{x_scale(x):.1f},{y_scale(cu_pl):.1f}" for x, _, _, cu_pl in values)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#252525;letter-spacing:0}.tick{font-size:14px;fill:#555}.label{font-size:17px;font-weight:600}.title{font-size:23px;font-weight:700}.subtitle{font-size:14px;fill:#666}.value{font-size:13px;font-weight:600}.legend{font-size:14px}</style>',
        f'<text class="title" x="{left}" y="32">{name}: SAD Speedup Comparison</text>',
        f'<text class="subtitle" x="{left}" y="54">Baseline time / SAD time (higher is better)</text>',
    ]
    tick_count = 5
    for index in range(tick_count + 1):
        value = y_max * index / tick_count
        y = y_scale(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dedede" stroke-width="1"/>')
        elements.append(f'<text class="tick" x="{left - 12}" y="{y + 5:.1f}" text-anchor="end">{value:.0f}x</text>')
    for value in range(x_min, x_max + 1, 2):
        x = x_scale(value)
        elements.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 6}" stroke="#777"/>')
        elements.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 26}" text-anchor="middle">{value}</text>')
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777" stroke-width="1.2"/>',
        f'<polyline points="{sad_pl_points}" fill="none" stroke="{COLORS["sad_pennylane"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
        f'<polyline points="{sad_cu_points}" fill="none" stroke="{COLORS["sad_cuquantum"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="8 5"/>',
        f'<polyline points="{cu_pl_points}" fill="none" stroke="{COLORS["cuquantum_pennylane"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="3 4"/>',
    ])
    for x, sad_pl, sad_cu, cu_pl in values:
        elements.append(f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(sad_pl):.1f}" r="4.5" fill="#fff" stroke="{COLORS["sad_pennylane"]}" stroke-width="2.5"/>')
        elements.append(f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(sad_cu):.1f}" r="4.5" fill="#fff" stroke="{COLORS["sad_cuquantum"]}" stroke-width="2.5"/>')
        elements.append(f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(cu_pl):.1f}" r="4.5" fill="#fff" stroke="{COLORS["cuquantum_pennylane"]}" stroke-width="2.5"/>')
    legend_x = right - 270
    elements.extend([
        f'<line x1="{legend_x}" y1="15" x2="{legend_x + 28}" y2="15" stroke="{COLORS["sad_pennylane"]}" stroke-width="3"/>',
        f'<text class="legend" x="{legend_x + 36}" y="20">SAD vs PennyLane</text>',
        f'<line x1="{legend_x}" y1="36" x2="{legend_x + 28}" y2="36" stroke="{COLORS["sad_cuquantum"]}" stroke-width="3" stroke-dasharray="8 5"/>',
        f'<text class="legend" x="{legend_x + 36}" y="41">SAD vs cuQuantum</text>',
        f'<line x1="{legend_x}" y1="57" x2="{legend_x + 28}" y2="57" stroke="{COLORS["cuquantum_pennylane"]}" stroke-width="3" stroke-dasharray="3 4"/>',
        f'<text class="legend" x="{legend_x + 36}" y="62">cuQuantum vs PennyLane</text>',
        f'<text class="label" x="{(left + right) / 2:.1f}" y="{HEIGHT - 22}" text-anchor="middle">Number of qubits</text>',
        f'<text class="label" transform="translate(25 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Speedup over PennyLane QNode</text>',
        '</svg>',
    ])
    return "\n".join(elements) + "\n"


def main() -> None:
    args = parse_args()
    grouped = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuits = (args.circuit,) if args.circuit else CIRCUITS
    for circuit in circuits:
        if circuit not in grouped:
            raise ValueError(f"Missing circuit in input CSV: {circuit}")
        output = args.output_dir / f"{circuit}_sad_cuquantum_vs_pennylane_speedup.svg"
        output.write_text(render_svg(circuit, grouped[circuit]), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
