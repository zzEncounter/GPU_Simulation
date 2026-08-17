#!/usr/bin/env python3
"""Plot SAD speedups over PennyLane QNode from the merged benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


WIDTH = 960
HEIGHT = 600
MARGIN = {"left": 92, "right": 38, "top": 70, "bottom": 78}
COLORS = {
    "equivariant-qnn": "#176B87",
    "mera": "#C44A35",
    "data-reuploading": "#5B7F36",
    "qaoa-ns": "#8A4F9E",
}
DISPLAY_NAMES = {
    "equivariant-qnn": "Equivariant QNN",
    "mera": "MERA",
    "data-reuploading": "Data Re-uploading",
    "qaoa-ns": "Non-shared-angle QAOA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results/native_baseline_comparison_merged.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results/figures",
    )
    parser.add_argument(
        "--circuit",
        choices=tuple(DISPLAY_NAMES),
        help="Render only one circuit; useful for a circuit-specific CSV.",
    )
    return parser.parse_args()


def nice_upper_bound(value: float) -> int:
    step = 10 ** max(0, math.floor(math.log10(value)) - 1)
    return int(math.ceil(value / step) * step)


def load_rows(path: Path) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            grouped.setdefault(row["circuit"], []).append(
                (int(row["qubits"]), float(row["speedup_vs_pennylane_qnode"]))
            )
    return {name: sorted(values) for name, values in grouped.items()}


def render_svg(circuit: str, values: list[tuple[int, float]]) -> str:
    left, right = MARGIN["left"], WIDTH - MARGIN["right"]
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    x_min, x_max = min(x for x, _ in values), max(x for x, _ in values)
    y_max = nice_upper_bound(max(y for _, y in values) * 1.1)
    x_scale = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    y_scale = lambda value: bottom - value / y_max * (bottom - top)
    color = COLORS.get(circuit, "#176B87")
    name = html.escape(DISPLAY_NAMES.get(circuit, circuit))
    points = " ".join(f"{x_scale(x):.1f},{y_scale(y):.1f}" for x, y in values)
    peak_x, peak_y = max(values, key=lambda pair: pair[1])

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#252525;letter-spacing:0}.tick{font-size:14px;fill:#555}.label{font-size:17px;font-weight:600}.title{font-size:23px;font-weight:700}.subtitle{font-size:14px;fill:#666}.value{font-size:13px;font-weight:600}</style>',
        f'<text class="title" x="{left}" y="32">{name}: SAD Speedup over PennyLane</text>',
        f'<text class="subtitle" x="{left}" y="54">End-to-end QNode time / SAD time (higher is better)</text>',
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

    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777" stroke-width="1.2"/>',
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
    )
    for x_value, y_value in values:
        elements.append(f'<circle cx="{x_scale(x_value):.1f}" cy="{y_scale(y_value):.1f}" r="5" fill="#fff" stroke="{color}" stroke-width="3"/>')

    label_x = x_scale(peak_x)
    label_y = y_scale(peak_y) - 14
    elements.append(f'<text class="value" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="{color}">Peak {peak_y:.1f}x</text>')
    elements.append(f'<text class="label" x="{(left + right) / 2:.1f}" y="{HEIGHT - 22}" text-anchor="middle">Number of qubits</text>')
    elements.append(f'<text class="label" transform="translate(25 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Speedup over PennyLane QNode</text>')
    elements.append('</svg>')
    return "\n".join(elements) + "\n"


def main() -> None:
    args = parse_args()
    grouped = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuits = (args.circuit,) if args.circuit else (
        "equivariant-qnn", "mera", "data-reuploading", "qaoa-ns"
    )
    for circuit in circuits:
        if circuit not in grouped:
            # The historical merged CSV contains the original three circuits;
            # callers may provide a separate CSV for qaoa-ns.
            if circuit == "qaoa-ns":
                continue
            raise ValueError(f"Missing circuit in input CSV: {circuit}")
        output = args.output_dir / f"{circuit}_sad_vs_pennylane_speedup.svg"
        output.write_text(render_svg(circuit, grouped[circuit]), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
