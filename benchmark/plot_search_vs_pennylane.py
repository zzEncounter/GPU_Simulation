"""Generate per-circuit linear-axis SAD-vs-PennyLane speedup SVG plots."""
from __future__ import annotations

import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "search" / "figures"
QUbits = tuple(range(4, 29, 2))
CIRCUITS = (
    "ra-hea", "su2-hea", "rzz-hea", "qaoa", "qaoa-ns",
    "equivariant-qnn", "data-reuploading", "xxz-hva", "mera",
)
COLORS = {"best": "#176b87", "average": "#c44a35", "median": "#5b7f36", "worst": "#7a4eab"}
LABELS = {"best": "SAD best", "average": "SAD average", "median": "SAD median", "worst": "SAD worst"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def baseline_pennylane() -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for name in (
        "native_baseline_comparison.csv",
        "native_baseline_comparison_merged_triple.csv",
        "native_baseline_comparison_qaoans.csv",
        "native_baseline_comparison_mera.csv",
    ):
        for row in read_csv(ROOT / "benchmark" / "results" / name):
            if row.get("pennylane_qnode_time_median_s"):
                result[row["circuit"], int(row["qubits"])] = float(row["pennylane_qnode_time_median_s"]) * 1000
    return result


def regular_stats(circuit: str) -> dict[int, list[float]]:
    from sys import path
    path.insert(0, str(ROOT / "benchmark"))
    from search_joint_phase_parameters import read_result_rows
    rows = [r for r in read_result_rows(ROOT / "benchmark" / "results" / "joint_phase" / f"{circuit}.csv")
            if r.get("status") == "ok" and r.get("median_ms")]
    result = {}
    for q in QUbits:
        values = [float(r["median_ms"]) for r in rows if int(r["qubits"]) == q]
        if values:
            result[q] = [min(values), statistics.fmean(values), statistics.median(values), max(values)]
    return result


def mera_stats() -> dict[int, list[float]]:
    rows = [r for r in read_csv(ROOT / "benchmark" / "results" / "mera_parameter_search.csv") if r["status"] == "ok"]
    result = {}
    for q in QUbits:
        values = [float(r["median_ms"]) for r in rows if int(r["qubits"]) == q]
        if values:
            result[q] = [min(values), statistics.fmean(values), statistics.median(values), max(values)]
    return result


def xxz_stats() -> dict[int, list[float]]:
    rows = [r for r in read_csv(ROOT / "benchmark" / "results" / "xxz_search_raw.csv")
            if r["stage"] == "partition" and r["component"] == "xx+yy+zz"]
    # The historical file contains three repetitions for each of the four
    # direction/parity matching scenarios. Average repetitions first, then
    # sum the four scenarios to obtain one candidate runtime.
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"], row["candidate"], row["qubits"], row["direction"], row["parity"]].append(float(row["average_ms"]))
    result = {}
    for q in QUbits:
        candidates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for (variant, candidate, qubits, _direction, _parity), repetitions in grouped.items():
            if int(qubits) == q:
                candidates[variant, candidate, qubits].append(statistics.fmean(repetitions))
        values = [sum(scenarios) for scenarios in candidates.values() if len(scenarios) == 4]
        if values:
            result[q] = [min(values), statistics.fmean(values), statistics.median(values), max(values)]
    return result


def svg_plot(circuit: str, ratios: dict[str, dict[int, float]], ymax: float) -> str:
    width, height = 1120, 700
    left, right, top, bottom = 90, 35, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    qs = sorted({q for values in ratios.values() for q in values})
    x = lambda q: left + (q - 4) / 24 * plot_w
    y = lambda value: top + plot_h - value / ymax * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{html.escape(circuit)}: SAD speedup vs PennyLane</text>']
    tick_step = max(1, round(ymax / 5))
    ticks = list(range(0, int(ymax), tick_step))
    if not ticks or ticks[-1] != int(ymax):
        ticks.append(int(ymax))
    for tick in ticks:
        yy = y(tick)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d9dee5"/>',
                  f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}x</text>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
              f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#222"/>',
              f'<text x="{width/2:.0f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="14">qubits</text>',
              f'<text x="20" y="{height/2:.0f}" transform="rotate(-90 20 {height/2:.0f})" text-anchor="middle" font-family="Arial" font-size="14">PennyLane time / SAD time (linear scale)</text>']
    for q in qs:
        xx = x(q)
        parts += [f'<line x1="{xx:.1f}" y1="{top+plot_h}" x2="{xx:.1f}" y2="{top+plot_h+6}" stroke="#222"/>',
                  f'<text x="{xx:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{q}</text>']
    for key in ("best", "average", "median", "worst"):
        points = " ".join(f'{x(q):.1f},{y(value):.1f}' for q, value in sorted(ratios[key].items()))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[key]}" stroke-width="3"/>')
        for q, value in sorted(ratios[key].items()):
            parts.append(f'<circle cx="{x(q):.1f}" cy="{y(value):.1f}" r="4" fill="{COLORS[key]}"/>')
    legend_x, legend_y = left + 20, top + 20
    for i, key in enumerate(("best", "average", "median", "worst")):
        lx = legend_x + i * 190
        parts += [f'<line x1="{lx}" y1="{legend_y}" x2="{lx+28}" y2="{legend_y}" stroke="{COLORS[key]}" stroke-width="3"/>',
                  f'<text x="{lx+36}" y="{legend_y+5}" font-family="Arial" font-size="13">{LABELS[key]}</text>']
    if circuit == "xxz-hva":
        parts.append(f'<text x="{width-right}" y="{height-25}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">historical matching microbenchmark; q=20,24,26</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pl = baseline_pennylane()
    stats = {c: (xxz_stats() if c == "xxz-hva" else mera_stats() if c == "mera" else regular_stats(c)) for c in CIRCUITS}
    circuit_ymax: dict[str, float] = {}
    for circuit in CIRCUITS:
        ratios = {"best": {}, "average": {}, "median": {}, "worst": {}}
        for q, values in stats[circuit].items():
            baseline = pl[circuit, q]
            for key, value in zip(("best", "average", "median", "worst"), (values[0], values[1], values[2], values[3])):
                ratios[key][q] = baseline / value
        stats[circuit] = ratios
        values = [value for series in ratios.values() for value in series.values()]
        if not values:
            raise ValueError(f"no speedup data available for {circuit}")
        # Each circuit gets its own linear scale, sized to its current data.
        circuit_ymax[circuit] = max(values) * 1.10
    for circuit in CIRCUITS:
        ymax = circuit_ymax[circuit]
        (OUT / f"{circuit}_speedup_vs_pennylane.svg").write_text(svg_plot(circuit, stats[circuit], ymax), encoding="utf-8")
    ranges = ", ".join(f"{circuit}=0-{circuit_ymax[circuit]:.2f}x" for circuit in CIRCUITS)
    print(f"wrote {len(CIRCUITS)} SVG plots to {OUT} (per-circuit linear ranges: {ranges})")


if __name__ == "__main__":
    main()
