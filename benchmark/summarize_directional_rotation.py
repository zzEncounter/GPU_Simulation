"""Summarize position scans and screen heterogeneous RX/RY phase schedules."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark" / "results" / "execution_search_final.csv"
DEFAULT_POSITION_OUTPUT = (
    ROOT / "benchmark" / "results" / "rotation_position_summary.csv"
)
DEFAULT_GROUP_OUTPUT = (
    ROOT / "benchmark" / "results" / "rotation_position_group_summary.csv"
)
DEFAULT_SCHEDULE_OUTPUT = (
    ROOT / "benchmark" / "results" / "heterogeneous_phase_candidates.csv"
)

POSITION_FIELDS = (
    "gate",
    "direction",
    "qubits",
    "variant",
    "threads",
    "register_bits",
    "tile_bits",
    "family",
    "first_qubit",
    "phase_gate_count",
    "samples",
    "median_ms",
    "median_ms_per_gate",
    "mad_ms",
    "relative_mad",
    "active_cta_per_sm",
    "registers_per_thread",
    "mailbox_bytes",
    "mailbox_chunks",
)

GROUP_FIELDS = (
    "gate",
    "direction",
    "qubits",
    "variant",
    "family",
    "positions",
    "min_median_ms",
    "max_median_ms",
    "position_spread_percent",
    "best_first_qubit",
    "worst_first_qubit",
    "median_relative_mad_percent",
)

SCHEDULE_FIELDS = (
    "gate",
    "direction",
    "qubits",
    "rank",
    "heterogeneous",
    "phase_count",
    "predicted_ms",
    "predicted_ms_per_gate",
    "best_measured_uniform_ms",
    "predicted_speedup_vs_best_uniform",
    "phase_division",
    "phase_parameters",
)


def _variant_bits(row: dict[str, str]) -> int:
    amplitudes = int(row["register_amplitudes"])
    return amplitudes.bit_length() - 1


def _median(value_rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in value_rows)


def summarize_positions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    samples: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["stage"] != "position":
            continue
        key = tuple(
            row[field]
            for field in (
                "gate",
                "direction",
                "qubits",
                "variant",
                "family",
                "candidate",
            )
        )
        samples[key].append(row)

    result: list[dict[str, object]] = []
    for key, value_rows in sorted(samples.items()):
        gate, direction, qubits, variant, family, candidate = key
        if len(value_rows) < 2 or ":" not in candidate:
            continue
        median_ms = _median(value_rows, "average_ms")
        mad_ms = statistics.median(
            abs(float(row["average_ms"]) - median_ms) for row in value_rows
        )
        representative = value_rows[0]
        result.append(
            {
                "gate": gate,
                "direction": direction,
                "qubits": int(qubits),
                "variant": variant,
                "threads": int(representative["threads"]),
                "register_bits": _variant_bits(representative),
                "tile_bits": int(representative["tile_bits"]),
                "family": family,
                "first_qubit": int(candidate.split(":", 1)[1]),
                "phase_gate_count": int(representative["gate_count"]),
                "samples": len(value_rows),
                "median_ms": median_ms,
                "median_ms_per_gate": _median(value_rows, "ms_per_gate"),
                "mad_ms": mad_ms,
                "relative_mad": mad_ms / median_ms,
                "active_cta_per_sm": int(representative["active_cta_per_sm"]),
                "registers_per_thread": int(representative["registers_per_thread"]),
                "mailbox_bytes": int(representative["mailbox_bytes"]),
                "mailbox_chunks": int(representative["mailbox_chunks"]),
            }
        )
    return result


def summarize_groups(
    position_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in position_rows:
        key = tuple(row[field] for field in ("gate", "direction", "qubits", "variant", "family"))
        groups[key].append(row)
    result: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda row: int(row["first_qubit"]))
        best = min(ordered, key=lambda row: float(row["median_ms"]))
        worst = max(ordered, key=lambda row: float(row["median_ms"]))
        result.append(
            {
                "gate": key[0],
                "direction": key[1],
                "qubits": key[2],
                "variant": key[3],
                "family": key[4],
                "positions": ":".join(str(row["first_qubit"]) for row in ordered),
                "min_median_ms": best["median_ms"],
                "max_median_ms": worst["median_ms"],
                "position_spread_percent": 100
                * (float(worst["median_ms"]) / float(best["median_ms"]) - 1),
                "best_first_qubit": best["first_qubit"],
                "worst_first_qubit": worst["first_qubit"],
                "median_relative_mad_percent": 100
                * statistics.median(float(row["relative_mad"]) for row in ordered),
            }
        )
    return result


@dataclass(frozen=True)
class Edge:
    first: int
    count: int
    cost: float
    variant: str
    family: str
    threads: int
    register_bits: int
    tile_bits: int

    @property
    def end(self) -> int:
        return self.first + self.count

    @property
    def parameter(self) -> str:
        return (
            f"q{self.first}:{self.end - 1}={self.variant}/"
            f"{self.family}/t{self.threads}r{self.register_bits}L{self.tile_bits}"
        )


def _uniform_medians(
    rows: list[dict[str, str]], qubits: set[int]
) -> dict[tuple[str, str, int], float]:
    samples: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["stage"] != "shape" or int(row["qubits"]) not in qubits:
            continue
        key = (
            row["gate"],
            row["direction"],
            int(row["qubits"]),
            row["variant"],
            row["family"],
        )
        samples[key].append(float(row["average_ms"]))
    result: dict[tuple[str, str, int], float] = {}
    for key, values in samples.items():
        scenario = key[:3]
        median = statistics.median(values)
        result[scenario] = min(result.get(scenario, math.inf), median)
    return result


def screen_schedules(
    raw_rows: list[dict[str, str]],
    position_rows: list[dict[str, object]],
    *,
    keep: int = 10,
) -> list[dict[str, object]]:
    by_scenario: dict[tuple[str, str, int], list[Edge]] = defaultdict(list)
    for row in position_rows:
        first = int(row["first_qubit"])
        count = int(row["phase_gate_count"])
        scenario = (str(row["gate"]), str(row["direction"]), int(row["qubits"]))
        if first + count > scenario[2]:
            continue
        by_scenario[scenario].append(
            Edge(
                first=first,
                count=count,
                cost=float(row["median_ms"]),
                variant=str(row["variant"]),
                family=str(row["family"]),
                threads=int(row["threads"]),
                register_bits=int(row["register_bits"]),
                tile_bits=int(row["tile_bits"]),
            )
        )
    uniform = _uniform_medians(raw_rows, {scenario[2] for scenario in by_scenario})
    result: list[dict[str, object]] = []
    for scenario, edges in sorted(by_scenario.items()):
        qubits = scenario[2]
        outgoing: dict[int, list[Edge]] = defaultdict(list)
        for edge in edges:
            outgoing[edge.first].append(edge)
        states: dict[int, list[tuple[float, tuple[Edge, ...]]]] = {0: [(0.0, ())]}
        for first in range(qubits):
            histories = states.get(first, ())
            if not histories:
                continue
            for cost, path in histories:
                for edge in outgoing.get(first, ()):
                    bucket = states.setdefault(edge.end, [])
                    bucket.append((cost + edge.cost, path + (edge,)))
                    bucket.sort(key=lambda item: (item[0], tuple(e.parameter for e in item[1])))
                    del bucket[64:]
        candidates = states.get(qubits, ())
        ranked = sorted(candidates, key=lambda item: item[0])[:keep]
        for rank, (cost, path) in enumerate(ranked, 1):
            parameter_classes = {(edge.variant, edge.family) for edge in path}
            measured_uniform = uniform.get(scenario, math.nan)
            result.append(
                {
                    "gate": scenario[0],
                    "direction": scenario[1],
                    "qubits": qubits,
                    "rank": rank,
                    "heterogeneous": int(len(parameter_classes) > 1),
                    "phase_count": len(path),
                    "predicted_ms": cost,
                    "predicted_ms_per_gate": cost / qubits,
                    "best_measured_uniform_ms": measured_uniform,
                    "predicted_speedup_vs_best_uniform": (
                        measured_uniform / cost if math.isfinite(measured_uniform) else math.nan
                    ),
                    "phase_division": "+".join(str(edge.count) for edge in path),
                    "phase_parameters": ";".join(edge.parameter for edge in path),
                }
            )
    return result


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--position-output", type=Path, default=DEFAULT_POSITION_OUTPUT)
    parser.add_argument("--group-output", type=Path, default=DEFAULT_GROUP_OUTPUT)
    parser.add_argument("--schedule-output", type=Path, default=DEFAULT_SCHEDULE_OUTPUT)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    positions = summarize_positions(raw_rows)
    groups = summarize_groups(positions)
    schedules = screen_schedules(raw_rows, positions)
    _write(args.position_output, POSITION_FIELDS, positions)
    _write(args.group_output, GROUP_FIELDS, groups)
    _write(args.schedule_output, SCHEDULE_FIELDS, schedules)
    print(
        f"wrote {len(positions)} position rows, {len(groups)} groups, "
        f"and {len(schedules)} heterogeneous schedule candidates"
    )


if __name__ == "__main__":
    main()
