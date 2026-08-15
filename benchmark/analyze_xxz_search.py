"""Aggregate XXZ pair-phase search and emit a compact ranked summary."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmark" / "results" / "xxz_search_raw.csv"
OUTPUT = ROOT / "benchmark" / "results" / "xxz_search_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(
                row[field]
                for field in (
                    "stage", "component", "direction", "qubits", "parity", "variant",
                    "candidate", "threads", "register_amplitudes", "tile_bits",
                    "phase_count", "static_shared_bytes", "dynamic_shared_bytes",
                    "registers_per_thread", "active_cta_per_sm",
                    "local_bytes_per_thread", "multiprocessors",
                )
            )
            groups[key].append(row)
    aggregates: list[dict[str, object]] = []
    for key, rows in groups.items():
        aggregates.append(
            {
                "stage": key[0], "component": key[1], "direction": key[2],
                "qubits": int(key[3]), "parity": int(key[4]),
                "variant": key[5], "candidate": key[6],
                "threads": int(key[7]), "register_amplitudes": int(key[8]),
                "tile_bits": int(key[9]), "phase_count": int(key[10]),
                "static_shared_bytes": int(key[11]),
                "dynamic_shared_bytes": int(key[12]),
                "registers_per_thread": int(key[13]),
                "active_cta_per_sm": int(key[14]),
                "local_bytes_per_thread": int(key[15]),
                "multiprocessors": int(key[16]),
                "samples": len(rows),
                "median_ms": statistics.median(
                    float(row["average_ms"]) for row in rows
                ),
            }
        )
    scenarios: dict[
        tuple[str, str, str, int, int], list[dict[str, object]]
    ] = defaultdict(list)
    for row in aggregates:
        scenarios[
            row["stage"], row["component"], row["direction"],
            row["qubits"], row["parity"]
        ].append(row)
    ranked: list[dict[str, object]] = []
    for candidates in scenarios.values():
        candidates.sort(key=lambda row: float(row["median_ms"]))
        best = float(candidates[0]["median_ms"])
        for rank, row in enumerate(candidates, 1):
            ranked.append(
                {
                    **row,
                    "rank": rank,
                    "relative_to_best": float(row["median_ms"]) / best,
                    "near_tie": int(float(row["median_ms"]) <= 1.02 * best),
                }
            )
    ranked.sort(
        key=lambda row: (
            row["stage"], row["component"], row["direction"],
            row["qubits"], row["parity"], row["rank"]
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranked[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranked)
    print(f"aggregated {len(ranked)} candidates to {args.output}")


if __name__ == "__main__":
    main()
