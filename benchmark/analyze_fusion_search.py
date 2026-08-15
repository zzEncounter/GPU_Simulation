"""Aggregate correctness-aware circuit fusion search results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark" / "results" / "fusion_search_raw.csv"
DEFAULT_SUMMARY = ROOT / "benchmark" / "results" / "fusion_search_summary.csv"
NEAR_TIE = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(
                row[field]
                for field in (
                    "circuit",
                    "qubits",
                    "layers",
                    "variant",
                    "execution_mode",
                    "shape",
                    "library",
                    "compile_flags",
                )
            )
            grouped[key].append(row)
    aggregated: list[dict[str, object]] = []
    for key, rows in grouped.items():
        medians = {
            field: statistics.median(float(row[field]) for row in rows)
            for field in ("forward_ms", "hamiltonian_ms", "backward_ms", "total_ms")
        }
        aggregated.append(
            {
                "circuit": key[0],
                "qubits": int(key[1]),
                "layers": int(key[2]),
                "variant": key[3],
                "execution_mode": key[4],
                "shape": key[5],
                **medians,
                "samples": len(rows),
                "correct": int(all(row["correct"] == "1" for row in rows)),
                "energy_abs_error": max(float(row["energy_abs_error"]) for row in rows),
                "gradient_max_abs_error": max(
                    float(row["gradient_max_abs_error"]) for row in rows
                ),
                "library": key[6],
                "compile_flags": key[7],
            }
        )
    scenarios: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in aggregated:
        scenarios[row["circuit"], row["qubits"]].append(row)
    ranked_rows: list[dict[str, object]] = []
    for candidates in scenarios.values():
        forward_best = min(float(row["forward_ms"]) for row in candidates)
        backward_best = min(float(row["backward_ms"]) for row in candidates)
        total_best = min(float(row["total_ms"]) for row in candidates)
        for row in candidates:
            ranked_rows.append(
                {
                    **row,
                    "forward_relative": float(row["forward_ms"]) / forward_best,
                    "backward_relative": float(row["backward_ms"]) / backward_best,
                    "total_relative": float(row["total_ms"]) / total_best,
                    "forward_near_tie": int(
                        float(row["forward_ms"]) <= forward_best * (1 + NEAR_TIE)
                    ),
                    "backward_near_tie": int(
                        float(row["backward_ms"]) <= backward_best * (1 + NEAR_TIE)
                    ),
                    "total_near_tie": int(
                        float(row["total_ms"]) <= total_best * (1 + NEAR_TIE)
                    ),
                }
            )
    ranked_rows.sort(
        key=lambda row: (row["circuit"], row["qubits"], row["total_ms"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=ranked_rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ranked_rows)
    print(f"aggregated {len(ranked_rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
