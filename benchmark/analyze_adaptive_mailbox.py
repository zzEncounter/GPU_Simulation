"""Aggregate matched full/adaptive per-phase mailbox measurements."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmark" / "results" / "adaptive_mailbox_paired_raw.csv"
OUTPUT = ROOT / "benchmark" / "results" / "adaptive_mailbox_summary.csv"
IDENTITY = (
    "gate", "direction", "qubits", "variant", "family", "candidate", "layout"
)


def aggregate(path: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[tuple(row[field] for field in IDENTITY)][row["mode"]].append(row)
    result: list[dict[str, object]] = []
    for key, modes in groups.items():
        if not {"full", "adaptive"} <= modes.keys():
            continue
        full = statistics.median(
            float(row["average_ms"]) for row in modes["full"]
        )
        adaptive = statistics.median(
            float(row["average_ms"]) for row in modes["adaptive"]
        )
        full_by_repetition = {
            row["repetition"]: float(row["average_ms"])
            for row in modes["full"]
        }
        adaptive_by_repetition = {
            row["repetition"]: float(row["average_ms"])
            for row in modes["adaptive"]
        }
        paired_repetitions = sorted(
            full_by_repetition.keys() & adaptive_by_repetition.keys(), key=int
        )
        paired_ratios = [
            adaptive_by_repetition[repetition]
            / full_by_repetition[repetition]
            for repetition in paired_repetitions
        ]
        paired_relative = statistics.median(paired_ratios)
        paired_mad = statistics.median(
            abs(value - paired_relative) for value in paired_ratios
        )
        exemplar = modes["adaptive"][0]
        warp_targets = tuple(
            int(value) for value in exemplar["phase_warp_targets"].split(":")
        )
        all_warp_mask = (1 << len(warp_targets)) - 1
        effective = (
            int(exemplar["mailbox_bytes"]) > 0
            and int(exemplar["warp_phase_mask"]) != all_warp_mask
        )
        result.append(
            {
                **dict(zip(IDENTITY, key, strict=True)),
                "phase_lane_targets": exemplar["phase_lane_targets"],
                "phase_register_targets": exemplar["phase_register_targets"],
                "phase_warp_targets": exemplar["phase_warp_targets"],
                "warp_phase_mask": exemplar["warp_phase_mask"],
                "mailbox_bytes": int(exemplar["mailbox_bytes"]),
                "full_active_cta_per_sm": int(exemplar["active_cta_per_sm"]),
                "no_mailbox_active_cta_per_sm": int(
                    exemplar["no_mailbox_active_cta_per_sm"]
                ),
                "samples_per_mode": len(paired_repetitions),
                "full_ms": full,
                "adaptive_ms": adaptive,
                "ratio_of_medians": adaptive / full,
                # Adjacent, order-alternating full/adaptive pairs are the
                # primary effect estimator; this is more robust to the GPU's
                # observed clock-state bimodality than a ratio of two medians.
                "adaptive_relative": paired_relative,
                "paired_ratio_mad": paired_mad,
                # A zero-mailbox one-warp kernel, or a schedule whose every
                # phase has W>0, executes identical code in both modes.  Keep
                # those rows as negative controls, never as optimization wins.
                "adaptive_changes_execution": int(effective),
                "near_tie": int(paired_relative <= 1.02),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["gate"], row["direction"], int(row["qubits"]),
            float(row["adaptive_relative"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = aggregate(args.input)
    if not rows:
        parser.error(f"no paired measurements in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"aggregated {len(rows)} matched candidates to {args.output}")


if __name__ == "__main__":
    main()
