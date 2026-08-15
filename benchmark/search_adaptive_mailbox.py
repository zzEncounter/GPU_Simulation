"""A/B full versus per-phase adaptive mailbox on the measured frontier."""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from analyze_execution_search import confirmed_rows, load_aggregates
from search_execution_strategies import (
    MICRO_FIELDS,
    ROOT,
    Variant,
    _cached_binary,
    _compile,
)


DEFAULT_INPUT = ROOT / "benchmark" / "results" / "execution_search_exhaustive.csv"
DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "adaptive_mailbox_paired_raw.csv"
)
PREFIX_FIELDS = (
    "gate", "direction", "qubits", "variant", "family", "candidate",
    "layout", "mode", "repetition", "iterations",
    "no_mailbox_active_cta_per_sm", "warp_phase_mask",
)
FIELDS = (
    *PREFIX_FIELDS,
    *(field for field in MICRO_FIELDS if field not in PREFIX_FIELDS),
)


def parse_variant(name: str) -> Variant:
    match = re.fullmatch(r"t(\d+)r(\d+)m(\d+)", name)
    if match is None:
        raise ValueError(f"invalid execution-search variant {name!r}")
    return Variant(*(int(value) for value in match.groups()))


def frontier(
    path: Path, threshold: float, minimum_samples: int = 3
) -> list[object]:
    grouped: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for row in confirmed_rows(load_aggregates(path), minimum_samples):
        if row.stage == "schedule":
            grouped[row.gate, row.direction, row.qubits].append(row)
    result: list[object] = []
    for rows in grouped.values():
        best = min(row.median_ms for row in rows)
        result.extend(row for row in rows if row.median_ms <= best * threshold)
    return result


def changes_execution(row: object) -> bool:
    """Whether omitting W=0 mailboxes changes the generated launch path."""

    return row.mailbox_bytes > 0 and any(
        warp_targets == 0 for warp_targets in row.phase_warp_targets
    )


def measure(
    binary: Path, row: object, adaptive: bool
) -> tuple[dict[str, str], int, int]:
    command = [
        str(binary), str(row.qubits), row.gate,
        row.layout.split(":all", 1)[0], row.direction, "all",
    ]
    if adaptive:
        command.append("adaptive")
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    lines = completed.stdout.strip().splitlines()
    values = lines[-1].split(",")
    if len(values) != len(MICRO_FIELDS):
        raise RuntimeError(completed.stdout)
    no_mailbox_cta = int(values[MICRO_FIELDS.index("active_cta_per_sm")])
    warp_phase_mask = -1
    if adaptive:
        diagnostic = dict(
            item.split("=", 1) for item in lines[-2].split(",")
        )
        no_mailbox_cta = int(diagnostic["no_mailbox_active_cta_per_sm"])
        warp_phase_mask = int(diagnostic["warp_phase_mask"])
    return (
        dict(zip(MICRO_FIELDS, values, strict=True)),
        no_mailbox_cta,
        warp_phase_mask,
    )


def row_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(row[field])
        for field in (
            "gate", "direction", "qubits", "variant", "family", "candidate",
            "mode", "repetition", "iterations",
        )
    )


def completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row_key(row) for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tie-threshold", type=float, default=1.02)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minimum-schedule-samples", type=int, default=3)
    args = parser.parse_args()
    if args.tie_threshold < 1 or min(args.repetitions, args.iterations) < 1:
        parser.error("threshold must be >=1 and counts must be positive")

    selected = [
        row for row in frontier(
        args.input, args.tie_threshold, args.minimum_schedule_samples
        ) if changes_execution(row)
    ]
    if not selected:
        parser.error(f"no schedule frontier in {args.input}")
    variants = {row.variant: parse_variant(row.variant) for row in selected}
    binaries: dict[str, Path] = {}
    for name, variant in sorted(variants.items()):
        binary = _cached_binary(variant, args.iterations)
        _compile(variant, binary, args.iterations)
        binaries[name] = binary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.repetitions):
            ordered = list(selected)
            random.Random(f"sad-adaptive-mailbox-{repetition}").shuffle(ordered)
            for row in ordered:
                # Measure the two modes next to each other so clock state and
                # background load are matched.  Alternate order to remove a
                # systematic warm-cache advantage.
                parity = (
                    repetition + sum(row.candidate.encode("utf-8"))
                ) & 1
                modes = (
                    ("full", "adaptive")
                    if parity == 0 else ("adaptive", "full")
                )
                for mode in modes:
                    prefix: dict[str, object] = {
                        "gate": row.gate,
                        "direction": row.direction,
                        "qubits": row.qubits,
                        "variant": row.variant,
                        "family": row.family,
                        "candidate": row.candidate,
                        "layout": row.layout,
                        "mode": mode,
                        "repetition": repetition,
                        "iterations": args.iterations,
                    }
                    if row_key(prefix) in done:
                        continue
                    measured, no_mailbox_cta, warp_phase_mask = measure(
                        binaries[row.variant], row, mode == "adaptive"
                    )
                    result = {
                        **prefix,
                        "no_mailbox_active_cta_per_sm": no_mailbox_cta,
                        "warp_phase_mask": warp_phase_mask,
                        **measured,
                    }
                    writer.writerow(result)
                    stream.flush()
                    done.add(row_key(result))
                    print(
                        f"{row.gate}/{row.direction[0]} q={row.qubits} "
                        f"{row.variant}/{row.family} {mode} "
                        f"{float(measured['average_ms']):.6f} ms",
                        flush=True,
                    )
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
