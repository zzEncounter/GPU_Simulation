from __future__ import annotations

import csv
import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from analyze_adaptive_mailbox import aggregate  # noqa: E402
from search_adaptive_mailbox import (  # noqa: E402
    FIELDS,
    changes_execution,
    frontier,
    parse_variant,
)


def test_parse_adaptive_variant():
    variant = parse_variant("t128r4m8")
    assert (variant.threads, variant.register_bits, variant.mailbox_chunks) == (
        128, 4, 8
    )


def test_adaptive_aggregation_is_matched_and_uses_medians(tmp_path):
    path = tmp_path / "raw.csv"
    base = {field: "0" for field in FIELDS}
    base.update(
        {
            "gate": "rx", "direction": "forward", "qubits": "24",
            "variant": "t64r4m1", "family": "fixed", "candidate": "x",
            "layout": "plan:fixed:x", "mailbox_bytes": "16384",
            "active_cta_per_sm": "5", "no_mailbox_active_cta_per_sm": "8",
            "phase_warp_targets": "1:0", "warp_phase_mask": "1",
            "phase_lane_targets": "5:5",
            "phase_register_targets": "4:4",
        }
    )
    rows = []
    for mode, samples in (("full", (10, 12, 11)), ("adaptive", (9, 8, 10))):
        for repetition, sample in enumerate(samples):
            rows.append(
                {**base, "mode": mode, "repetition": repetition,
                 "average_ms": sample}
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = aggregate(path)
    assert len(result) == 1
    assert result[0]["full_ms"] == 11
    assert result[0]["adaptive_ms"] == 9
    assert result[0]["ratio_of_medians"] == 9 / 11
    assert result[0]["adaptive_relative"] == 0.9
    assert result[0]["adaptive_changes_execution"] == 1


def test_frontier_rejects_single_sample_screening_rows(tmp_path):
    path = tmp_path / "execution.csv"
    fields = (
        "stage", "variant", "family", "candidate", "gate", "direction",
        "qubits", "layout", "threads", "register_amplitudes", "tile_bits",
        "phase_count", "gate_count", "registers_per_thread",
        "static_shared_bytes", "dynamic_shared_bytes", "active_cta_per_sm",
        "mailbox_bytes", "mailbox_chunks", "local_bytes_per_thread",
        "multiprocessors", "phase_targets", "phase_lane_targets",
        "phase_register_targets", "phase_warp_targets", "average_ms",
    )
    base = {field: "1" for field in fields}
    base.update({
        "stage": "schedule", "variant": "t32r2m1", "family": "compact",
        "gate": "rx", "direction": "forward", "qubits": "20",
        "layout": "plan:x", "threads": "32", "register_amplitudes": "4",
        "tile_bits": "7", "phase_count": "1", "gate_count": "1",
        "phase_targets": "1", "phase_lane_targets": "1",
        "phase_register_targets": "0", "phase_warp_targets": "0",
    })
    rows = [{**base, "candidate": "screen", "average_ms": "0.1"}]
    rows.extend(
        {**base, "candidate": "confirmed", "average_ms": value}
        for value in ("1.0", "1.1", "0.9")
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selected = frontier(path, 1.02)
    assert [row.candidate for row in selected] == ["confirmed"]


def test_adaptive_filter_rejects_identical_execution_controls():
    class Row:
        phase_warp_targets = (0, 0)

    Row.mailbox_bytes = 0
    assert not changes_execution(Row())
    Row.mailbox_bytes = 1024
    assert changes_execution(Row())
    Row.phase_warp_targets = (1, 1)
    assert not changes_execution(Row())
