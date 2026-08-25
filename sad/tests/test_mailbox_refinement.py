from __future__ import annotations

import csv
import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from benchmark_phase_mailbox_refinement import (  # noqa: E402
    FixedSchedule,
    _fixed_schedules,
    mailbox_options,
    normalize_mailboxes,
    set_phase_mailbox,
    single_phase_candidates,
)


def test_mailbox_refinement_preserves_fixed_phase_geometry():
    source = (
        "t64r4m2/compact/10;"
        "t32r4m1/compact/9;"
        "t128r3m8/fixed/5"
    )
    base = normalize_mailboxes(source)
    assert base == (
        "t64r4m1/compact/10;"
        "t32r4m1/compact/9;"
        "t128r3m1/fixed/5"
    )
    assert mailbox_options("t64r4m1") == (1, 2, 4, 8, 16)
    assert mailbox_options("t32r4m1") == (1,)
    assert mailbox_options("t128r3m1") == (1, 2, 4, 8)
    assert set_phase_mailbox(base, 2, 4).endswith("t128r3m4/fixed/5")

    candidates = single_phase_candidates(FixedSchedule("winner", source, base))
    assert len(candidates) == 7
    assert {candidate[0] for candidate in candidates} == {
        "phase-0-m2",
        "phase-0-m4",
        "phase-0-m8",
        "phase-0-m16",
        "phase-2-m2",
        "phase-2-m4",
        "phase-2-m8",
    }
    for _, phase_index, _, schedule in candidates:
        source_tokens = base.split(";")
        candidate_tokens = schedule.split(";")
        assert len(source_tokens) == len(candidate_tokens)
        assert sum(a != b for a, b in zip(source_tokens, candidate_tokens)) == 1
        assert source_tokens[phase_index] != candidate_tokens[phase_index]


def test_fixed_schedule_uses_measured_complete_layer_winner(tmp_path):
    path = tmp_path / "heterogeneous.csv"
    fields = (
        "gate",
        "direction",
        "qubits",
        "candidate",
        "schedule",
        "average_ms",
        "correct",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for value in (10.0, 9.8, 10.2):
            writer.writerow(
                {
                    "gate": "rx",
                    "direction": "forward",
                    "qubits": 24,
                    "candidate": "uniform",
                    "schedule": "t64r4m1/compact/10;t64r4m1/compact/10;t64r4m1/compact/4",
                    "average_ms": value,
                    "correct": 1,
                }
            )
        for value in (9.0, 9.1, 8.9):
            writer.writerow(
                {
                    "gate": "rx",
                    "direction": "forward",
                    "qubits": 24,
                    "candidate": "heterogeneous-1",
                    "schedule": "t64r4m2/compact/10;t32r4m1/compact/9;t32r4m1/compact/5",
                    "average_ms": value,
                    "correct": 1,
                }
            )
    selected = _fixed_schedules(path)[("rx", "forward", 24)]
    assert selected.source_candidate == "heterogeneous-1"
    assert selected.source_schedule.startswith("t64r4m2")
    assert selected.base_schedule.startswith("t64r4m1")

