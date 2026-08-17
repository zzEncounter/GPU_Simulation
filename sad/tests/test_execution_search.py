from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from search_execution_strategies import (  # noqa: E402
    Phase,
    Variant,
    candidate_schedules,
    k_best_schedules,
    minimum_phase_count,
    phase_options,
    all_variants,
    production_variants,
    _measure,
    _position_jobs,
    _refine_schedule_jobs,
    _select_scenario_survivors,
    _shape_jobs,
    position_starts,
)


def test_tile_12_is_opened_only_by_chunked_mailbox():
    tile_12 = [variant for variant in all_variants() if variant.tile_bits == 12]
    assert tile_12
    assert all(variant.mailbox_chunks >= 2 for variant in tile_12)


def test_full_shape_space_reaches_every_legal_mailbox_divisor():
    chunks_by_register_bits = {
        register_bits: {
            variant.mailbox_chunks
            for variant in all_variants()
            if variant.threads > 32
            and variant.register_bits == register_bits
            and variant.tile_bits <= 12
        }
        for register_bits in range(2, 7)
    }
    assert chunks_by_register_bits[2] == {1, 2, 4}
    assert chunks_by_register_bits[6] == {2, 4, 8, 16, 32, 64}


def test_production_shape_screen_covers_small_tiles_without_mailbox_cross_product():
    variants = production_variants()
    assert {variant.tile_bits for variant in variants} == {7, 8, 9, 10, 11}
    assert all(variant.mailbox_chunks == 1 for variant in variants)
    assert Variant(32, 2) in variants
    assert Variant(128, 2) in variants


@pytest.mark.parametrize(
    ("family", "expected"),
    [("compact", 3), ("fixed", 4), ("pairs", 5)],
)
def test_minimum_phase_count(family, expected):
    assert minimum_phase_count(24, Variant(64, 4), family) == expected


def test_fixed_and_pair_options_reserve_continuity_slots():
    variant = Variant(128, 3)
    fixed = phase_options(variant, "fixed", 1)
    pairs = phase_options(variant, "pairs", 1)
    assert fixed and all(phase.lane == 0 for phase in fixed)
    assert pairs and all(phase.lane == 0 and phase.warp <= 1 for phase in pairs)


def test_fixed_first_phase_consumes_reserved_physical_targets():
    schedules = k_best_schedules(
        12, Variant(128, 2), "pairs", phase_count=2, keep=32
    )
    assert schedules
    assert all(schedule.phases[0].targets >= 6 for schedule in schedules)


@pytest.mark.parametrize("family", ["compact", "fixed", "pairs"])
def test_k_best_schedules_are_valid_and_stably_sorted(family):
    variant = Variant(128, 3, 2)
    count = minimum_phase_count(24, variant, family)
    schedules = k_best_schedules(
        24, variant, family, phase_count=count, keep=8
    )
    assert schedules
    assert list(schedules) == sorted(
        schedules, key=lambda item: (item.predicted_cost, item.candidate)
    )
    for schedule in schedules:
        assert sum(phase.targets for phase in schedule.phases) == 24
        if family in {"fixed", "pairs"}:
            assert schedule.phases[0].targets >= 5 + int(family == "pairs")


def test_dynamic_program_visits_class_mix_not_only_packed_prefixes():
    variant = Variant(128, 3)
    register_favoring = lambda phase, _index: (
        phase.lane + 0.01 * phase.register + 10 * phase.warp
    )
    schedule = k_best_schedules(
        7,
        variant,
        "compact",
        phase_count=2,
        keep=1,
        phase_cost=register_favoring,
    )[0]
    assert schedule.phases == (Phase(0, 3, 0), Phase(0, 3, 1)) or sum(
        phase.register for phase in schedule.phases
    ) == 6


def test_candidate_schedules_cover_minimum_and_extra_phase_counts():
    schedules = candidate_schedules(
        20,
        Variant(64, 4),
        per_family=2,
        extra_phases=1,
    )
    compact_counts = {
        len(schedule.phases)
        for schedule in schedules
        if schedule.family == "compact"
    }
    assert compact_counts == {2, 3}


def test_measure_skips_only_known_cuda_launch_resource_failures(
    monkeypatch, tmp_path
):
    failure = subprocess.CompletedProcess(
        [], 134, "", "too many resources requested for launch"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: failure)
    assert _measure(tmp_path / "micro", 26, "rx", "full", "backward") == {}


def test_schedule_refinement_keeps_top_and_relative_frontier(tmp_path):
    variant = Variant(64, 4)
    names = (
        "L5R4W1-L5R4W1-L2R2W0",
        "L5R4W1-L5R4W1-L1R3W0",
        "L5R4W1-L5R4W1-L0R4W0",
    )
    jobs = [
        (
            variant,
            24,
            "rx",
            "forward",
            "compact",
            f"plan:compact:{name}",
            None,
        )
        for name in names
    ]
    path = tmp_path / "raw.csv"
    path.write_text(
        "stage,variant,gate,direction,qubits,family,candidate,average_ms\n"
        + "".join(
            f"schedule,{variant.name},rx,forward,24,compact,{name},{value}\n"
            for name, value in zip(names, (10.0, 10.4, 12.0), strict=True)
        ),
        encoding="utf-8",
    )
    refined = _refine_schedule_jobs(
        path,
        jobs,
        minimum_per_scenario=1,
        relative_limit=1.05,
    )
    assert refined == tuple(jobs[:2])


def test_scenario_survivors_ignore_variants_outside_current_sparse_run(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "stage,variant,gate,direction,qubits,family,candidate,average_ms\n"
        "shape,t32r2m1,rx,forward,12,compact,full,1.0\n"
        "shape,t128r4m1,rx,forward,12,compact,full,0.5\n",
        encoding="utf-8",
    )
    selected = _select_scenario_survivors(
        path, (Variant(32, 2),), count=1
    )
    assert selected == {("rx", "forward", 12): (Variant(32, 2),)}


def test_small_qubit_jobs_prune_impossible_continuity_families():
    jobs = tuple(_shape_jobs((Variant(64, 2),), (4, 5, 6)))
    families_by_qubits = {
        qubits: {job[4] for job in jobs if job[1] == qubits}
        for qubits in (4, 5, 6)
    }
    assert families_by_qubits == {
        4: {"compact"},
        5: {"compact", "fixed"},
        6: {"compact", "fixed", "pairs"},
    }


def test_position_starts_follow_phase_stride_and_include_highest_window():
    assert position_starts(24, 10, 0) == (0, 10, 14)
    assert position_starts(24, 5, 5) == (5, 10, 15, 19)
    assert position_starts(24, 4, 6) == (6, 10, 14, 18, 20)


def test_position_jobs_separate_direction_position_and_continuity():
    survivors = {
        ("rx", "forward", 24): (Variant(64, 4),),
        ("ry", "backward", 24): (Variant(32, 2),),
    }
    jobs = tuple(_position_jobs(survivors))
    rx = [job for job in jobs if job[2:4] == ("rx", "forward")]
    ry = [job for job in jobs if job[2:4] == ("ry", "backward")]
    assert {job[4] for job in rx} == {"compact", "fixed", "pairs"}
    assert {job[4] for job in ry} == {"compact", "fixed"}
    assert all(job[6] is None for job in jobs)
    assert any(job[5] == "range:14" for job in rx)
    assert any(job[5] == "fixed:19" for job in rx)
    assert any(job[5] == "pairs:20" for job in rx)
