from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from search_xxz_strategies import (  # noqa: E402
    Variant,
    _existing_shape_samples,
    _parse_components,
    candidate_partitions,
    compositions,
)


def test_compositions_cover_every_bounded_partition():
    assert set(compositions(5, 2, 3)) == {(2, 3), (3, 2)}


def test_xxz_candidates_include_every_minimum_phase_split():
    candidates = candidate_partitions(24, 9)
    minimum = {values for values in candidates if len(values) == 3}
    assert minimum == {(4, 4, 4)}
    assert any(len(values) == 4 for values in candidates)


def test_xxz_component_parser_keeps_compile_time_modes_distinct():
    assert _parse_components("xx+yy+zz,xx,yy") == ("xx+yy+zz", "xx", "yy")


def test_existing_shape_samples_make_resume_selection_reproducible(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "stage,component,direction,qubits,parity,variant,average_ms,iterations\n"
        "shape,xx,forward,20,0,t32r2,1.25,8\n"
        "shape,xx,forward,20,0,t32r2,9.0,7\n",
        encoding="utf-8",
    )
    assert _existing_shape_samples(path, 8) == {
        (Variant(32, 2), "xx", 20, "forward", 0): [1.25]
    }
