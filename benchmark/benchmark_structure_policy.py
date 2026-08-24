"""Paired E2E validation of the production structure/diagonal policy.

Each comparison keeps the production execution shape and runtime phase plans
fixed, changes one structural decision, alternates measurement order, and checks
the complete energy and gradient.  The candidate set is intentionally sparse:
it brackets every production boundary without forming a Cartesian product.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402
from sad_baseline import runner as sad_runner  # noqa: E402


RAW_OUTPUT = ROOT / "benchmark" / "results" / "structure_policy_paired_raw.csv"
SUMMARY_OUTPUT = ROOT / "benchmark" / "results" / "structure_strategy_experiment.csv"
LAYERS = 8
SEED = 42
REPRESENTATIVE_QUBITS = tuple(range(4, 29, 2))
CIRCUIT_IDS = {
    "ra-hea": 0,
    "su2-hea": 1,
    "rzz-hea": 2,
    "qaoa": 3,
    "xxz-hva": 4,
}


@dataclass(frozen=True)
class Comparison:
    category: str
    decision: str
    circuit: str
    qubits: int
    production_choice: str
    candidate_choice: str
    production_flags: tuple[str, ...] = ()
    candidate_flags: tuple[str, ...] = ()
    scope: str = "production_e2e"


RAW_FIELDS = (
    "timestamp_utc",
    "source_fingerprint",
    "repetition",
    "order",
    "category",
    "decision",
    "scope",
    "circuit",
    "qubits",
    "layers",
    "shape",
    "forward_phase_plan",
    "backward_phase_plan",
    "production_choice",
    "candidate_choice",
    "production_flags",
    "candidate_flags",
    "steps",
    "warmup_steps",
    "production_median_ms",
    "candidate_median_ms",
    "candidate_over_production",
    "production_forward_mean_ms",
    "candidate_forward_mean_ms",
    "candidate_over_production_forward",
    "production_hamiltonian_mean_ms",
    "candidate_hamiltonian_mean_ms",
    "candidate_over_production_hamiltonian",
    "production_backward_mean_ms",
    "candidate_backward_mean_ms",
    "candidate_over_production_backward",
    "energy_abs_error",
    "gradient_max_abs_error",
    "correct",
    "production_library",
    "candidate_library",
)

SUMMARY_FIELDS = (
    "experiment_group",
    "hardware",
    "category",
    "decision",
    "scope",
    "circuit",
    "qubits",
    "layers",
    "shape",
    "forward_phase_plan",
    "backward_phase_plan",
    "production_choice",
    "candidate_choice",
    "paired_samples",
    "candidate_over_production",
    "paired_ratio_mad",
    "production_improvement_percent",
    "candidate_over_production_forward",
    "candidate_over_production_hamiltonian",
    "candidate_over_production_backward",
    "verdict_at_2_percent",
    "correct",
    "max_energy_abs_error",
    "max_gradient_abs_error",
    "source_fingerprint",
    "source_raw",
)


def _flag(name: str, value: int) -> tuple[str, ...]:
    return (f"-D{name}={value}",)


def comparisons() -> tuple[Comparison, ...]:
    rows: list[Comparison] = []
    for q in REPRESENTATIVE_QUBITS:
        rows.append(
            Comparison(
                "fusion",
                "ra_real_amplitude_fused_path",
                "ra-hea",
                q,
                "real-fused",
                "complex-auto",
                (),
                _flag("SAD_REAL_AMPLITUDE", 0),
            )
        )

        production_f = (
            ("phased", 2) if q == 10 else ("lookup-fused", 1)
        )[0]
        for candidate, value in (
            item
            for item in (("split", 0), ("lookup-fused", 1), ("phased", 2))
            if item[0] != production_f
        ):
            rows.append(
                Comparison(
                    "fusion",
                    "su2_forward",
                    "su2-hea",
                    q,
                    production_f,
                    candidate,
                    (),
                    _flag("SAD_SU2_FORWARD_STRATEGY", value),
                )
            )
        production_b = (
            ("phased", 2)
            if q == 4
            else ("lookup-fused", 1)
            if q in (6, 16)
            else ("split", 0)
        )[0]
        for candidate, value in (
            item
            for item in (("split", 0), ("lookup-fused", 1), ("phased", 2))
            if item[0] != production_b
        ):
            rows.append(
                Comparison(
                    "fusion",
                    "su2_backward",
                    "su2-hea",
                    q,
                    production_b,
                    candidate,
                    (),
                    _flag("SAD_SU2_BACKWARD_STRATEGY", value),
                )
            )

        rows.append(
            Comparison(
                "fusion",
                "rzz_forward",
                "rzz-hea",
                q,
                "rx-rz-rzz-fused",
                "split",
                (),
                _flag("SAD_RZZ_FORWARD_FUSED", 0),
            )
        )
        production_b = (
            ("combined-diagonal", 1)
            if q <= 14
            else ("rx-rz-rzz-fused", 2)
            if q == 18
            else ("split", 0)
        )[0]
        for candidate, value in (
            item
            for item in (
                ("split", 0),
                ("combined-diagonal", 1),
                ("rx-rz-rzz-fused", 2),
            )
            if item[0] != production_b
        ):
            rows.append(
                Comparison(
                    "fusion",
                    "rzz_backward",
                    "rzz-hea",
                    q,
                    production_b,
                    candidate,
                    (),
                    _flag("SAD_RZZ_BACKWARD_STRATEGY", value),
                )
            )

        rows.append(
            Comparison(
                "fusion",
                "qaoa_forward",
                "qaoa",
                q,
                "cost-rx-fused",
                "split",
                (),
                _flag("SAD_QAOA_FUSE_COST_RX", 0),
            )
        )
        rows.append(
            Comparison(
                "fusion",
                "qaoa_backward",
                "qaoa",
                q,
                "cost-rx-fused",
                "split",
                (),
                _flag("SAD_QAOA_FUSED_BACKWARD", 0),
            )
        )

        production_matching = "separate-matchings" if q == 6 else "cross-matching"
        candidate_matching = "cross-matching" if q == 6 else "separate-matchings"
        candidate_matching_value = 1 if q == 6 else 0
        rows.append(
            Comparison(
                "bond_schedule",
                "xxz_matching",
                "xxz-hva",
                q,
                production_matching,
                candidate_matching,
                (),
                _flag("SAD_XXZ_CROSS_MATCHING", candidate_matching_value),
            )
        )

    for q in (20, 24, 26, 28):
        production = "combined-init-cost"
        for candidate, value in (
            ("combined-init-cost", 0),
            ("split", 1),
            ("cost-rx-fused", 2),
        ):
            if candidate == production:
                continue
            rows.append(
                Comparison(
                    "fusion",
                    "qaoa_initial_layer",
                    "qaoa",
                    q,
                    production,
                    candidate,
                    (),
                    _flag("SAD_QAOA_INITIAL_STRATEGY", value),
                )
            )

    for q in (10, 20, 24, 28):
        rows.append(
            Comparison(
                "diagonal",
                "qaoa_domain_wall_lookup",
                "qaoa",
                q,
                "compact-domain-wall",
                "chunk-code",
                (
                    "-DSAD_QAOA_COMPACT_LOOKUP=1",
                    "-DSAD_QAOA_FUSED_BACKWARD=0",
                ),
                (
                    "-DSAD_QAOA_COMPACT_LOOKUP=0",
                    "-DSAD_QAOA_FUSED_BACKWARD=0",
                ),
                "isolated_same_split_backward",
            )
        )
        for candidate, threads in (("t64", 64), ("t256", 256)):
            rows.append(
                Comparison(
                    "diagonal",
                    "qaoa_shared_diagonal_threads",
                    "qaoa",
                    q,
                    "t128",
                    candidate,
                    (),
                    _flag("SAD_SHARED_DIAGONAL_BLOCK_THREADS", threads),
                )
            )

        for circuit in ("su2-hea", "rzz-hea"):
            production_lookup = (
                "k6" if circuit == "rzz-hea" and q == 10
                else "k10" if circuit == "rzz-hea" and q == 20
                else "k8"
            )
            for candidate, bits in (
                item for item in (("k6", 6), ("k8", 8), ("k10", 10))
                if item[0] != production_lookup
            ):
                rows.append(
                    Comparison(
                        "diagonal",
                        "diagonal_lookup_bits",
                        circuit,
                        q,
                        production_lookup,
                        candidate,
                        (),
                        _flag("SAD_DIAGONAL_LOOKUP_BITS", bits),
                    )
                )
            if q >= 20:
                rows.extend(
                    (
                        Comparison(
                            "diagonal",
                            "ordinary_diagonal_threads",
                            circuit,
                            q,
                            "t64",
                            "t128",
                            (),
                            _flag("SAD_DIAGONAL_BLOCK_THREADS", 128),
                        ),
                        Comparison(
                            "diagonal",
                            "diagonal_reduction",
                            circuit,
                            q,
                            "cta",
                            "warp-atomic",
                            (),
                            _flag("SAD_DIAGONAL_WARP_ATOMIC", 1),
                        ),
                    )
                )
    return tuple(rows)


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    files = sorted(
        file
        for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h")
        for file in SAD_ROOT.glob(pattern)
    )
    for file in files:
        digest.update(str(file.relative_to(SAD_ROOT)).encode())
        digest.update(file.read_bytes())
    # Dispatch policy and comparison construction are part of the experiment,
    # even though they do not alter the CUDA source tree.
    for file in (Path(sad_runner.__file__), Path(__file__)):
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def _shape(circuit: str, qubits: int) -> tuple[str, tuple[str, ...]]:
    explicit = os.environ.pop("SAD_LIBRARY_PATH", None)
    try:
        variant, _ = sad_runner._select_library(
            CIRCUIT_IDS[circuit], qubits, "optimized"
        )
    finally:
        if explicit is not None:
            os.environ["SAD_LIBRARY_PATH"] = explicit
    return variant, tuple(sad_runner._VARIANT_FLAGS.get(variant, ()))


def _plans(circuit: str, qubits: int) -> tuple[str, str]:
    return sad_runner._select_phase_plans(
        CIRCUIT_IDS[circuit], qubits, "optimized"
    )


def _merged_flags(
    shape_flags: tuple[str, ...], overrides: tuple[str, ...]
) -> tuple[str, ...]:
    """Apply -D overrides without passing duplicate macro definitions to nvcc."""

    override_names = {
        flag.split("=", 1)[0]
        for flag in overrides
        if flag.startswith("-D")
    }
    return tuple(
        flag for flag in shape_flags
        if not (flag.startswith("-D") and flag.split("=", 1)[0] in override_names)
    ) + overrides


def _library(shape: str, shape_flags: tuple[str, ...], flags: tuple[str, ...]) -> Path:
    all_flags = _merged_flags(shape_flags, flags)
    digest = hashlib.sha256("\0".join(all_flags).encode()).hexdigest()[:12]
    return SAD_ROOT / "build" / f"libsad_structure_{shape}_{digest}.so"


def _build(path: Path, flags: tuple[str, ...]) -> None:
    source_mtime = max(
        file.stat().st_mtime
        for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h")
        for file in SAD_ROOT.glob(pattern)
    )
    if path.exists() and path.stat().st_mtime >= source_mtime:
        return
    command = [
        "make",
        "-C",
        str(SAD_ROOT),
        f"TARGET={path.relative_to(SAD_ROOT)}",
    ]
    if flags:
        command.append(f"EXTRA_NVCCFLAGS={' '.join(flags)}")
    subprocess.run(command, cwd=ROOT, check=True)


def _steps(qubits: int) -> int:
    if qubits <= 12:
        return 10
    if qubits <= 20:
        return 5
    if qubits <= 24:
        return 3
    return 2


def _repetitions(qubits: int) -> int:
    return 4 if qubits <= 24 else 3


def _measure(
    comparison: Comparison,
    library: Path,
    plans: tuple[str, str],
) -> object:
    os.environ["SAD_LIBRARY_PATH"] = str(library)
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    return energy_and_grad(
        circuit=comparison.circuit,
        random_seed=SEED,
        scalability=(comparison.qubits, LAYERS),
        steps=_steps(comparison.qubits),
        warmup_steps=1,
        forward_phase_plan=plans[0],
        backward_phase_plan=plans[1],
    )


def _mean_ms(values: tuple[float, ...]) -> float:
    return 1000 * statistics.fmean(values)


def _median_ms(values: tuple[float, ...]) -> float:
    return 1000 * statistics.median(values)


def _mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _write_summary(raw_path: Path, output: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    family_fields = (
        "category", "decision", "scope", "circuit", "qubits", "layers"
    )
    expected_candidates: dict[tuple[str, ...], set[str]] = {}
    for item in comparisons():
        family = (
            item.category, item.decision, item.scope, item.circuit,
            str(item.qubits), str(LAYERS),
        )
        expected_candidates.setdefault(family, set()).add(item.candidate_choice)
    versions: dict[tuple[tuple[str, ...], str], list[dict[str, str]]] = {}
    for row in raw:
        family = tuple(row[field] for field in family_fields)
        versions.setdefault((family, row["source_fingerprint"]), []).append(row)
    latest_complete: list[dict[str, str]] = []
    for family, candidate_set in expected_candidates.items():
        eligible: list[list[dict[str, str]]] = []
        for (row_family, _), rows in versions.items():
            if row_family != family:
                continue
            counts = Counter(row["candidate_choice"] for row in rows)
            expected_repetitions = _repetitions(int(family[4]))
            if set(counts) == candidate_set and all(
                counts[candidate] == expected_repetitions
                for candidate in candidate_set
            ):
                eligible.append(rows)
        if eligible:
            latest_complete.extend(
                max(eligible, key=lambda rows: max(row["timestamp_utc"] for row in rows))
            )
    raw = latest_complete
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    group_fields = (
        "category",
        "decision",
        "scope",
        "circuit",
        "qubits",
        "layers",
        "shape",
        "forward_phase_plan",
        "backward_phase_plan",
        "production_choice",
        "candidate_choice",
        "source_fingerprint",
    )
    for row in raw:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    summary: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        meta = dict(zip(group_fields, key, strict=True))
        ratios = [float(row["candidate_over_production"]) for row in rows]
        ratio = statistics.median(ratios)
        correct = all(row["correct"] == "1" for row in rows)
        if ratio >= 1.02:
            verdict = "production_faster"
        elif ratio <= 1 / 1.02:
            verdict = "candidate_faster"
        else:
            verdict = "near_tie_keep_production"
        summary.append(
            {
                "experiment_group": "structure_and_diagonal_strategy_experiment",
                "hardware": "NVIDIA RTX 6000 Ada",
                **meta,
                "paired_samples": len(rows),
                "candidate_over_production": ratio,
                "paired_ratio_mad": _mad(ratios),
                "production_improvement_percent": 100 * (ratio - 1),
                "candidate_over_production_forward": statistics.median(
                    float(row["candidate_over_production_forward"]) for row in rows
                ),
                "candidate_over_production_hamiltonian": statistics.median(
                    float(row["candidate_over_production_hamiltonian"])
                    for row in rows
                ),
                "candidate_over_production_backward": statistics.median(
                    float(row["candidate_over_production_backward"]) for row in rows
                ),
                "verdict_at_2_percent": verdict,
                "correct": int(correct),
                "max_energy_abs_error": max(
                    float(row["energy_abs_error"]) for row in rows
                ),
                "max_gradient_abs_error": max(
                    float(row["gradient_max_abs_error"]) for row in rows
                ),
                "source_raw": str(raw_path.relative_to(ROOT)),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--category", choices=("fusion", "diagonal", "bond_schedule"))
    parser.add_argument(
        "--qubits",
        help="optional comma-separated qubit filter for a targeted refresh",
    )
    args = parser.parse_args()
    qubit_filter = (
        {int(value) for value in args.qubits.split(",")}
        if args.qubits else None
    )
    selected = tuple(
        item for item in comparisons()
        if (args.category is None or item.category == args.category)
        and (qubit_filter is None or item.qubits in qubit_filter)
    )
    fingerprint = _source_fingerprint()

    libraries: dict[tuple[str, tuple[str, ...]], Path] = {}
    for comparison in selected:
        shape, shape_flags = _shape(comparison.circuit, comparison.qubits)
        for flags in (comparison.production_flags, comparison.candidate_flags):
            key = shape, flags
            if key in libraries:
                continue
            path = _library(shape, shape_flags, flags)
            print(f"build {shape} {' '.join(flags) or 'production'}", flush=True)
            _build(path, _merged_flags(shape_flags, flags))
            libraries[key] = path

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, str, int, int]] = set()
    if args.resume and args.raw_output.exists():
        with args.raw_output.open(newline="", encoding="utf-8") as stream:
            existing = {
                (
                    row["source_fingerprint"],
                    f"{row['decision']}:{row['circuit']}:{row['candidate_choice']}",
                    int(row["qubits"]),
                    int(row["repetition"]),
                )
                for row in csv.DictReader(stream)
                if row.get("correct") == "1"
            }
    mode = "a" if args.resume and args.raw_output.exists() else "w"
    needs_header = mode == "w" or args.raw_output.stat().st_size == 0
    total = sum(_repetitions(item.qubits) for item in selected)
    completed = 0
    with args.raw_output.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, lineterminator="\n")
        if needs_header:
            writer.writeheader()
        max_repetitions = max(_repetitions(item.qubits) for item in selected)
        for repetition in range(max_repetitions):
            eligible = [
                item for item in selected
                if repetition < _repetitions(item.qubits)
            ]
            random.Random(20260817 + repetition).shuffle(eligible)
            for index, comparison in enumerate(eligible):
                unique = (
                    fingerprint,
                    f"{comparison.decision}:{comparison.circuit}:{comparison.candidate_choice}",
                    comparison.qubits,
                    repetition,
                )
                if unique in existing:
                    completed += 1
                    continue
                shape, _ = _shape(comparison.circuit, comparison.qubits)
                plans = _plans(comparison.circuit, comparison.qubits)
                production_library = libraries[shape, comparison.production_flags]
                candidate_library = libraries[shape, comparison.candidate_flags]
                order = ("production", "candidate")
                if (repetition + index) & 1:
                    order = tuple(reversed(order))
                measured = {}
                for name in order:
                    library = (
                        production_library if name == "production" else candidate_library
                    )
                    measured[name] = _measure(comparison, library, plans)
                production = measured["production"]
                candidate = measured["candidate"]
                production_total = _median_ms(production.step_times_s)
                candidate_total = _median_ms(candidate.step_times_s)
                production_forward = _mean_ms(production.forward_times_s)
                candidate_forward = _mean_ms(candidate.forward_times_s)
                production_hamiltonian = _mean_ms(production.hamiltonian_times_s)
                candidate_hamiltonian = _mean_ms(candidate.hamiltonian_times_s)
                production_backward = _mean_ms(production.backward_times_s)
                candidate_backward = _mean_ms(candidate.backward_times_s)
                energy_error = abs(production.energy - candidate.energy)
                gradient_error = float(
                    np.max(np.abs(production.grad - candidate.grad))
                )
                correct = energy_error <= 1e-10 and gradient_error <= 2e-9
                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "source_fingerprint": fingerprint,
                        "repetition": repetition,
                        "order": "-".join(order),
                        "category": comparison.category,
                        "decision": comparison.decision,
                        "scope": comparison.scope,
                        "circuit": comparison.circuit,
                        "qubits": comparison.qubits,
                        "layers": LAYERS,
                        "shape": shape,
                        "forward_phase_plan": plans[0],
                        "backward_phase_plan": plans[1],
                        "production_choice": comparison.production_choice,
                        "candidate_choice": comparison.candidate_choice,
                        "production_flags": json.dumps(comparison.production_flags),
                        "candidate_flags": json.dumps(comparison.candidate_flags),
                        "steps": _steps(comparison.qubits),
                        "warmup_steps": 1,
                        "production_median_ms": production_total,
                        "candidate_median_ms": candidate_total,
                        "candidate_over_production": candidate_total / production_total,
                        "production_forward_mean_ms": production_forward,
                        "candidate_forward_mean_ms": candidate_forward,
                        "candidate_over_production_forward": candidate_forward / production_forward,
                        "production_hamiltonian_mean_ms": production_hamiltonian,
                        "candidate_hamiltonian_mean_ms": candidate_hamiltonian,
                        "candidate_over_production_hamiltonian": candidate_hamiltonian / production_hamiltonian,
                        "production_backward_mean_ms": production_backward,
                        "candidate_backward_mean_ms": candidate_backward,
                        "candidate_over_production_backward": candidate_backward / production_backward,
                        "energy_abs_error": energy_error,
                        "gradient_max_abs_error": gradient_error,
                        "correct": int(correct),
                        "production_library": production_library.name,
                        "candidate_library": candidate_library.name,
                    }
                )
                stream.flush()
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] {comparison.circuit} "
                    f"q={comparison.qubits} {comparison.decision}: "
                    f"{comparison.candidate_choice}/production="
                    f"{candidate_total / production_total:.4f}x "
                    f"correct={correct}",
                    flush=True,
                )
                if not correct:
                    raise RuntimeError(
                        f"correctness failure: {comparison} energy={energy_error} "
                        f"gradient={gradient_error}"
                    )
                gc.collect()
    os.environ.pop("SAD_LIBRARY_PATH", None)
    _write_summary(args.raw_output, args.summary_output)
    print(f"raw CSV written to {args.raw_output}")
    print(f"summary CSV written to {args.summary_output}")


if __name__ == "__main__":
    main()
