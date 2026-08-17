"""Refine mailbox cuts after RX/RY phase geometry has been fixed.

The phase division, per-phase threads/register geometry, and continuity family
come from the already measured heterogeneous finalists.  This benchmark then
changes exactly one phase's ``SAD_MAILBOX_CHUNKS`` at a time, selects the best
cut independently for every phase, and validates the combined assignment on
the complete direction layer.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from benchmark_heterogeneous_phases import ROOT, SHAPES, _build, _measure_pair


DEFAULT_HETEROGENEOUS_RAW = (
    ROOT / "benchmark" / "results" / "heterogeneous_phase_paired_raw.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "mailbox_phase_refinement_final_raw.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "benchmark" / "results" / "mailbox_phase_refinement_summary.csv"
)
DEFAULT_SOURCE_OUTPUT = (
    ROOT / "benchmark" / "results" / "mailbox_phase_refinement_source_raw.csv"
)

VARIANT = re.compile(r"t(?P<threads>\d+)r(?P<register_bits>\d+)m(?P<chunks>\d+)")

RAW_FIELDS = (
    "stage",
    "repetition",
    "iterations",
    "order",
    "gate",
    "direction",
    "qubits",
    "fixed_source_candidate",
    "fixed_source_schedule",
    "base_m1_schedule",
    "candidate",
    "phase_index",
    "mailbox_chunks",
    "selected_cuts",
    "schedule",
    "average_ms",
    "base_average_ms",
    "base_over_candidate",
    "predicted_combined_ms",
    "predicted_combined_speedup",
    "phi_checksum_abs_error",
    "lambda_checksum_abs_error",
    "gradient_checksum_abs_error",
    "correct",
)

SUMMARY_FIELDS = (
    "gate",
    "direction",
    "qubits",
    "fixed_source_candidate",
    "fixed_source_schedule",
    "base_m1_schedule",
    "eligible_phases",
    "tested_single_phase_cuts",
    "selected_cuts",
    "combined_schedule",
    "samples",
    "base_median_ms",
    "combined_median_ms",
    "actual_speedup",
    "source_samples",
    "source_median_ms",
    "refined_median_ms_against_source",
    "refined_speedup_vs_source",
    "decision",
    "predicted_combined_ms",
    "predicted_combined_speedup",
    "prediction_error_percent",
    "max_phi_checksum_abs_error",
    "max_lambda_checksum_abs_error",
    "max_gradient_checksum_abs_error",
    "correct",
)

SOURCE_FIELDS = (
    "repetition",
    "iterations",
    "gate",
    "direction",
    "qubits",
    "fixed_source_candidate",
    "fixed_source_schedule",
    "selected_cuts",
    "refined_schedule",
    "source_average_ms",
    "refined_average_ms",
    "source_over_refined",
    "phi_checksum_abs_error",
    "lambda_checksum_abs_error",
    "gradient_checksum_abs_error",
    "correct",
)


@dataclass(frozen=True)
class FixedSchedule:
    source_candidate: str
    source_schedule: str
    base_schedule: str


def _tokens(schedule: str) -> list[list[str]]:
    result: list[list[str]] = []
    for token in schedule.split(";"):
        fields = token.split("/")
        if len(fields) != 3 or VARIANT.fullmatch(fields[0]) is None:
            raise ValueError(f"invalid phase token {token!r}")
        result.append(fields)
    return result


def _variant_with_chunks(variant: str, chunks: int) -> str:
    match = VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError(f"invalid variant {variant!r}")
    name = (
        f"t{match.group('threads')}r{match.group('register_bits')}m{chunks}"
    )
    if name not in SHAPES:
        raise ValueError(f"mailbox variant {name!r} is not compiled")
    return name


def normalize_mailboxes(schedule: str) -> str:
    """Preserve phase geometry and continuity while resetting every cut to m1."""

    tokens = _tokens(schedule)
    for fields in tokens:
        fields[0] = _variant_with_chunks(fields[0], 1)
    return ";".join("/".join(fields) for fields in tokens)


def mailbox_options(variant: str) -> tuple[int, ...]:
    match = VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError(f"invalid variant {variant!r}")
    prefix = f"t{match.group('threads')}r{match.group('register_bits')}m"
    return tuple(
        sorted(
            int(name.rsplit("m", 1)[1])
            for name in SHAPES
            if name.startswith(prefix)
        )
    )


def set_phase_mailbox(schedule: str, phase_index: int, chunks: int) -> str:
    tokens = _tokens(schedule)
    if phase_index < 0 or phase_index >= len(tokens):
        raise IndexError(phase_index)
    tokens[phase_index][0] = _variant_with_chunks(tokens[phase_index][0], chunks)
    return ";".join("/".join(fields) for fields in tokens)


def _fixed_schedules(path: Path) -> dict[tuple[str, str, int], FixedSchedule]:
    samples: dict[
        tuple[str, str, int, str, str], list[float]
    ] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("correct") != "1":
                continue
            key = (
                row["gate"],
                row["direction"],
                int(row["qubits"]),
                row["candidate"],
                row["schedule"],
            )
            samples[key].append(float(row["average_ms"]))
    by_scenario: dict[
        tuple[str, str, int], list[tuple[float, str, str]]
    ] = defaultdict(list)
    for key, values in samples.items():
        by_scenario[key[:3]].append(
            (statistics.median(values), key[3], key[4])
        )
    result: dict[tuple[str, str, int], FixedSchedule] = {}
    for scenario, candidates in by_scenario.items():
        _, source_candidate, source_schedule = min(candidates)
        result[scenario] = FixedSchedule(
            source_candidate,
            source_schedule,
            normalize_mailboxes(source_schedule),
        )
    return result


def single_phase_candidates(
    fixed: FixedSchedule,
) -> list[tuple[str, int, int, str]]:
    """Return all one-phase cuts after fixing the phase schedule."""

    result: list[tuple[str, int, int, str]] = []
    for phase_index, fields in enumerate(_tokens(fixed.base_schedule)):
        for chunks in mailbox_options(fields[0]):
            if chunks == 1:
                continue
            result.append(
                (
                    f"phase-{phase_index}-m{chunks}",
                    phase_index,
                    chunks,
                    set_phase_mailbox(fixed.base_schedule, phase_index, chunks),
                )
            )
    return result


def _completed(path: Path) -> set[tuple[str, int, str, str, int, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (
                row["stage"],
                int(row["repetition"]),
                row["gate"],
                row["direction"],
                int(row["qubits"]),
                row["candidate"],
                row["schedule"],
            )
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def _errors(
    measurement: tuple[float, float, float, float, int],
    baseline: tuple[float, float, float, float, int],
) -> tuple[tuple[float, float, float], bool]:
    errors = tuple(abs(measurement[index] - baseline[index]) for index in (1, 2, 3))
    scale = max(1.0, *(abs(baseline[index]) for index in (1, 2, 3)))
    return errors, max(errors) <= 1e-8 * scale


def _write_measurement(
    writer: csv.DictWriter,
    *,
    stage: str,
    repetition: int,
    iterations: int,
    order: list[str],
    scenario: tuple[str, str, int],
    fixed: FixedSchedule,
    candidate: str,
    phase_index: int,
    chunks: int,
    selected_cuts: str,
    schedule: str,
    measurement: tuple[float, float, float, float, int],
    baseline: tuple[float, float, float, float, int],
    predicted_ms: float | None = None,
    predicted_speedup: float | None = None,
) -> bool:
    errors, correct = _errors(measurement, baseline)
    writer.writerow(
        {
            "stage": stage,
            "repetition": repetition,
            "iterations": iterations,
            "order": "-".join(order),
            "gate": scenario[0],
            "direction": scenario[1],
            "qubits": scenario[2],
            "fixed_source_candidate": fixed.source_candidate,
            "fixed_source_schedule": fixed.source_schedule,
            "base_m1_schedule": fixed.base_schedule,
            "candidate": candidate,
            "phase_index": phase_index,
            "mailbox_chunks": chunks,
            "selected_cuts": selected_cuts,
            "schedule": schedule,
            "average_ms": measurement[0],
            "base_average_ms": baseline[0],
            "base_over_candidate": baseline[0] / measurement[0],
            "predicted_combined_ms": "" if predicted_ms is None else predicted_ms,
            "predicted_combined_speedup": (
                "" if predicted_speedup is None else predicted_speedup
            ),
            "phi_checksum_abs_error": errors[0],
            "lambda_checksum_abs_error": errors[1],
            "gradient_checksum_abs_error": errors[2],
            "correct": int(correct),
        }
    )
    return correct


def _local_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row["stage"] == "local" and row.get("correct") == "1"
        ]


def select_independent_cuts(
    path: Path,
    fixed_schedules: dict[tuple[str, str, int], FixedSchedule],
    *,
    min_speedup: float,
) -> dict[tuple[str, str, int], tuple[str, str, float, float]]:
    """Select each phase's mailbox cut from one-variable paired medians."""

    samples: dict[tuple[str, str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in _local_rows(path):
        if row["candidate"] == "base-m1":
            continue
        key = (
            row["gate"],
            row["direction"],
            int(row["qubits"]),
            int(row["phase_index"]),
            int(row["mailbox_chunks"]),
        )
        samples[key].append(row)

    result: dict[tuple[str, str, int], tuple[str, str, float, float]] = {}
    for scenario, fixed in fixed_schedules.items():
        choices: list[tuple[int, int, float, float]] = []
        phase_indices = range(len(_tokens(fixed.base_schedule)))
        for phase_index in phase_indices:
            options: list[tuple[float, int, float]] = []
            for key, rows in samples.items():
                if key[:3] != scenario or key[3] != phase_index:
                    continue
                ratio = statistics.median(
                    float(row["base_over_candidate"]) for row in rows
                )
                delta = statistics.median(
                    float(row["base_average_ms"]) - float(row["average_ms"])
                    for row in rows
                )
                options.append((ratio, key[4], delta))
            if options:
                ratio, chunks, delta = max(options)
                if ratio >= min_speedup:
                    choices.append((phase_index, chunks, ratio, delta))

        schedule = fixed.base_schedule
        for phase_index, chunks, _, _ in choices:
            schedule = set_phase_mailbox(schedule, phase_index, chunks)
        selected = ";".join(
            f"phase-{phase_index}=m{chunks}({ratio:.4f}x)"
            for phase_index, chunks, ratio, _ in choices
        ) or "none"

        scenario_rows = [
            row
            for row in _local_rows(path)
            if (row["gate"], row["direction"], int(row["qubits"])) == scenario
        ]
        if scenario_rows:
            base_ms = statistics.median(
                float(row["base_average_ms"]) for row in scenario_rows
            )
            predicted_ms = base_ms - sum(max(0.0, choice[3]) for choice in choices)
            if predicted_ms <= 0:
                raise RuntimeError(f"invalid additive prediction for {scenario}")
            predicted_speedup = base_ms / predicted_ms
        else:
            # A schedule containing only single-warp phases has no mailbox.
            # The paired combined pass will fill in the actual base time.
            predicted_ms = math.nan
            predicted_speedup = 1.0
        result[scenario] = (schedule, selected, predicted_ms, predicted_speedup)
    return result


def write_summary(
    raw_path: Path,
    source_path: Path,
    summary_path: Path,
    fixed_schedules: dict[tuple[str, str, int], FixedSchedule],
    selections: dict[tuple[str, str, int], tuple[str, str, float, float]],
) -> None:
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with source_path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    summaries: list[dict[str, object]] = []
    for scenario, fixed in sorted(fixed_schedules.items()):
        selected_schedule = selections[scenario][0]
        combined = [
            row
            for row in rows
            if row["stage"] == "combined"
            and row.get("correct") == "1"
            and row["schedule"] == selected_schedule
            and (row["gate"], row["direction"], int(row["qubits"])) == scenario
        ]
        if not combined:
            continue
        local = [
            row
            for row in rows
            if row["stage"] == "local"
            and row["candidate"] != "base-m1"
            and (row["gate"], row["direction"], int(row["qubits"])) == scenario
        ]
        base_ms = statistics.median(float(row["base_average_ms"]) for row in combined)
        combined_ms = statistics.median(float(row["average_ms"]) for row in combined)
        predicted_ms = statistics.median(
            float(row["predicted_combined_ms"]) for row in combined
        )
        source_comparisons = [
            row
            for row in source_rows
            if row.get("correct") == "1"
            and row["refined_schedule"] == selected_schedule
            and (row["gate"], row["direction"], int(row["qubits"])) == scenario
        ]
        if not source_comparisons:
            continue
        source_ms = statistics.median(
            float(row["source_average_ms"]) for row in source_comparisons
        )
        refined_source_ms = statistics.median(
            float(row["refined_average_ms"]) for row in source_comparisons
        )
        source_speedup = statistics.median(
            float(row["source_over_refined"]) for row in source_comparisons
        )
        actual_speedup = base_ms / combined_ms
        decision = (
            "accept-refined"
            if actual_speedup >= 1.02 and source_speedup >= 1.01
            else "keep-fixed-source"
        )
        tokens = _tokens(fixed.base_schedule)
        eligible = [
            index
            for index, fields in enumerate(tokens)
            if len(mailbox_options(fields[0])) > 1
        ]
        summaries.append(
            {
                "gate": scenario[0],
                "direction": scenario[1],
                "qubits": scenario[2],
                "fixed_source_candidate": fixed.source_candidate,
                "fixed_source_schedule": fixed.source_schedule,
                "base_m1_schedule": fixed.base_schedule,
                "eligible_phases": ":".join(map(str, eligible)) or "none",
                "tested_single_phase_cuts": len(
                    {(row["phase_index"], row["mailbox_chunks"]) for row in local}
                ),
                "selected_cuts": combined[0]["selected_cuts"],
                "combined_schedule": combined[0]["schedule"],
                "samples": len(combined),
                "base_median_ms": base_ms,
                "combined_median_ms": combined_ms,
                "actual_speedup": actual_speedup,
                "source_samples": len(source_comparisons),
                "source_median_ms": source_ms,
                "refined_median_ms_against_source": refined_source_ms,
                "refined_speedup_vs_source": source_speedup,
                "decision": decision,
                "predicted_combined_ms": predicted_ms,
                "predicted_combined_speedup": base_ms / predicted_ms,
                "prediction_error_percent": 100 * (predicted_ms / combined_ms - 1),
                "max_phi_checksum_abs_error": max(
                    float(row["phi_checksum_abs_error"]) for row in combined
                ),
                "max_lambda_checksum_abs_error": max(
                    float(row["lambda_checksum_abs_error"]) for row in combined
                ),
                "max_gradient_checksum_abs_error": max(
                    float(row["gradient_checksum_abs_error"]) for row in combined
                ),
                "correct": int(all(row["correct"] == "1" for row in combined)),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heterogeneous-raw", type=Path, default=DEFAULT_HETEROGENEOUS_RAW
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--small-iterations",
        type=int,
        default=30,
        help="paired iterations for q<=24, where a complete layer is short",
    )
    parser.add_argument("--min-speedup", type=float, default=1.01)
    args = parser.parse_args()
    if (
        min(args.rounds, args.iterations, args.small_iterations) < 1
        or args.iterations % 2
        or args.small_iterations % 2
        or args.min_speedup < 1
    ):
        parser.error(
            "rounds must be positive, iteration counts must be positive/even, "
            "and min-speedup must be >= 1"
        )

    fixed_schedules = _fixed_schedules(args.heterogeneous_raw)
    binary = _build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0

    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            scenarios = list(fixed_schedules)
            random.Random(20260817 + repetition).shuffle(scenarios)
            for scenario in scenarios:
                fixed = fixed_schedules[scenario]
                scenario_iterations = (
                    args.small_iterations if scenario[2] <= 24 else args.iterations
                )
                candidates = single_phase_candidates(fixed)
                pending = [
                    value
                    for value in candidates
                    if (
                        "local",
                        repetition,
                        *scenario,
                        value[0],
                        value[3],
                    )
                    not in done
                ]
                if not pending:
                    continue
                order = list(pending)
                random.Random(f"mailbox-local-{repetition}-{scenario}").shuffle(order)
                for name, phase_index, chunks, schedule in order:
                    baseline, candidate_measurement = _measure_pair(
                        binary,
                        scenario,
                        fixed.base_schedule,
                        schedule,
                        scenario_iterations,
                    )
                    correct = _write_measurement(
                        writer,
                        stage="local",
                        repetition=repetition,
                        iterations=scenario_iterations,
                        order=["interleaved"],
                        scenario=scenario,
                        fixed=fixed,
                        candidate=name,
                        phase_index=phase_index,
                        chunks=chunks,
                        selected_cuts="",
                        schedule=schedule,
                        measurement=candidate_measurement,
                        baseline=baseline,
                    )
                    stream.flush()
                    if not correct:
                        raise RuntimeError(f"checksum mismatch for {scenario} {name}")
                    done.add(
                        ("local", repetition, *scenario, name, schedule)
                    )
                    print(
                        f"local {scenario[0]}/{scenario[1][0]} q={scenario[2]} "
                        f"{name}: {baseline[0] / candidate_measurement[0]:.3f}x",
                        flush=True,
                    )

    selections = select_independent_cuts(
        args.output, fixed_schedules, min_speedup=args.min_speedup
    )
    done = _completed(args.output)
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, lineterminator="\n")
        for repetition in range(args.rounds):
            scenarios = list(fixed_schedules)
            random.Random(20260827 + repetition).shuffle(scenarios)
            for scenario in scenarios:
                fixed = fixed_schedules[scenario]
                scenario_iterations = (
                    args.small_iterations if scenario[2] <= 24 else args.iterations
                )
                schedule, selected, predicted_ms, predicted_speedup = selections[scenario]
                key = (
                    "combined",
                    repetition,
                    *scenario,
                    "independent-combined",
                    schedule,
                )
                if key in done:
                    continue
                order = ["interleaved"]
                baseline, candidate = _measure_pair(
                    binary,
                    scenario,
                    fixed.base_schedule,
                    schedule,
                    scenario_iterations,
                )
                if math.isnan(predicted_ms):
                    predicted_ms = baseline[0]
                    predicted_speedup = 1.0
                correct = _write_measurement(
                    writer,
                    stage="combined",
                    repetition=repetition,
                    iterations=scenario_iterations,
                    order=order,
                    scenario=scenario,
                    fixed=fixed,
                    candidate="independent-combined",
                    phase_index=-1,
                    chunks=-1,
                    selected_cuts=selected,
                    schedule=schedule,
                    measurement=candidate,
                    baseline=baseline,
                    predicted_ms=predicted_ms,
                    predicted_speedup=predicted_speedup,
                )
                stream.flush()
                if not correct:
                    raise RuntimeError(f"combined checksum mismatch for {scenario}")
                done.add(key)
                print(
                    f"combined {scenario[0]}/{scenario[1][0]} q={scenario[2]} "
                    f"{baseline[0] / candidate[0]:.3f}x ({selected})",
                    flush=True,
                )

    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    source_done: set[tuple[int, str, str, int, str]] = set()
    source_exists = args.source_output.exists() and args.source_output.stat().st_size > 0
    if source_exists:
        with args.source_output.open(newline="", encoding="utf-8") as stream:
            source_done = {
                (
                    int(row["repetition"]),
                    row["gate"],
                    row["direction"],
                    int(row["qubits"]),
                    row["refined_schedule"],
                )
                for row in csv.DictReader(stream)
                if row.get("correct") == "1"
            }
    with args.source_output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        if not source_exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            for scenario, fixed in sorted(fixed_schedules.items()):
                iterations = (
                    args.small_iterations if scenario[2] <= 24 else args.iterations
                )
                refined_schedule, selected, _, _ = selections[scenario]
                key = (repetition, *scenario, refined_schedule)
                if key in source_done:
                    continue
                source_measurement, refined_measurement = _measure_pair(
                    binary,
                    scenario,
                    fixed.source_schedule,
                    refined_schedule,
                    iterations,
                )
                errors, correct = _errors(refined_measurement, source_measurement)
                writer.writerow(
                    {
                        "repetition": repetition,
                        "iterations": iterations,
                        "gate": scenario[0],
                        "direction": scenario[1],
                        "qubits": scenario[2],
                        "fixed_source_candidate": fixed.source_candidate,
                        "fixed_source_schedule": fixed.source_schedule,
                        "selected_cuts": selected,
                        "refined_schedule": refined_schedule,
                        "source_average_ms": source_measurement[0],
                        "refined_average_ms": refined_measurement[0],
                        "source_over_refined": (
                            source_measurement[0] / refined_measurement[0]
                        ),
                        "phi_checksum_abs_error": errors[0],
                        "lambda_checksum_abs_error": errors[1],
                        "gradient_checksum_abs_error": errors[2],
                        "correct": int(correct),
                    }
                )
                stream.flush()
                if not correct:
                    raise RuntimeError(
                        f"source/refined checksum mismatch for {scenario}"
                    )
                print(
                    f"source {scenario[0]}/{scenario[1][0]} q={scenario[2]} "
                    f"{source_measurement[0] / refined_measurement[0]:.3f}x",
                    flush=True,
                )

    write_summary(
        args.output,
        args.source_output,
        args.summary,
        fixed_schedules,
        selections,
    )
    print(f"raw CSV written to {args.output}")
    print(f"source comparison CSV written to {args.source_output}")
    print(f"summary CSV written to {args.summary}")


if __name__ == "__main__":
    main()
