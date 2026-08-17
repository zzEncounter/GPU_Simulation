"""Aggregate execution-search data and fit an interpretable cost model."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark" / "results" / "execution_search_raw.csv"
DEFAULT_SUMMARY = (
    ROOT / "benchmark" / "results" / "execution_search_summary.csv"
)
DEFAULT_REPORT = ROOT / "docs" / "research" / "EXECUTION_STRATEGY_REPORT.md"
NEAR_TIE_FRACTION = 0.02
MODEL_FEATURES = (
    "state_pass_gib",
    "lane_gates_g",
    "register_gates_g",
    "warp_gates_g",
    "mailbox_gib",
    "barrier_million",
    "gradient_atomic_million",
    "cta_waves",
    "occupancy_pressure_gib",
    "launches",
)


@dataclass(frozen=True)
class Aggregate:
    stage: str
    variant: str
    family: str
    candidate: str
    gate: str
    direction: str
    qubits: int
    layout: str
    threads: int
    register_amplitudes: int
    tile_bits: int
    phase_count: int
    gate_count: int
    registers_per_thread: int
    static_shared_bytes: int
    dynamic_shared_bytes: int
    active_cta_per_sm: int
    mailbox_bytes: int
    mailbox_chunks: int
    local_bytes_per_thread: int
    multiprocessors: int
    phase_targets: tuple[int, ...]
    phase_lane_targets: tuple[int, ...]
    phase_register_targets: tuple[int, ...]
    phase_warp_targets: tuple[int, ...]
    samples: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples)

    @property
    def spread_percent(self) -> float:
        if len(self.samples) < 2 or self.median_ms == 0:
            return 0.0
        return 100 * (max(self.samples) - min(self.samples)) / self.median_ms

    @property
    def register_bits(self) -> int:
        return int(math.log2(self.register_amplitudes))

    @property
    def warp_bits(self) -> int:
        return self.tile_bits - 5 - self.register_bits

    def features(self) -> np.ndarray:
        amplitudes = float(1 << self.qubits)
        state_bytes_per_phase = (32 if self.direction == "forward" else 64)
        state_pass_gib = (
            amplitudes * state_bytes_per_phase * self.phase_count / (1 << 30)
        )
        lane = amplitudes * sum(self.phase_lane_targets) / 1e9
        register = amplitudes * sum(self.phase_register_targets) / 1e9
        warp_count = sum(self.phase_warp_targets)
        warp = amplitudes * warp_count / 1e9
        exchange_bytes = 32 if self.direction == "forward" else 64
        mailbox_gib = amplitudes * warp_count * exchange_bytes / (1 << 30)
        tiles = float(1 << max(0, self.qubits - self.tile_bits))
        sync_multiplier = 2 if self.direction == "forward" else 4
        barriers = (
            tiles
            * self.threads
            * warp_count
            * self.mailbox_chunks
            * sync_multiplier
            / 1e6
        )
        gradient_atomics = (
            0.0
            if self.direction == "forward"
            else tiles * self.gate_count / 1e6
        )
        resident_ctas = max(1, self.active_cta_per_sm * self.multiprocessors)
        cta_waves = self.phase_count * math.ceil(tiles / resident_ctas)
        occupancy_pressure = state_pass_gib / max(1, self.active_cta_per_sm)
        return np.asarray(
            (
                state_pass_gib,
                lane,
                register,
                warp,
                mailbox_gib,
                barriers,
                gradient_atomics,
                cta_waves,
                occupancy_pressure,
                float(self.phase_count),
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Model:
    coefficients: np.ndarray

    def predict(self, rows: Sequence[Aggregate]) -> np.ndarray:
        return np.asarray([row.features() for row in rows]) @ self.coefficients


def _tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(":")) if value else ()


def load_aggregates(path: Path) -> list[Aggregate]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(
                row[field]
                for field in (
                    "stage",
                    "variant",
                    "family",
                    "candidate",
                    "gate",
                    "direction",
                    "qubits",
                    "layout",
                )
            )
            grouped[key].append(row)
    result: list[Aggregate] = []
    for key, rows in grouped.items():
        row = rows[0]
        result.append(
            Aggregate(
                stage=row["stage"],
                variant=row["variant"],
                family=row["family"],
                candidate=row["candidate"],
                gate=row["gate"],
                direction=row["direction"],
                qubits=int(row["qubits"]),
                layout=row["layout"],
                threads=int(row["threads"]),
                register_amplitudes=int(row["register_amplitudes"]),
                tile_bits=int(row["tile_bits"]),
                phase_count=int(row["phase_count"]),
                gate_count=int(row["gate_count"]),
                registers_per_thread=int(row["registers_per_thread"]),
                static_shared_bytes=int(row["static_shared_bytes"]),
                dynamic_shared_bytes=int(row["dynamic_shared_bytes"]),
                active_cta_per_sm=int(row["active_cta_per_sm"]),
                mailbox_bytes=int(row["mailbox_bytes"]),
                mailbox_chunks=int(row["mailbox_chunks"]),
                local_bytes_per_thread=int(row["local_bytes_per_thread"]),
                multiprocessors=int(row["multiprocessors"]),
                phase_targets=_tuple(row["phase_targets"]),
                phase_lane_targets=_tuple(row["phase_lane_targets"]),
                phase_register_targets=_tuple(row["phase_register_targets"]),
                phase_warp_targets=_tuple(row["phase_warp_targets"]),
                samples=tuple(float(item["average_ms"]) for item in rows),
            )
        )
    return sorted(
        result,
        key=lambda item: (
            item.stage,
            item.gate,
            item.direction,
            item.qubits,
            item.median_ms,
        ),
    )


def confirmed_rows(
    rows: Iterable[Aggregate], minimum_schedule_samples: int
) -> list[Aggregate]:
    """Keep all calibration/shape rows and sufficiently repeated schedules."""

    if minimum_schedule_samples < 1:
        raise ValueError("minimum_schedule_samples must be positive")
    return [
        row for row in rows
        if row.stage != "schedule"
        or len(row.samples) >= minimum_schedule_samples
    ]


def nonnegative_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small exact active-set NNLS suitable for the eight physical features."""

    if x.shape[1] > 16:
        raise ValueError("exact subset NNLS is intended for small models")
    best_error = math.inf
    best = np.zeros(x.shape[1], dtype=np.float64)
    # Weight every row by 1/time: the fit minimizes relative rather than
    # q=28-dominated absolute error.
    weights = 1.0 / np.maximum(y, np.finfo(np.float64).eps)
    weighted_x = x * weights[:, None]
    weighted_y = y * weights
    for mask in range(1, 1 << x.shape[1]):
        active = [index for index in range(x.shape[1]) if mask & (1 << index)]
        coefficients, *_ = np.linalg.lstsq(
            weighted_x[:, active], weighted_y, rcond=None
        )
        if np.any(coefficients < -1e-12):
            continue
        prediction = weighted_x[:, active] @ coefficients
        error = float(np.sum((prediction - weighted_y) ** 2))
        if error < best_error:
            best_error = error
            best = np.zeros(x.shape[1], dtype=np.float64)
            best[active] = np.maximum(coefficients, 0)
    return best


def fit_model(rows: Sequence[Aggregate]) -> Model:
    if not rows:
        return Model(np.zeros(len(MODEL_FEATURES), dtype=np.float64))
    x = np.asarray([row.features() for row in rows])
    y = np.asarray([row.median_ms for row in rows])
    return Model(nonnegative_least_squares(x, y))


def relative_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(predicted - actual) / np.maximum(actual, 1e-12)


def ranking_metrics(
    rows: Sequence[Aggregate], predicted: np.ndarray
) -> tuple[int, int, list[float]]:
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row.gate, row.direction, row.qubits)].append(index)
    successes = 0
    regrets: list[float] = []
    for indices in grouped.values():
        chosen = min(indices, key=lambda index: predicted[index])
        best = min(rows[index].median_ms for index in indices)
        regret = rows[chosen].median_ms / best - 1
        regrets.append(regret)
        successes += regret <= NEAR_TIE_FRACTION
    return successes, len(grouped), regrets


def cross_validate(rows: Sequence[Aggregate]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for gate, direction, family in itertools.product(
        ("rx", "ry"), ("forward", "backward"), ("compact", "fixed", "pairs")
    ):
        subset = [
            row
            for row in rows
            if row.gate == gate
            and row.direction == direction
            and row.family == family
        ]
        for heldout in sorted({row.qubits for row in subset}):
            train = [row for row in subset if row.qubits != heldout]
            test = [row for row in subset if row.qubits == heldout]
            if not train or not test:
                continue
            prediction = fit_model(train).predict(test)
            actual = np.asarray([row.median_ms for row in test])
            errors = relative_errors(actual, prediction)
            successes, scenarios, regrets = ranking_metrics(test, prediction)
            results.append(
                {
                    "gate": gate,
                    "direction": direction,
                    "family": family,
                    "heldout_q": heldout,
                    "rows": len(test),
                    "median_ape_percent": 100 * statistics.median(errors),
                    "p90_ape_percent": 100 * float(np.percentile(errors, 90)),
                    "near_best_scenarios": successes,
                    "scenarios": scenarios,
                    "mean_selection_regret_percent": 100
                    * statistics.fmean(regrets),
                }
            )
    return results


def _scenario_groups(
    rows: Iterable[Aggregate], stage: str
) -> dict[tuple[str, str, int], list[Aggregate]]:
    result: dict[tuple[str, str, int], list[Aggregate]] = defaultdict(list)
    for row in rows:
        if row.stage == stage:
            result[(row.gate, row.direction, row.qubits)].append(row)
    return result


def write_summary(path: Path, rows: Sequence[Aggregate]) -> None:
    fields = (
        "stage",
        "gate",
        "direction",
        "qubits",
        "rank",
        "near_tie",
        "relative_to_best",
        "variant",
        "family",
        "candidate",
        "phase_count",
        "phase_targets",
        "phase_lane_targets",
        "phase_register_targets",
        "phase_warp_targets",
        "median_ms",
        "spread_percent",
        "samples",
        "threads",
        "register_amplitudes",
        "tile_bits",
        "registers_per_thread",
        "shared_bytes",
        "active_cta_per_sm",
        "mailbox_bytes",
        "mailbox_chunks",
    )
    output: list[dict[str, object]] = []
    for stage in ("shape", "schedule"):
        for (gate, direction, q), candidates in _scenario_groups(
            rows, stage
        ).items():
            ranked = sorted(candidates, key=lambda item: item.median_ms)
            best = ranked[0].median_ms
            for rank, row in enumerate(ranked, 1):
                relative = row.median_ms / best
                output.append(
                    {
                        "stage": stage,
                        "gate": gate,
                        "direction": direction,
                        "qubits": q,
                        "rank": rank,
                        "near_tie": int(relative <= 1 + NEAR_TIE_FRACTION),
                        "relative_to_best": relative,
                        "variant": row.variant,
                        "family": row.family,
                        "candidate": row.candidate,
                        "phase_count": row.phase_count,
                        "phase_targets": ":".join(map(str, row.phase_targets)),
                        "phase_lane_targets": ":".join(
                            map(str, row.phase_lane_targets)
                        ),
                        "phase_register_targets": ":".join(
                            map(str, row.phase_register_targets)
                        ),
                        "phase_warp_targets": ":".join(
                            map(str, row.phase_warp_targets)
                        ),
                        "median_ms": row.median_ms,
                        "spread_percent": row.spread_percent,
                        "samples": len(row.samples),
                        "threads": row.threads,
                        "register_amplitudes": row.register_amplitudes,
                        "tile_bits": row.tile_bits,
                        "registers_per_thread": row.registers_per_thread,
                        "shared_bytes": row.static_shared_bytes
                        + row.dynamic_shared_bytes,
                        "active_cta_per_sm": row.active_cta_per_sm,
                        "mailbox_bytes": row.mailbox_bytes,
                        "mailbox_chunks": row.mailbox_chunks,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def _winner_table(rows: Sequence[Aggregate], stage: str) -> list[str]:
    lines = [
        "| gate | dir | q | measured winner | phases | ms | within 2% |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for (gate, direction, q), candidates in sorted(
        _scenario_groups(rows, stage).items()
    ):
        ranked = sorted(candidates, key=lambda item: item.median_ms)
        best = ranked[0]
        ties = [
            f"{row.variant}/{row.family}/{row.candidate}"
            for row in ranked
            if row.median_ms <= best.median_ms * (1 + NEAR_TIE_FRACTION)
        ]
        lines.append(
            f"| {gate.upper()} | {direction[0].upper()} | {q} | "
            f"`{best.variant}/{best.family}/{best.candidate}` | "
            f"`{':'.join(map(str, best.phase_targets))}` | "
            f"{best.median_ms:.6f} | {', '.join(ties)} |"
        )
    return lines


def write_report(
    path: Path,
    source: Path,
    rows: Sequence[Aggregate],
    validation: Sequence[dict[str, object]],
) -> None:
    model_rows = [row for row in rows if row.stage in {"shape", "schedule"}]
    try:
        source_label = source.relative_to(ROOT)
    except ValueError:
        source_label = source
    lines = [
        "# Execution Strategy Search",
        "",
        "This report is generated by `benchmark/analyze_execution_search.py` "
        f"from `{source_label}`. Raw repetitions are reduced by "
        "median; alternatives within 2% are reported as ties.",
        "",
        "## Search semantics",
        "",
        "The search is exhaustive over lane/register/warp *equivalence "
        "classes*, not over permutations of slots that compile to the same "
        "instruction sequence. All reachable target totals are visited by a "
        "dynamic program. The measured frontier includes minimum-phase and "
        "configured extra-phase candidates.",
        "",
        "## Measured winners",
        "",
        *_winner_table(rows, "schedule"),
        "",
        "## White-box cost model",
        "",
        "For each gate and direction the non-negative model is:",
        "",
        "`T = c_mem*state_pass_GiB + c_lane*lane_gate_G + "
        "c_reg*register_gate_G + c_warp*warp_gate_G + "
        "c_smem*mailbox_GiB + c_barrier*barrier_M + "
        "c_atomic*gradient_atomic_M + c_wave*CTA_waves + "
        "c_occ*occupancy_pressure_GiB + c_launch*phase_count`.",
        "",
        "The terms are computed from q, phase masks, tile shape, mailbox "
        "chunking, and direction. Coefficients are calibrated on the target "
        "GPU; no Ada-specific bandwidth constant is baked into the searcher.",
        "",
        "| gate | dir | " + " | ".join(MODEL_FEATURES) + " |",
        "|---|---:|" + "---:|" * len(MODEL_FEATURES),
    ]
    for gate, direction in itertools.product(("rx", "ry"), ("forward", "backward")):
        subset = [
            row
            for row in model_rows
            if row.gate == gate and row.direction == direction
        ]
        coefficients = fit_model(subset).coefficients
        lines.append(
            f"| {gate.upper()} | {direction[0].upper()} | "
            + " | ".join(f"{value:.6g}" for value in coefficients)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-q-out validation",
            "",
            "`selection regret` is the measured slowdown of the model-selected "
            "candidate relative to the measured winner. `near-best` counts a "
            "selection as correct when regret is at most 2%.",
            "",
            "| gate | dir | family | held-out q | rows | median APE | p90 APE | "
            "near-best | mean regret |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in validation:
        lines.append(
            f"| {str(item['gate']).upper()} | "
            f"{str(item['direction'])[0].upper()} | {item['family']} | "
            f"{item['heldout_q']} | "
            f"{item['rows']} | {float(item['median_ape_percent']):.2f}% | "
            f"{float(item['p90_ape_percent']):.2f}% | "
            f"{item['near_best_scenarios']}/{item['scenarios']} | "
            f"{float(item['mean_selection_regret_percent']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Portable selection procedure",
            "",
            "1. Compile the legal tile/mailbox variants and measure canonical "
            "compact/fixed/pair layers at three q values spanning the target "
            "range.",
            "2. Keep variants on the per-scenario 2% Pareto front; mailbox "
            "chunking is useful only when its reduced shared allocation raises "
            "active CTAs enough to offset added barriers.",
            "3. Calibrate empty, lane, register and warp phases for survivors. "
            "Use the model to traverse all class schedules and benchmark its "
            "frontier, including one additional phase.",
            "4. Validate circuit fusion separately: fusion changes state-pass "
            "count and live ranges, so isolated ms/gate is not a sufficient "
            "dispatch criterion.",
            "5. Retain all configurations within 2%. Prefer the one with fewer "
            "special cases unless repeated end-to-end measurements establish a "
            "larger margin.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--minimum-schedule-samples", type=int, default=1,
        help="exclude schedule-screening rows with fewer repetitions",
    )
    args = parser.parse_args()
    rows = load_aggregates(args.input)
    if not rows:
        parser.error(f"no measurements in {args.input}")
    try:
        rows = confirmed_rows(rows, args.minimum_schedule_samples)
    except ValueError as error:
        parser.error(str(error))
    model_rows = [row for row in rows if row.stage in {"shape", "schedule"}]
    validation = cross_validate(model_rows)
    write_summary(args.summary, rows)
    write_report(args.report, args.input, rows, validation)
    print(f"aggregated {len(rows)} candidates")
    print(f"summary written to {args.summary}")
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
