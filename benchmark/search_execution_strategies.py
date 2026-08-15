"""Hierarchical search over rotation phase layouts and CUDA launch shapes.

The full Cartesian product is unnecessarily repetitive: exchanging two lane
slots (or two register slots) generates the same device instruction sequence.
This search therefore traverses the hardware-equivalence classes
``(lane targets, register targets, warp targets)``.  It measures every launch
shape on representative whole-layer schedules, calibrates the three target
classes for the survivors, and uses a k-best dynamic program to score all
reachable phase totals before benchmarking the predicted Pareto frontier.

The raw CSV is append-only and keyed, so a long run can be resumed safely.
Use ``--preset quick`` for a smoke run and ``--preset exhaustive`` for the
intended GPU characterization run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import os
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark" / "microbench_rotation.cu"
INCLUDE = ROOT / "sad" / "src"
DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "execution_search_raw.csv"
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
BUILD_CACHE = ROOT / "sad" / "build" / "execution_search"

MICRO_FIELDS = (
    "gate",
    "layout",
    "direction",
    "qubits",
    "threads",
    "register_amplitudes",
    "tile_bits",
    "phase_count",
    "gate_count",
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "active_cta_per_sm",
    "average_ms",
    "ms_per_gate",
    "mailbox_bytes",
    "mailbox_chunks",
    "scalar_mailbox",
    "persistent",
    "legacy_reduction",
    "local_bytes_per_thread",
    "multiprocessors",
    "phase_targets",
    "phase_lane_targets",
    "phase_register_targets",
    "phase_warp_targets",
)
PREFIX_FIELDS = (
    "stage",
    "variant",
    "family",
    "candidate",
    "repetition",
    "iterations",
)
FIELDS = (*PREFIX_FIELDS, *MICRO_FIELDS)


@dataclass(frozen=True, order=True)
class Variant:
    threads: int
    register_bits: int
    mailbox_chunks: int = 1

    @property
    def warp_bits(self) -> int:
        return int(math.log2(self.threads // 32))

    @property
    def tile_bits(self) -> int:
        return 5 + self.register_bits + self.warp_bits

    @property
    def name(self) -> str:
        return f"t{self.threads}r{self.register_bits}m{self.mailbox_chunks}"

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            f"-DSAD_FORWARD_BLOCK_THREADS={self.threads}",
            f"-DSAD_FORWARD_REGISTER_BITS={self.register_bits}",
            f"-DSAD_BLOCK_THREADS={self.threads}",
            f"-DSAD_REGISTER_BITS={self.register_bits}",
            f"-DSAD_MAILBOX_CHUNKS={self.mailbox_chunks}",
            "-DSAD_ROTATION_PERSISTENT=0",
        )


@dataclass(frozen=True, order=True)
class Phase:
    lane: int
    register: int
    warp: int

    @property
    def targets(self) -> int:
        return self.lane + self.register + self.warp

    def encode(self) -> str:
        return f"L{self.lane}R{self.register}W{self.warp}"


@dataclass(frozen=True)
class Schedule:
    family: str
    phases: tuple[Phase, ...]
    predicted_cost: float

    @property
    def layout(self) -> str:
        return f"plan:{self.family}:" + "-".join(
            phase.encode() for phase in self.phases
        )

    @property
    def candidate(self) -> str:
        return "/".join(phase.encode() for phase in self.phases)


@dataclass(frozen=True)
class Preset:
    qubits: tuple[int, ...]
    variants: tuple[Variant, ...]
    repetitions: int
    iterations: int
    shape_survivors: int
    schedules_per_family: int
    extra_phases: int


def _powers_of_two(limit: int) -> tuple[int, ...]:
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return tuple(values)


def all_variants(*, mailbox_limit: int = 64) -> tuple[Variant, ...]:
    variants: list[Variant] = []
    for threads in (32, 64, 128, 256, 512):
        warp_bits = int(math.log2(threads // 32))
        for register_bits in range(2, 7):
            tile_bits = 5 + register_bits + warp_bits
            if tile_bits > 12:
                continue
            chunks = (
                (1,)
                if threads == 32
                else _powers_of_two(min(1 << register_bits, mailbox_limit))
            )
            variants.extend(
                Variant(threads, register_bits, chunk)
                for chunk in chunks
                # A double-complex tile-12 backward mailbox is 64 KiB before
                # chunking and exceeds the 48 KiB static-shared ELF limit.
                # m>=2 makes these previously excluded tiles compilable.
                if tile_bits < 12 or chunk >= 2
            )
    return tuple(variants)


def quick_variants() -> tuple[Variant, ...]:
    return (
        Variant(32, 4, 1),
        Variant(64, 3, 1),
        Variant(64, 4, 1),
        Variant(64, 4, 2),
        Variant(128, 2, 1),
        Variant(128, 3, 2),
    )


PRESETS = {
    "quick": Preset(
        qubits=(20, 24),
        variants=quick_variants(),
        repetitions=2,
        iterations=4,
        shape_survivors=3,
        schedules_per_family=3,
        extra_phases=0,
    ),
    "standard": Preset(
        qubits=(12, 16, 20, 24, 26),
        # Keep every legal power-of-two mailbox partition in the primary
        # search.  Large chunks are usually dominated, but measuring them is
        # cheap relative to the full schedule search and makes the resource
        # boundary explicit instead of encoding an assumed cutoff.
        variants=all_variants(mailbox_limit=64),
        repetitions=3,
        iterations=8,
        shape_survivors=5,
        schedules_per_family=6,
        extra_phases=1,
    ),
    "exhaustive": Preset(
        qubits=(12, 16, 20, 22, 24, 26, 28),
        variants=all_variants(mailbox_limit=64),
        repetitions=5,
        iterations=12,
        shape_survivors=8,
        schedules_per_family=12,
        extra_phases=1,
    ),
}


def minimum_phase_count(qubits: int, variant: Variant, family: str) -> int:
    if family == "compact":
        first = later = variant.tile_bits
    elif family == "fixed":
        first, later = variant.tile_bits, variant.tile_bits - 5
    elif family == "pairs":
        if variant.warp_bits == 0:
            raise ValueError("pair-contiguous layout needs more than one warp")
        first, later = variant.tile_bits, variant.tile_bits - 6
    else:
        raise ValueError(f"unknown phase family {family!r}")
    return 1 if qubits <= first else 1 + math.ceil((qubits - first) / later)


def phase_options(
    variant: Variant, family: str, phase_index: int
) -> tuple[Phase, ...]:
    reserve_low = family in {"fixed", "pairs"} and phase_index > 0
    reserve_pair = family == "pairs" and phase_index > 0
    lane_limit = 0 if reserve_low else 5
    warp_limit = variant.warp_bits - int(reserve_pair)
    if warp_limit < 0:
        return ()
    result = tuple(
        Phase(lane, register, warp)
        for lane in range(lane_limit + 1)
        for register in range(variant.register_bits + 1)
        for warp in range(warp_limit + 1)
        if lane + register + warp > 0
    )
    return result


def default_phase_cost(phase: Phase, variant: Variant) -> float:
    """Architecture-neutral prior used until calibration data is available."""

    underfill = variant.tile_bits - phase.targets
    return (
        1.0
        + 0.018 * phase.lane
        + 0.014 * phase.register
        + (0.09 + 0.012 * variant.mailbox_chunks) * phase.warp
        + 0.003 * underfill
    )


def k_best_schedules(
    qubits: int,
    variant: Variant,
    family: str,
    *,
    phase_count: int,
    keep: int,
    phase_cost: Callable[[Phase, int], float] | None = None,
) -> tuple[Schedule, ...]:
    """Return k best hardware-equivalence schedules via dynamic programming.

    Every reachable target total and every phase-class transition is visited.
    Only the best ``keep`` histories for an identical ``(phase, total)`` state
    are retained; histories outside that beam cannot affect reachability and
    are deliberately treated as near-ties rather than separately benchmarked.
    """

    if keep < 1:
        raise ValueError("keep must be positive")
    scorer = phase_cost or (lambda phase, _: default_phase_cost(phase, variant))
    states: dict[int, list[tuple[float, tuple[Phase, ...]]]] = {0: [(0.0, ())]}
    for phase_index in range(phase_count):
        next_states: dict[int, list[tuple[float, tuple[Phase, ...]]]] = {}
        options = phase_options(variant, family, phase_index)
        remaining_phases = phase_count - phase_index - 1
        for total, histories in states.items():
            for option in options:
                if (
                    phase_index == 0
                    and family in {"fixed", "pairs"}
                    and option.targets < 5 + int(family == "pairs")
                ):
                    continue
                new_total = total + option.targets
                if new_total > qubits:
                    continue
                remaining = qubits - new_total
                if remaining < remaining_phases:
                    continue
                for cost, phases in histories:
                    bucket = next_states.setdefault(new_total, [])
                    bucket.append(
                        (cost + scorer(option, phase_index), phases + (option,))
                    )
        states = {}
        for total, histories in next_states.items():
            histories.sort(key=lambda item: (item[0], item[1]))
            states[total] = histories[:keep]
    candidates = []
    for cost, phases in states.get(qubits, []):
        if family in {"fixed", "pairs"}:
            required = 5 + int(family == "pairs")
            if phases[0].targets < required:
                continue
        candidates.append(Schedule(family, phases, cost))
    candidates.sort(key=lambda item: (item.predicted_cost, item.candidate))
    return tuple(candidates[:keep])


def candidate_schedules(
    qubits: int,
    variant: Variant,
    *,
    per_family: int,
    extra_phases: int,
    phase_cost: Callable[[Phase, int], float] | None = None,
) -> tuple[Schedule, ...]:
    schedules: list[Schedule] = []
    for family in ("compact", "fixed", "pairs"):
        if family == "pairs" and variant.warp_bits == 0:
            continue
        minimum = minimum_phase_count(qubits, variant, family)
        for phase_count in range(minimum, minimum + extra_phases + 1):
            schedules.extend(
                k_best_schedules(
                    qubits,
                    variant,
                    family,
                    phase_count=phase_count,
                    keep=per_family,
                    phase_cost=phase_cost,
                )
            )
    unique = {schedule.layout: schedule for schedule in schedules}
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                len(item.phases), item.predicted_cost, item.family, item.candidate
            ),
        )
    )


def _parse_qubits(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item < 7 or item > 30 for item in result):
        raise argparse.ArgumentTypeError("qubits must be comma-separated 7..30")
    return result


def _latest_source_mtime() -> float:
    return max(
        file.stat().st_mtime
        for file in (SOURCE, *INCLUDE.glob("**/*.cuh"))
    )


def _cached_binary(variant: Variant, iterations: int) -> Path:
    digest = hashlib.sha256(
        "\0".join((*variant.flags, str(iterations))).encode()
    ).hexdigest()[:12]
    return BUILD_CACHE / f"{variant.name}-i{iterations}-{digest}"


def _compile(variant: Variant, binary: Path, iterations: int) -> None:
    if binary.exists() and binary.stat().st_mtime >= _latest_source_mtime():
        return
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            NVCC,
            "-O3",
            "-std=c++17",
            "-arch=native",
            "-lineinfo",
            f"-I{INCLUDE}",
            f"-DSAD_MICRO_ITERATIONS={iterations}",
            *variant.flags,
            str(SOURCE),
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )


def _row_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(row[field])
        for field in (
            "stage",
            "variant",
            "gate",
            "direction",
            "qubits",
            "layout",
            "repetition",
            "iterations",
        )
    )


def _load_completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {_row_key(row) for row in csv.DictReader(stream)}


def _measure(
    binary: Path,
    qubits: int,
    gate: str,
    layout: str,
    direction: str,
    target_spec: str | None = None,
    adaptive_mailbox: bool = False,
) -> dict[str, str]:
    command = [str(binary), str(qubits), gate, layout, direction]
    if target_spec is not None:
        command.append(target_spec)
    elif adaptive_mailbox:
        command.append("all")
    if adaptive_mailbox:
        command.append("adaptive")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        if (
            "too many resources requested for launch" in diagnostic
            or "invalid configuration argument" in diagnostic
        ):
            return {}
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    values = completed.stdout.strip().splitlines()[-1].split(",")
    if len(values) != len(MICRO_FIELDS):
        raise RuntimeError(
            f"unexpected microbenchmark output ({len(values)} fields):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return dict(zip(MICRO_FIELDS, values, strict=True))


def _shape_jobs(
    variants: Sequence[Variant], qubits: Sequence[int]
) -> Iterable[tuple[Variant, int, str, str, str, str, str | None]]:
    for variant, q, gate, direction in itertools.product(
        variants, qubits, ("rx", "ry"), ("forward", "backward")
    ):
        yield variant, q, gate, direction, "compact", "full", None
        yield variant, q, gate, direction, "fixed", "full-fixed", None
        if variant.warp_bits:
            yield variant, q, gate, direction, "pairs", "full-pairs", None


def _calibration_case(
    variant: Variant, q: int, gate: str, direction: str
) -> Iterable[tuple[Variant, int, str, str, str, str, str | None]]:
    classes = {
            "L0R0W0",
            "L1R0W0",
            "L5R0W0",
            "L0R1W0",
            f"L0R{variant.register_bits}W0",
            f"L5R{variant.register_bits}W0",
    }
    if variant.warp_bits:
        classes.update(
                {
                    "L0R0W1",
                    f"L0R0W{variant.warp_bits}",
                    f"L5R{variant.register_bits}W{variant.warp_bits}",
                }
        )
    for target_spec in sorted(classes):
        yield (
                variant,
                q,
                gate,
                direction,
                "compact",
                "low",
                "none" if target_spec == "L0R0W0" else target_spec,
        )
    fixed_classes = {
            "L0R0W0",
            "L0R1W0",
            f"L0R{variant.register_bits}W0",
    }
    if variant.warp_bits:
        fixed_classes.update(
                {
                    "L0R0W1",
                    f"L0R0W{variant.warp_bits}",
                    f"L0R{variant.register_bits}W{variant.warp_bits}",
                }
        )
    for target_spec in sorted(fixed_classes):
        yield (
                variant,
                q,
                gate,
                direction,
                "fixed",
                "fixed:5",
                "none" if target_spec == "L0R0W0" else target_spec,
        )


def _calibration_jobs(
    survivors: dict[tuple[str, str, int], tuple[Variant, ...]]
) -> Iterable[tuple[Variant, int, str, str, str, str, str | None]]:
    for (gate, direction, q), variants in sorted(survivors.items()):
        for variant in variants:
            yield from _calibration_case(variant, q, gate, direction)


def _median_rows(path: Path, stage: str) -> dict[tuple[str, ...], float]:
    samples: dict[tuple[str, ...], list[float]] = {}
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] != stage:
                continue
            key = (
                row["variant"],
                row["gate"],
                row["direction"],
                row["qubits"],
                row["family"],
                row["candidate"],
            )
            samples.setdefault(key, []).append(float(row["average_ms"]))
    return {key: statistics.median(values) for key, values in samples.items()}


def _select_survivors(
    output: Path, variants: Sequence[Variant], count: int
) -> tuple[Variant, ...]:
    medians = _median_rows(output, "shape")
    if not medians:
        return tuple(variants[:count])
    by_scenario: dict[tuple[str, str, str], list[tuple[float, str]]] = {}
    for key, value in medians.items():
        variant, gate, direction, qubits, _family, _candidate = key
        by_scenario.setdefault((gate, direction, qubits), []).append(
            (value, variant)
        )
    votes: dict[str, int] = {}
    normalized: dict[str, list[float]] = {}
    for values in by_scenario.values():
        best = min(item[0] for item in values)
        best_by_variant: dict[str, float] = {}
        for value, variant in values:
            best_by_variant[variant] = min(
                value, best_by_variant.get(variant, math.inf)
            )
        for rank, (variant, value) in enumerate(
            sorted(best_by_variant.items(), key=lambda item: item[1])
        ):
            if rank < count:
                votes[variant] = votes.get(variant, 0) + 1
            normalized.setdefault(variant, []).append(value / best)
    ranked = sorted(
        variants,
        key=lambda variant: (
            -votes.get(variant.name, 0),
            statistics.fmean(normalized.get(variant.name, [math.inf])),
            variant.name,
        ),
    )
    return tuple(ranked[:count])


def _select_scenario_survivors(
    output: Path, variants: Sequence[Variant], count: int
) -> dict[tuple[str, str, int], tuple[Variant, ...]]:
    medians = _median_rows(output, "shape")
    if not medians:
        return {}
    by_name = {variant.name: variant for variant in variants}
    by_scenario: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, value in medians.items():
        variant, gate, direction, qubits, _family, _candidate = key
        best = by_scenario.setdefault((gate, direction, int(qubits)), {})
        best[variant] = min(value, best.get(variant, math.inf))
    result: dict[tuple[str, str, int], tuple[Variant, ...]] = {}
    for scenario, values in by_scenario.items():
        ranked = sorted(values, key=lambda name: (values[name], name))
        best = values[ranked[0]]
        # Keep all statistically near-tied shapes even when that exceeds the
        # nominal count; the next stage is specifically meant to break ties.
        selected = [name for name in ranked if values[name] <= best * 1.02]
        if len(selected) < count:
            selected = ranked[:count]
        result[scenario] = tuple(by_name[name] for name in selected)
    return result


def _calibrated_costs(
    output: Path,
    variant: Variant,
    q: int,
    gate: str,
    direction: str,
    family: str,
) -> Callable[[Phase, int], float]:
    medians = _median_rows(output, "calibration")

    def lookup(family: str, candidate: str) -> float | None:
        return medians.get(
            (variant.name, gate, direction, str(q), family, candidate)
        )

    compact_base = lookup("compact", "none")
    fixed_base = lookup("fixed", "none")
    if compact_base is None:
        return lambda phase, _: default_phase_cost(phase, variant)

    def slope(
        calibration_family: str,
        candidate: str,
        count: int,
        fallback: float,
    ) -> float:
        value = lookup(calibration_family, candidate)
        base = compact_base if calibration_family == "compact" else fixed_base
        return (
            fallback
            if value is None or base is None
            else max(0.0, (value - base) / count)
        )

    lane = slope("compact", "L5R0W0", 5, 0.018 * compact_base)
    register = slope(
        "compact",
        f"L0R{variant.register_bits}W0",
        variant.register_bits,
        0.014 * compact_base,
    )
    warp = (
        slope(
            "compact",
            f"L0R0W{variant.warp_bits}",
            variant.warp_bits,
            0.1 * compact_base,
        )
        if variant.warp_bits
        else 0.0
    )
    fixed_register = slope(
        "fixed",
        f"L0R{variant.register_bits}W0",
        variant.register_bits,
        register,
    )
    fixed_warp = (
        slope(
            "fixed",
            f"L0R0W{variant.warp_bits}",
            variant.warp_bits,
            warp,
        )
        if variant.warp_bits
        else 0.0
    )

    def score(phase: Phase, phase_index: int) -> float:
        use_fixed = family in {"fixed", "pairs"} and phase_index > 0
        base = (fixed_base if fixed_base is not None else compact_base) \
            if use_fixed else compact_base
        pair_penalty = 0.01 * base if family == "pairs" and use_fixed else 0.0
        return (
            base
            + lane * phase.lane
            + (fixed_register if use_fixed else register) * phase.register
            + (fixed_warp if use_fixed else warp) * phase.warp
            + pair_penalty
        )

    return score


def _schedule_jobs(
    output: Path,
    survivors: dict[tuple[str, str, int], tuple[Variant, ...]],
    per_family: int,
    extra_phases: int,
) -> Iterable[tuple[Variant, int, str, str, str, str, str | None]]:
    for (gate, direction, q), variants in sorted(survivors.items()):
        for variant in variants:
            for family in ("compact", "fixed", "pairs"):
                if family == "pairs" and variant.warp_bits == 0:
                    continue
                scorer = _calibrated_costs(
                    output, variant, q, gate, direction, family
                )
                minimum = minimum_phase_count(q, variant, family)
                for phase_count in range(minimum, minimum + extra_phases + 1):
                    for schedule in k_best_schedules(
                        q,
                        variant,
                        family,
                        phase_count=phase_count,
                        keep=per_family,
                        phase_cost=scorer,
                    ):
                        yield (
                            variant,
                            q,
                            gate,
                            direction,
                            schedule.family,
                            schedule.layout,
                            None,
                        )


def _refine_schedule_jobs(
    output: Path,
    jobs: Iterable[
        tuple[Variant, int, str, str, str, str, str | None]
    ],
    *,
    minimum_per_scenario: int,
    relative_limit: float,
) -> tuple[tuple[Variant, int, str, str, str, str, str | None], ...]:
    """Keep measured schedule finalists for repeated confirmation.

    Every generated schedule receives one screening sample.  Later shuffled
    repetitions are reserved for the top-N and all candidates within the
    configured relative band, which is successive halving rather than a
    premature model-only prune.
    """

    if minimum_per_scenario < 1 or relative_limit < 1:
        raise ValueError("invalid schedule refinement limits")
    medians = _median_rows(output, "schedule")
    by_scenario: dict[
        tuple[str, str, int],
        list[tuple[float, tuple[Variant, int, str, str, str, str, str | None]]],
    ] = {}
    for job in dict.fromkeys(jobs):
        variant, q, gate, direction, family, layout, target_spec = job
        candidate = target_spec or layout.removeprefix(f"plan:{family}:")
        value = medians.get(
            (variant.name, gate, direction, str(q), family, candidate)
        )
        if value is not None:
            by_scenario.setdefault((gate, direction, q), []).append((value, job))
    selected: list[
        tuple[Variant, int, str, str, str, str, str | None]
    ] = []
    for candidates in by_scenario.values():
        candidates.sort(key=lambda item: (item[0], item[1]))
        best = candidates[0][0]
        keep = [job for value, job in candidates if value <= best * relative_limit]
        if len(keep) < minimum_per_scenario:
            keep = [job for _, job in candidates[:minimum_per_scenario]]
        selected.extend(keep)
    return tuple(dict.fromkeys(selected))


def _run_jobs(
    *,
    stage: str,
    jobs: Iterable[tuple[Variant, int, str, str, str, str, str | None]],
    binaries: dict[Variant, Path],
    output: Path,
    repetitions: int,
    iterations: int,
) -> None:
    completed_keys = _load_completed(output)
    exists = output.exists() and output.stat().st_size > 0
    unique_jobs = list(dict.fromkeys(jobs))
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(repetitions):
            ordered = list(unique_jobs)
            random.Random(f"sad-{stage}-{repetition}").shuffle(ordered)
            for variant, q, gate, direction, family, layout, target_spec in ordered:
                candidate = target_spec or layout.removeprefix(f"plan:{family}:")
                prospective: dict[str, object] = {
                    "stage": stage,
                    "variant": variant.name,
                    "family": family,
                    "candidate": candidate,
                    "repetition": repetition,
                    "iterations": iterations,
                    "gate": gate,
                    "direction": direction,
                    "qubits": q,
                    "layout": layout.replace(",", ";")
                    + (f":{target_spec}" if target_spec else ""),
                }
                if _row_key(prospective) in completed_keys:
                    continue
                measured = _measure(
                    binaries[variant], q, gate, layout, direction, target_spec
                )
                if not measured:
                    print(
                        f"skip illegal launch {variant.name} q={q} "
                        f"{gate}/{direction[0]} {family} {candidate}",
                        flush=True,
                    )
                    continue
                row = {**prospective, **measured}
                writer.writerow(row)
                stream.flush()
                completed_keys.add(_row_key(row))
                print(
                    f"{stage:11s} {variant.name:11s} q={q:2d} "
                    f"{gate}/{direction[0]} {family:7s} {candidate:30.30s} "
                    f"{float(measured['average_ms']):.6f} ms",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="standard"
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("shape", "calibration", "schedule"),
        help="repeat to select stages; default runs all stages",
    )
    parser.add_argument("--qubits", type=_parse_qubits)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--shape-survivors", type=int)
    parser.add_argument("--schedules-per-family", type=int)
    parser.add_argument("--extra-phases", type=int)
    parser.add_argument(
        "--variant",
        action="append",
        help="repeat to run only named variants from the selected preset",
    )
    parser.add_argument(
        "--refine-schedules",
        action="store_true",
        help=(
            "repeat only the measured per-scenario top schedules after a "
            "complete one-repetition screening pass"
        ),
    )
    parser.add_argument("--schedule-refine-top", type=int, default=5)
    parser.add_argument("--schedule-refine-percent", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--list-space",
        action="store_true",
        help="print the compile/search space without compiling or running CUDA",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    qubits = args.qubits or preset.qubits
    repetitions = args.repetitions or preset.repetitions
    iterations = args.iterations or preset.iterations
    survivor_count = args.shape_survivors or preset.shape_survivors
    schedules_per_family = (
        args.schedules_per_family or preset.schedules_per_family
    )
    extra_phases = (
        preset.extra_phases
        if args.extra_phases is None
        else args.extra_phases
    )
    if min(
        repetitions,
        iterations,
        survivor_count,
        schedules_per_family,
        args.schedule_refine_top,
    ) < 1:
        parser.error("counts and iterations must be positive")
    if extra_phases < 0:
        parser.error("extra phases cannot be negative")
    if args.schedule_refine_percent < 0:
        parser.error("schedule refine percent cannot be negative")

    stages = tuple(args.stage or ("shape", "calibration", "schedule"))
    variants = preset.variants
    if args.variant:
        by_name = {variant.name: variant for variant in variants}
        unknown = sorted(set(args.variant) - by_name.keys())
        if unknown:
            parser.error("unknown variants: " + ", ".join(unknown))
        variants = tuple(dict.fromkeys(by_name[name] for name in args.variant))
    if args.list_space:
        schedule_classes = sum(
            len(
                candidate_schedules(
                    q,
                    variant,
                    per_family=schedules_per_family,
                    extra_phases=extra_phases,
                )
            )
            for variant in variants
            for q in qubits
        )
        print(
            f"preset={args.preset} variants={len(variants)} "
            f"qubits={','.join(map(str, qubits))} "
            f"frontier_schedule_classes={schedule_classes}"
        )
        for variant in variants:
            print(
                f"{variant.name}: threads={variant.threads} "
                f"register_bits={variant.register_bits} "
                f"warp_bits={variant.warp_bits} tile_bits={variant.tile_bits} "
                f"mailbox_chunks={variant.mailbox_chunks}"
            )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binaries: dict[Variant, Path] = {}

    def compile_many(items: Sequence[Variant]) -> None:
        for variant in items:
            if variant in binaries:
                continue
            binary = _cached_binary(variant, iterations)
            print(f"compile {variant.name} ({' '.join(variant.flags)})")
            _compile(variant, binary, iterations)
            binaries[variant] = binary

    if "shape" in stages:
        compile_many(variants)
        _run_jobs(
            stage="shape",
            jobs=_shape_jobs(variants, qubits),
            binaries=binaries,
            output=args.output,
            repetitions=repetitions,
            iterations=iterations,
        )
    global_survivors = _select_survivors(
        args.output, variants, survivor_count
    )
    scenario_survivors = _select_scenario_survivors(
        args.output, variants, survivor_count
    )
    survivors = tuple(
        sorted(
            {
                variant
                for values in scenario_survivors.values()
                for variant in values
            },
            key=lambda item: item.name,
        )
    ) or global_survivors
    if not scenario_survivors:
        scenario_survivors = {
            (gate, direction, q): survivors
            for gate, direction, q in itertools.product(
                ("rx", "ry"), ("forward", "backward"), qubits
            )
        }
    print("survivors:", ", ".join(item.name for item in survivors))
    if "calibration" in stages:
        compile_many(survivors)
        _run_jobs(
            stage="calibration",
            jobs=_calibration_jobs(scenario_survivors),
            binaries=binaries,
            output=args.output,
            repetitions=repetitions,
            iterations=iterations,
        )
    if "schedule" in stages:
        schedule_jobs = tuple(
            _schedule_jobs(
                args.output,
                scenario_survivors,
                schedules_per_family,
                extra_phases,
            )
        )
        if args.refine_schedules:
            schedule_jobs = _refine_schedule_jobs(
                args.output,
                schedule_jobs,
                minimum_per_scenario=args.schedule_refine_top,
                relative_limit=1 + args.schedule_refine_percent / 100,
            )
            if not schedule_jobs:
                parser.error(
                    "--refine-schedules requires a complete schedule "
                    "screening pass in the output CSV"
                )
            print(f"refined schedule jobs: {len(schedule_jobs)}")
        compile_many(
            tuple(
                sorted(
                    {job[0] for job in schedule_jobs},
                    key=lambda item: item.name,
                )
            )
        )
        _run_jobs(
            stage="schedule",
            jobs=schedule_jobs,
            binaries=binaries,
            output=args.output,
            repetitions=repetitions,
            iterations=iterations,
        )
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
