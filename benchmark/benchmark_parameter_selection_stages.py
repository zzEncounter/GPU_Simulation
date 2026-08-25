"""Paired E2E benchmark for the three parameter-selection stages.

The stages deliberately separate two classes of choices:

``all_fixed``
    One conservative compile-time policy for every circuit and qubit count.
``structure_selected``
    Production structure/diagonal policy, but F128r2/B128r2 and canonical
    phase plans for every scenario.
``fully_selected``
    Production structure/diagonal policy plus the selected direction-specific
    execution geometry and phase plans.

All three policies are measured next to each other in every repetition.  Their
order is rotated so that a slow policy is not systematically measured first or
last.  The complete energy and gradient are checked against ``fully_selected``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402

import benchmark_sad  # noqa: E402


RAW_OUTPUT = ROOT / "benchmark" / "results" / "parameter_selection_stages_raw.csv"
SUMMARY_OUTPUT = ROOT / "benchmark" / "results" / "parameter_selection_stages.csv"
POLICIES = ("all_fixed", "structure_selected", "fully_selected")

# This is an operational conservative baseline on the current source tree, not
# a claim about the exact flags of an earlier Git revision.  Every searched
# axis is explicit so future default changes cannot silently move the baseline.
ALL_FIXED_FLAGS = (
    "-DSAD_FORWARD_BLOCK_THREADS=128",
    "-DSAD_FORWARD_REGISTER_BITS=2",
    "-DSAD_BLOCK_THREADS=128",
    "-DSAD_REGISTER_BITS=2",
    "-DSAD_FORWARD_FIXED_LOW_LANES=0",
    "-DSAD_FIXED_LOW_LANES=0",
    "-DSAD_MAILBOX_CHUNKS=1",
    "-DSAD_ORDINARY_BLOCK_THREADS=128",
    "-DSAD_DIAGONAL_BLOCK_THREADS=64",
    "-DSAD_SHARED_DIAGONAL_BLOCK_THREADS=128",
    "-DSAD_DIAGONAL_LOOKUP_BITS=8",
    "-DSAD_DIAGONAL_WARP_ATOMIC=0",
    "-DSAD_CNOT_FORWARD_SCATTER=0",
    "-DSAD_REAL_AMPLITUDE=0",
    "-DSAD_RA_FORWARD_FUSED=0",
    "-DSAD_RA_BACKWARD_FUSED=0",
    "-DSAD_SU2_FORWARD_STRATEGY=0",
    "-DSAD_SU2_BACKWARD_STRATEGY=0",
    "-DSAD_RZZ_FORWARD_FUSED=0",
    "-DSAD_RZZ_BACKWARD_STRATEGY=0",
    "-DSAD_QAOA_INITIAL_STRATEGY=1",
    "-DSAD_QAOA_FUSE_COST_RX=0",
    "-DSAD_QAOA_COMPACT_LOOKUP=0",
    "-DSAD_QAOA_FUSED_BACKWARD=0",
    "-DSAD_XXZ_CROSS_MATCHING=0",
)

RAW_FIELDS = (
    "timestamp_utc",
    "source_fingerprint",
    "repetition",
    "order",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "steps",
    "warmup_steps",
    "policy",
    "kernel_variant",
    "forward_phase_plan",
    "backward_phase_plan",
    "median_ms",
    "forward_mean_ms",
    "hamiltonian_mean_ms",
    "backward_mean_ms",
    "energy_abs_error_vs_best",
    "gradient_max_abs_error_vs_best",
    "correct",
    "library",
)

SUMMARY_FIELDS = (
    "experiment_group",
    "hardware",
    "circuit",
    "qubits",
    "layers",
    "precision",
    "paired_samples",
    "all_fixed_policy",
    "all_fixed_variant",
    "all_fixed_median_ms",
    "structure_selected_policy",
    "structure_selected_variant",
    "structure_selected_median_ms",
    "fully_selected_policy",
    "fully_selected_variant",
    "fully_selected_forward_phase_plan",
    "fully_selected_backward_phase_plan",
    "fully_selected_median_ms",
    "structure_speedup_vs_all_fixed",
    "execution_speedup_vs_structure_selected",
    "fully_selected_speedup_vs_all_fixed",
    "fully_selected_speedup_mad",
    "correct",
    "max_energy_abs_error",
    "max_gradient_abs_error",
    "source_fingerprint",
    "source_raw",
)


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h"):
        for path in sorted(SAD_ROOT.glob(pattern)):
            digest.update(str(path.relative_to(SAD_ROOT)).encode())
            digest.update(path.read_bytes())
    for path in (Path(__file__), Path(benchmark_sad.__file__)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _all_fixed_library() -> Path:
    flags_hash = hashlib.sha256("\0".join(ALL_FIXED_FLAGS).encode()).hexdigest()[:12]
    return SAD_ROOT / "build" / f"libsad_all_fixed_{flags_hash}.so"


def _build_all_fixed(path: Path) -> None:
    source_mtime = max(
        file.stat().st_mtime
        for pattern in ("src/**/*.cu", "src/**/*.cuh", "include/**/*.h")
        for file in SAD_ROOT.glob(pattern)
    )
    if path.exists() and path.stat().st_mtime >= source_mtime:
        return
    subprocess.run(
        (
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={path.relative_to(SAD_ROOT)}",
            f"EXTRA_NVCCFLAGS={' '.join(ALL_FIXED_FLAGS)}",
        ),
        cwd=ROOT,
        check=True,
    )


def repetitions_for_qubits(qubits: int) -> int:
    return 3 if qubits <= 24 else 2


def _set_policy(policy: str, all_fixed_library: Path) -> None:
    if policy == "all_fixed":
        os.environ["SAD_LIBRARY_PATH"] = str(all_fixed_library)
        os.environ["SAD_DISABLE_VARIANT_DISPATCH"] = "1"
    elif policy == "structure_selected":
        os.environ.pop("SAD_LIBRARY_PATH", None)
        os.environ["SAD_DISABLE_VARIANT_DISPATCH"] = "1"
    elif policy == "fully_selected":
        os.environ.pop("SAD_LIBRARY_PATH", None)
        os.environ.pop("SAD_DISABLE_VARIANT_DISPATCH", None)
    else:
        raise ValueError(policy)
    os.environ["SAD_EXECUTION_MODE"] = "optimized"


def _run(circuit: str, qubits: int, policy: str, library: Path) -> object:
    _set_policy(policy, library)
    return energy_and_grad(
        circuit=circuit,
        random_seed=benchmark_sad.RANDOM_SEED,
        scalability=(qubits, benchmark_sad.LAYERS),
        batches=benchmark_sad.BATCHES,
        precision=benchmark_sad.PRECISION,
        steps=benchmark_sad.steps_for_qubits(qubits),
        warmup_steps=1,
        device_name=benchmark_sad.DEVICE_NAME,
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
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[
            (
                row["circuit"],
                int(row["qubits"]),
                int(row["layers"]),
                row["source_fingerprint"],
            )
        ].append(row)

    rows: list[dict[str, object]] = []
    for (circuit, qubits, layers, fingerprint), samples in sorted(grouped.items()):
        by_repetition: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        for sample in samples:
            by_repetition[int(sample["repetition"])][sample["policy"]] = sample
        complete = [
            policies
            for policies in by_repetition.values()
            if set(policies) == set(POLICIES)
        ]
        if not complete:
            continue
        expected = repetitions_for_qubits(qubits)
        if len(complete) != expected:
            continue
        ratios_structure = []
        ratios_execution = []
        ratios_total = []
        for policies in complete:
            fixed = float(policies["all_fixed"]["median_ms"])
            structure = float(policies["structure_selected"]["median_ms"])
            best = float(policies["fully_selected"]["median_ms"])
            ratios_structure.append(fixed / structure)
            ratios_execution.append(structure / best)
            ratios_total.append(fixed / best)
        representative = complete[-1]
        all_samples = [sample for policies in complete for sample in policies.values()]
        rows.append(
            {
                "experiment_group": "parameter_selection_stages",
                "hardware": "NVIDIA RTX 6000 Ada",
                "circuit": circuit,
                "qubits": qubits,
                "layers": layers,
                "precision": representative["fully_selected"]["precision"],
                "paired_samples": len(complete),
                "all_fixed_policy": "uniform compile-time conservative switches",
                "all_fixed_variant": representative["all_fixed"]["kernel_variant"],
                "all_fixed_median_ms": statistics.median(
                    float(policies["all_fixed"]["median_ms"])
                    for policies in complete
                ),
                "structure_selected_policy": "production structure; fixed execution",
                "structure_selected_variant": representative["structure_selected"]["kernel_variant"],
                "structure_selected_median_ms": statistics.median(
                    float(policies["structure_selected"]["median_ms"])
                    for policies in complete
                ),
                "fully_selected_policy": "production structure and execution dispatch",
                "fully_selected_variant": representative["fully_selected"]["kernel_variant"],
                "fully_selected_forward_phase_plan": representative["fully_selected"]["forward_phase_plan"],
                "fully_selected_backward_phase_plan": representative["fully_selected"]["backward_phase_plan"],
                "fully_selected_median_ms": statistics.median(
                    float(policies["fully_selected"]["median_ms"])
                    for policies in complete
                ),
                "structure_speedup_vs_all_fixed": statistics.median(ratios_structure),
                "execution_speedup_vs_structure_selected": statistics.median(ratios_execution),
                "fully_selected_speedup_vs_all_fixed": statistics.median(ratios_total),
                "fully_selected_speedup_mad": _mad(ratios_total),
                "correct": int(all(sample["correct"] == "1" for sample in all_samples)),
                "max_energy_abs_error": max(
                    float(sample["energy_abs_error_vs_best"])
                    for sample in all_samples
                ),
                "max_gradient_abs_error": max(
                    float(sample["gradient_max_abs_error_vs_best"])
                    for sample in all_samples
                ),
                "source_fingerprint": fingerprint,
                "source_raw": str(raw_path.relative_to(ROOT)),
            }
        )

    expected_keys = {
        (circuit, qubits)
        for circuit in benchmark_sad.CIRCUITS
        for qubits in benchmark_sad.QUBITS
    }
    actual_keys = {(str(row["circuit"]), int(row["qubits"])) for row in rows}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"incomplete stage summary: missing={missing}, extra={extra}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--output", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()

    library = _all_fixed_library()
    _build_all_fixed(library)
    fingerprint = _source_fingerprint()
    total = sum(
        repetitions_for_qubits(qubits)
        for _ in benchmark_sad.CIRCUITS
        for qubits in benchmark_sad.QUBITS
    )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, lineterminator="\n")
        writer.writeheader()
        completed = 0
        max_repetitions = max(repetitions_for_qubits(q) for q in benchmark_sad.QUBITS)
        for repetition in range(max_repetitions):
            scenarios = [
                (circuit, qubits)
                for circuit in benchmark_sad.CIRCUITS
                for qubits in benchmark_sad.QUBITS
                if repetition < repetitions_for_qubits(qubits)
            ]
            random.Random(20260817 + repetition).shuffle(scenarios)
            for scenario_index, (circuit, qubits) in enumerate(scenarios):
                shift = (repetition + scenario_index) % len(POLICIES)
                order = POLICIES[shift:] + POLICIES[:shift]
                measured: dict[str, object] = {}
                for policy in order:
                    measured[policy] = _run(circuit, qubits, policy, library)
                best = measured["fully_selected"]
                for policy in POLICIES:
                    result = measured[policy]
                    energy_error = abs(result.energy - best.energy)
                    gradient_error = float(np.max(np.abs(result.grad - best.grad)))
                    writer.writerow(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "source_fingerprint": fingerprint,
                            "repetition": repetition,
                            "order": "-".join(order),
                            "circuit": circuit,
                            "qubits": qubits,
                            "layers": benchmark_sad.LAYERS,
                            "precision": benchmark_sad.PRECISION,
                            "steps": benchmark_sad.steps_for_qubits(qubits),
                            "warmup_steps": 1,
                            "policy": policy,
                            "kernel_variant": result.kernel_variant,
                            "forward_phase_plan": result.forward_phase_plan,
                            "backward_phase_plan": result.backward_phase_plan,
                            "median_ms": _median_ms(result.step_times_s),
                            "forward_mean_ms": _mean_ms(result.forward_times_s),
                            "hamiltonian_mean_ms": _mean_ms(result.hamiltonian_times_s),
                            "backward_mean_ms": _mean_ms(result.backward_times_s),
                            "energy_abs_error_vs_best": energy_error,
                            "gradient_max_abs_error_vs_best": gradient_error,
                            "correct": int(energy_error <= 1e-10 and gradient_error <= 1e-9),
                            "library": (
                                str(library.relative_to(ROOT))
                                if policy == "all_fixed"
                                else result.kernel_variant
                            ),
                        }
                    )
                stream.flush()
                completed += 1
                fixed_ms = _median_ms(measured["all_fixed"].step_times_s)
                structure_ms = _median_ms(measured["structure_selected"].step_times_s)
                best_ms = _median_ms(best.step_times_s)
                print(
                    f"[{completed:03d}/{total:03d}] {circuit} {qubits}q rep "
                    f"{repetition + 1}: structure={fixed_ms / structure_ms:.3f}x, "
                    f"execution={structure_ms / best_ms:.3f}x, "
                    f"total={fixed_ms / best_ms:.3f}x",
                    flush=True,
                )
                gc.collect()
    os.environ.pop("SAD_LIBRARY_PATH", None)
    os.environ.pop("SAD_DISABLE_VARIANT_DISPATCH", None)
    _write_summary(args.raw_output, args.output)
    print(f"Raw CSV written to {args.raw_output}")
    print(f"Summary CSV written to {args.output}")


if __name__ == "__main__":
    main()
