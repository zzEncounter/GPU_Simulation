"""Run the staged, resumable kernel-parameter search for Equivariant-QNN."""

from __future__ import annotations

import argparse
from pathlib import Path

from search_circuit_common import (
    ORDINARY_GRID, REGISTER_GRID, ROOT, SHAPE_GRID, best_config, make_candidate,
    phase_plan, run_search, valid_shape,
)


def parse_qubits(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not result or any(q < 2 or q > 30 for q in result):
        raise argparse.ArgumentTypeError("qubits must be comma-separated values in 2..30")
    return result


def baseline() -> dict[str, int]:
    return {
        "forward_threads": 128, "forward_register_bits": 2,
        "backward_threads": 128, "backward_register_bits": 2,
        "ordinary_threads": 128, "diagonal_threads": 64,
        "shared_diagonal_threads": 128, "lookup_bits": 8,
        "mailbox_chunks": 1, "legacy_reduction": 1,
        "rotation_warp_atomic": 0, "diagonal_warp_atomic": 0,
    }


def candidates(qubits: tuple[int, ...], csv_path: Path):
    for q in qubits:
        base = baseline()
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if valid_shape(threads, registers):
                    yield make_candidate("forward_shape", q, {
                        **base, "forward_threads": threads,
                        "forward_register_bits": registers})
        base, _, _ = best_config(csv_path, q, "forward_shape", "forward_ms", base)
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if valid_shape(threads, registers):
                    yield make_candidate("backward_shape", q, {
                        **base, "backward_threads": threads,
                        "backward_register_bits": registers})
        base, _, _ = best_config(csv_path, q, "backward_shape", "backward_ms", base)
        f_plans = ("", phase_plan(q, base["forward_threads"],
                                  base["forward_register_bits"], "compact"),
                   phase_plan(q, base["forward_threads"],
                              base["forward_register_bits"], "fixed"))
        for plan in dict.fromkeys(f_plans):
            yield make_candidate("forward_phase", q, base, forward_phase_plan=plan)
        base, f_plan, _ = best_config(csv_path, q, "forward_phase", "forward_ms", base)
        b_plans = ("", phase_plan(q, base["backward_threads"],
                                  base["backward_register_bits"], "compact"),
                   phase_plan(q, base["backward_threads"],
                              base["backward_register_bits"], "fixed"))
        for plan in dict.fromkeys(b_plans):
            yield make_candidate("backward_phase", q, base,
                                 forward_phase_plan=f_plan, backward_phase_plan=plan)
        base, f_plan, b_plan = best_config(csv_path, q, "backward_phase", "backward_ms", base)
        for chunks in (1, 2):
            yield make_candidate("mailbox", q, {**base, "mailbox_chunks": chunks},
                                 forward_phase_plan=f_plan, backward_phase_plan=b_plan)
        base, f_plan, b_plan = best_config(csv_path, q, "mailbox", "median_ms", base)
        for threads in ORDINARY_GRID:
            yield make_candidate("ordinary_threads", q,
                                 {**base, "ordinary_threads": threads},
                                 forward_phase_plan=f_plan, backward_phase_plan=b_plan)
        base, f_plan, b_plan = best_config(csv_path, q, "ordinary_threads", "median_ms", base)
        for legacy, warp_atomic in ((1, 0), (0, 0), (0, 1)):
            yield make_candidate("reduction", q, {
                **base, "legacy_reduction": legacy,
                "rotation_warp_atomic": warp_atomic},
                forward_phase_plan=f_plan, backward_phase_plan=b_plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=parse_qubits, default=(4, 8, 12, 16, 20, 24, 28))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark/results/equivariant_qnn_parameter_search.csv")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "benchmark/results/equivariant_qnn_parameter_search.json")
    args = parser.parse_args()
    run_search(circuit="equivariant-qnn",
               candidates=candidates(args.qubits, args.output),
               csv_path=args.output, json_path=args.summary, layers=args.layers,
               steps=args.steps, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
