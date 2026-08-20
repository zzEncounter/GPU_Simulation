"""Staged, resumable parameter search for the QAOA-BD circuit.

Unlike QAOA-NS, QAOA-BD's cost layer is implemented by the matching CNOT-RZ-
CNOT kernels.  The search covers the cost-path fusion switch, RX forward and
backward shape (threads/register bits), ordinary kernel block size, and the
shared reduction axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from search_circuit_common import (
    ORDINARY_GRID, REGISTER_GRID, ROOT, SHAPE_GRID, best_config, make_candidate,
    phase_plan, run_search, valid_shape,
)

# One warp is legal for the ordinary permutation/matching kernels and is a
# useful low-qubit candidate, so BD explicitly includes it in the search.
BD_ORDINARY_GRID = (32,) + tuple(item for item in ORDINARY_GRID if item != 32)


def parse_qubits(value: str) -> tuple[int, ...]:
    """Parse unique even qubit counts in the supported BD benchmark range."""
    try:
        result = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("qubits must be comma-separated integers") from exc
    if not result or any(q < 4 or q > 28 or q % 2 for q in result):
        raise argparse.ArgumentTypeError("qubits must be even comma-separated values in 4..28")
    return result


def baseline() -> dict[str, int]:
    # Shape fields configure the RX mixer; ordinary_threads configures the BD
    # matching/CNOT kernels.
    return {
        "forward_threads": 128,
        "forward_register_bits": 2,
        "backward_threads": 128,
        "backward_register_bits": 2,
        "ordinary_threads": 128,
        "diagonal_threads": 64,
        "shared_diagonal_threads": 128,
        "lookup_bits": 8,
        "mailbox_chunks": 1,
        "legacy_reduction": 1,
        "rotation_warp_atomic": 0,
        "diagonal_warp_atomic": 0,
        "qaoa_bd_fusion": 1,
    }


def candidates(qubits: tuple[int, ...], csv_path: Path):
    """Yield candidates in dependency order, one independent chain per q."""
    for q in qubits:
        base = baseline()

        # Stage 0: establish whether the fused BD cost path wins.  Keeping
        # this first makes all later stages use the selected implementation.
        for fusion in (0, 1):
            yield make_candidate("fusion", q, {**base, "qaoa_bd_fusion": fusion})
        base, _, _ = best_config(csv_path, q, "fusion", "median_ms", base)

        # Stage 1: forward RX shape.  Register bits 2, 3 and 4 are all valid
        # and change the amount of per-thread rotation work.
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if not valid_shape(threads, registers):
                    continue
                yield make_candidate("forward_shape", q, {
                    **base, "forward_threads": threads,
                    "forward_register_bits": registers,
                }, forward_phase_plan=phase_plan(q, threads, registers, "compact"))
        base, f_plan, _ = best_config(csv_path, q, "forward_shape", "forward_ms", base)

        # Stage 2: backward RX shape, selected independently from forward.
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if not valid_shape(threads, registers):
                    continue
                yield make_candidate("backward_shape", q, {
                    **base, "backward_threads": threads,
                    "backward_register_bits": registers,
                }, forward_phase_plan=f_plan,
                                     backward_phase_plan=phase_plan(q, threads, registers, "compact"))
        base, f_plan, b_plan = best_config(csv_path, q, "backward_shape", "backward_ms", base)

        # Stage 3: block size used by pair-CNOT, matching RZ and initialisation.
        for threads in BD_ORDINARY_GRID:
            yield make_candidate("ordinary_threads", q,
                                 {**base, "ordinary_threads": threads},
                                 forward_phase_plan=f_plan,
                                 backward_phase_plan=b_plan)
        base, _, _ = best_config(csv_path, q, "ordinary_threads", "median_ms", base)

        # Stage 4: shared reduction settings still affect the RX mixer in every
        # layer, so retain the common reduction alternatives as a final pass.
        for legacy, warp_atomic in ((1, 0), (0, 0), (0, 1)):
            yield make_candidate("reduction", q, {
                **base,
                "legacy_reduction": legacy,
                "rotation_warp_atomic": warp_atomic,
            }, forward_phase_plan=f_plan, backward_phase_plan=b_plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=parse_qubits,
                        default=(4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark/results/qaoa_bd_parameter_search.csv")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "benchmark/results/qaoa_bd_parameter_search.json")
    args = parser.parse_args()
    run_search(circuit="qaoa-bd", candidates=candidates(args.qubits, args.output),
               csv_path=args.output, json_path=args.summary, layers=args.layers,
               steps=args.steps, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
