"""Run the staged, resumable kernel-parameter search for MERA."""

from __future__ import annotations

import argparse
from pathlib import Path

from search_circuit_common import (
    ORDINARY_GRID, REGISTER_GRID, ROOT, SHAPE_GRID, best_config, make_candidate,
    run_search, valid_shape,
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
        layers = (q - 1).bit_length()
        base = baseline()
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if not valid_shape(threads, registers):
                    continue
                config = {**base, "forward_threads": threads,
                          "forward_register_bits": registers}
                item = make_candidate("forward_shape", q, config)
                item["layers"] = layers
                yield item
        base, _, _ = best_config(csv_path, q, "forward_shape", "forward_ms", base)
        for threads in SHAPE_GRID:
            for registers in REGISTER_GRID:
                if not valid_shape(threads, registers):
                    continue
                config = {**base, "backward_threads": threads,
                          "backward_register_bits": registers}
                item = make_candidate("backward_shape", q, config)
                item["layers"] = layers
                yield item
        base, _, _ = best_config(csv_path, q, "backward_shape", "backward_ms", base)
        for threads in ORDINARY_GRID:
            item = make_candidate("ordinary_threads", q,
                                  {**base, "ordinary_threads": threads})
            item["layers"] = layers
            yield item
        base, _, _ = best_config(csv_path, q, "ordinary_threads", "hamiltonian_ms", base)
        for legacy, warp_atomic in ((1, 0), (0, 0), (0, 1)):
            config = {**base, "legacy_reduction": legacy,
                      "rotation_warp_atomic": warp_atomic}
            item = make_candidate("reduction", q, config)
            item["layers"] = layers
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=parse_qubits, default=(8, 12, 16, 20, 24, 28))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark/results/mera_parameter_search.csv")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "benchmark/results/mera_parameter_search.json")
    args = parser.parse_args()
    run_search(circuit="mera", candidates=candidates(args.qubits, args.output),
               csv_path=args.output, json_path=args.summary, layers=1,
               steps=args.steps, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
