"""Compare standalone training runs across gradient strategies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising import RunConfig, run_standalone
from ring_ising.config import STANDALONE_GRADIENT_STRATEGIES

VALID_MODES = STANDALONE_GRADIENT_STRATEGIES


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 5x8."
        ) from exc


def default_steps_for(case: BenchmarkCase) -> int:
    base_steps_by_layers = {8: 120, 32: 80, 128: 40, 512: 12, 2048: 4}
    base = base_steps_by_layers.get(case.layers, max(2, 960 // max(case.layers, 1)))
    return max(2, int(round(base * 6.0 / case.num_qubits)))


def parse_mode(text: str) -> str:
    mode = text.strip()
    if mode not in VALID_MODES:
        choices = ", ".join(VALID_MODES)
        raise argparse.ArgumentTypeError(
            f"Invalid mode {text!r}. Expected one of: {choices}"
        )
    return mode


def build_run_config(
    mode: str,
    *,
    case: BenchmarkCase,
    steps: int,
    args: argparse.Namespace,
) -> RunConfig:
    return RunConfig(
        backend="standalone",
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=args.field,
        steps=steps,
        stepsize=args.stepsize,
        seed=args.seed,
        init_scale=args.init_scale,
        gradient_strategy=mode,
        verbose=False,
        show_progress=False,
        gpu_telemetry=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[
            parse_case("12x8"),
            parse_case("12x32"),
        ],
        help="Problem sizes in QxL format.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        type=parse_mode,
        default=["inverse_walk"],
        help="Standalone gradient strategies to compare.",
    )
    parser.add_argument(
        "--reference-mode",
        type=parse_mode,
        default="inverse_walk",
        help="Mode used as the timing and energy reference.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--stepsize", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    print("Standalone gradient strategy comparison")
    print(f"  Modes: {', '.join(args.modes)}")
    print(f"  Reference mode: {args.reference_mode}")
    print()

    for case in args.cases:
        steps = default_steps_for(case)
        case_rows: list[dict[str, object]] = []
        for mode in args.modes:
            if mode == "dense_scan" and case.num_qubits > 6:
                print(
                    f"Skipping dense_scan for {case.num_qubits} qubits x {case.layers} layers "
                    "(dense_scan requires qubits <= 6)"
                )
                continue
            result = run_standalone(build_run_config(mode, case=case, steps=steps, args=args))
            case_rows.append(
                {
                    "num_qubits": case.num_qubits,
                    "layers": case.layers,
                    "steps": steps,
                    "mode": mode,
                    "avg_step_ms": result.wall_s * 1000.0 / steps,
                    "final_energy": float(result.final_energy),
                }
            )

        if not case_rows:
            print(f"{case.num_qubits} qubits x {case.layers} layers, steps={steps}")
            print("  no runnable modes for this case")
            print()
            continue

        reference = next(
            (row for row in case_rows if row["mode"] == args.reference_mode),
            case_rows[0],
        )
        if reference["mode"] != args.reference_mode:
            print(
                f"  reference mode {args.reference_mode} not runnable for this case; "
                f"falling back to {reference['mode']}"
            )
        ref_energy = float(reference["final_energy"])
        ref_step_ms = float(reference["avg_step_ms"])
        for row in case_rows:
            row["speedup_vs_reference"] = (
                ref_step_ms / float(row["avg_step_ms"])
                if float(row["avg_step_ms"]) > 0.0
                else None
            )
            row["final_energy_abs_diff_vs_reference"] = abs(
                float(row["final_energy"]) - ref_energy
            )
            rows.append(row)

        print(f"{case.num_qubits} qubits x {case.layers} layers, steps={steps}")
        for row in case_rows:
            print(
                f"  {row['mode']:<18} "
                f"avg_step_ms={float(row['avg_step_ms']):.3f} "
                f"speedup={float(row['speedup_vs_reference']):.3f}x "
                f"energy_diff={float(row['final_energy_abs_diff_vs_reference']):.3e}"
            )
        print()

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "num_qubits",
                    "layers",
                    "steps",
                    "mode",
                    "avg_step_ms",
                    "speedup_vs_reference",
                    "final_energy",
                    "final_energy_abs_diff_vs_reference",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {args.csv_out}")


if __name__ == "__main__":
    main()
