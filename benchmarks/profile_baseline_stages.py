"""Profile energy_and_grad stage timings for standalone modes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
from pathlib import Path
import statistics as st
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising.backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig
from ring_ising.params import make_initial_params_array


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
            f"Invalid case {text!r}. Expected format like 12x16."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[parse_case("12x8"), parse_case("12x32"), parse_case("12x128"),
                 parse_case("14x8"), parse_case("14x32"), parse_case("14x128"),
                 parse_case("16x4"), parse_case("16x8"), parse_case("16x16"),
                 parse_case("18x4"), parse_case("18x8"), parse_case("18x16"),
                 parse_case("20x2"), parse_case("20x4"), parse_case("20x8"),],
        help="Problem sizes in QxL format.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--mode2-widths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 8],
        help="mode2 structured rotation chunk widths to benchmark.",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def make_backend(
    case: BenchmarkCase,
    *,
    mode: str,
    field: float,
    mode2_width: int = 1,
) -> RingIsingAdjointBackend:
    return RingIsingAdjointBackend(
        StandaloneBackendConfig(
            num_qubits=case.num_qubits,
            layers=case.layers,
            field=field,
            gradient_strategy=mode,
            mode2_rotation_chunk_width=mode2_width,
        )
    )


def stage_profile_summary(
    backend: RingIsingAdjointBackend,
    params: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, dict[str, float]]:
    for _ in range(warmup):
        backend.profile_energy_and_grad(params)

    stage_series: dict[str, list[float]] = {}
    total_ms: list[float] = []
    for _ in range(repeats):
        gc.collect()
        profile = backend.profile_energy_and_grad(params)
        timings = profile["timings_ms"]
        total_ms.append(sum(float(value) for value in timings.values()))
        for name, value in timings.items():
            stage_series.setdefault(name, []).append(float(value))

    stage_medians = {
        name: st.median(values)
        for name, values in sorted(stage_series.items())
    }
    return st.median(total_ms), stage_medians


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    print("Standalone baseline stage profiling")
    print(f"  Cases: {', '.join(f'{case.num_qubits}x{case.layers}' for case in args.cases)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Repeats: {args.repeats}")
    print()

    for case in args.cases:
        params = make_initial_params_array(
            num_qubits=case.num_qubits,
            layers=case.layers,
            seed=args.seed,
            init_scale=args.init_scale,
        )
        print(f"{case.num_qubits} qubits x {case.layers} layers")
        mode_specs = [("inverse_walk", "inverse_walk", 1)]
        mode_specs.extend(
            (f"mode2_w{width}", "mode2", width)
            for width in args.mode2_widths
        )
        for label, mode, mode2_width in mode_specs:
            backend = make_backend(
                case, mode=mode, field=args.field, mode2_width=mode2_width
            )
            total_median_ms, stages = stage_profile_summary(
                backend,
                params,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print(f"  {label:<17} total_stage_ms={total_median_ms:.3f}")
            for name, value_ms in stages.items():
                print(f"    {name:<24} {value_ms:9.3f}")
                rows.append(
                    {
                        "num_qubits": case.num_qubits,
                        "layers": case.layers,
                        "mode": label,
                        "gradient_strategy": mode,
                        "mode2_rotation_chunk_width": mode2_width,
                        "stage": name,
                        "median_ms": value_ms,
                        "total_stage_ms": total_median_ms,
                    }
                )
        print()

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "num_qubits",
                    "layers",
                    "mode",
                    "gradient_strategy",
                    "mode2_rotation_chunk_width",
                    "stage",
                    "median_ms",
                    "total_stage_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {args.csv_out}")


if __name__ == "__main__":
    main()
