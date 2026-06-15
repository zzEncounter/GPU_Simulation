"""Profile standalone baseline modes with Nsight Compute and summarize kernel time share."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising.backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig
from ring_ising.params import make_initial_params_array

MODES = ("inverse_walk")
NCU_METRICS = ("gpu__time_duration.sum",)


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


_CUDART: ctypes.CDLL | None = None


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 12x16."
        ) from exc


def parse_mode(text: str) -> str:
    mode = text.strip()
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise argparse.ArgumentTypeError(
            f"Invalid mode {text!r}. Expected one of: {choices}"
        )
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[
            parse_case("12x8"),
            parse_case("12x32"),
            parse_case("12x128"),
            parse_case("16x8"),
            parse_case("16x32"),
            parse_case("20x2"),
            parse_case("20x8"),
        ],
        help="Problem sizes in QxL format.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        type=parse_mode,
        default=list(MODES),
        help="Baseline modes to profile.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unprofiled warmup energy_and_grad calls before the captured run.",
    )
    parser.add_argument(
        "--profile-count",
        type=int,
        default=3,
        help="Number of Nsight Compute runs per case/mode.",
    )
    parser.add_argument(
        "--ncu-path",
        default="ncu",
        help="Path to the Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--ncu-launch-count",
        type=int,
        default=1024,
        help="Maximum profiled CUDA kernel launches per Nsight Compute run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top kernels to print per case/mode.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Summary CSV output path. Also writes a sibling *_profiles.csv file.",
    )
    parser.add_argument(
        "--internal-kernel-profile-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-num-qubits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-mode", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_ncu_available(ncu_path: str) -> str:
    if Path(ncu_path).name != ncu_path:
        candidate = Path(ncu_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(
            f"Nsight Compute CLI executable was not found at {candidate}. "
            "Install NVIDIA Nsight Compute or pass a valid path with --ncu-path."
        )

    candidates: list[Path] = []

    which_result = shutil.which(ncu_path)
    if which_result is not None:
        candidates.append(Path(which_result))

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_raw = os.environ.get(env_name)
        if env_raw:
            candidates.append(Path(env_raw).expanduser() / "bin" / ncu_path)

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        candidates.append(Path(nvcc).resolve().parent.parent / "bin" / ncu_path)

    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        candidates.append(Path("/home") / sudo_user / "cuda" / "bin" / ncu_path)

    candidates.extend(
        [
            Path.home() / "cuda" / "bin" / ncu_path,
            Path("/usr/local/cuda/bin") / ncu_path,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return str(resolved)

    searched = ", ".join(str(path) for path in seen)
    raise RuntimeError(
        "Nsight Compute CLI executable was not found. Install NVIDIA Nsight Compute, "
        "pass its path with --ncu-path, or make sure one of these paths exists: "
        f"{searched}"
    )


def load_cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    def candidate_paths_under(root: Path) -> list[Path]:
        directories = [
            root / "lib64",
            root / "targets" / "x86_64-linux" / "lib",
        ]
        paths: list[Path] = []
        for directory in directories:
            try:
                if not directory.is_dir():
                    continue
                paths.append(directory / "libcudart.so")
                paths.extend(sorted(directory.glob("libcudart.so.*")))
            except PermissionError:
                continue
        return paths

    candidates: list[Path | str] = []
    candidates.extend(candidate_paths_under(Path.home() / "cuda"))

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_raw = os.environ.get(env_name)
        if env_raw:
            candidates.extend(candidate_paths_under(Path(env_raw).expanduser()))

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        candidates.extend(candidate_paths_under(Path(nvcc).resolve().parent.parent))

    ncu = shutil.which("ncu")
    if ncu is not None:
        candidates.extend(candidate_paths_under(Path(ncu).resolve().parent.parent))

    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        candidates.extend(candidate_paths_under(Path("/home") / sudo_user / "cuda"))

    candidates.extend(candidate_paths_under(Path("/usr/local/cuda")))

    found_by_linker = ctypes.util.find_library("cudart")
    if found_by_linker:
        candidates.append(found_by_linker)

    candidates.extend(
        [
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "libcudart.so",
        ]
    )

    last_error: OSError | None = None
    seen: set[str] = set()
    searched: list[str] = []
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        searched.append(candidate_text)
        try:
            _CUDART = ctypes.CDLL(candidate_text)
            return _CUDART
        except OSError as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to load libcudart.so for cudaProfilerStart/cudaProfilerStop. "
        "Searched: " + ", ".join(searched)
    ) from last_error


def cuda_profiler_start() -> None:
    cudart = load_cudart()
    status = int(cudart.cudaProfilerStart())
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {status}.")


def cuda_profiler_stop() -> None:
    cudart = load_cudart()
    status = int(cudart.cudaProfilerStop())
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {status}.")


def make_backend(case: BenchmarkCase, *, mode: str, field: float) -> RingIsingAdjointBackend:
    return RingIsingAdjointBackend(
        StandaloneBackendConfig(
            num_qubits=case.num_qubits,
            layers=case.layers,
            field=field,
            gradient_strategy=mode,
        )
    )


def run_internal_profile_case(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in (
            ("--internal-num-qubits", args.internal_num_qubits),
            ("--internal-layers", args.internal_layers),
            ("--internal-mode", args.internal_mode),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing internal kernel profile arguments: {', '.join(missing)}")

    case = BenchmarkCase(args.internal_num_qubits, args.internal_layers)
    backend = make_backend(case, mode=args.internal_mode, field=args.field)
    params = make_initial_params_array(
        num_qubits=case.num_qubits,
        layers=case.layers,
        seed=args.seed,
        init_scale=args.init_scale,
    )

    for _ in range(max(0, args.warmup)):
        backend.energy_and_grad(params)

    cuda_profiler_start()
    try:
        energy, _ = backend.energy_and_grad(params)
    finally:
        cuda_profiler_stop()

    print(f"ncu_profile_final_energy={energy}", file=sys.stderr)


def parse_ncu_raw_csv(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            continue
        if ("Metric Name" in parsed and "Metric Value" in parsed) or (
            "Kernel Name" in parsed and any(metric in parsed for metric in NCU_METRICS)
        ):
            header_index = index
            break
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def normalize_ncu_metric_value(value: str) -> Any:
    normalized = value.strip().replace(",", "")
    if normalized in {"", "N/A", "nan", "NaN", "-"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return value.strip()


def build_launch_rows(
    *,
    run_id: str,
    mode: str,
    case: BenchmarkCase,
    profile_index: int,
    metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_launch: dict[tuple[str, str], dict[str, Any]] = {}
    for metric_row in metric_rows:
        launch_id = metric_row.get("ID", "").strip()
        kernel_name = (
            metric_row.get("Kernel Name", "").strip()
            or metric_row.get("launch__kernel_name", "").strip()
        )
        if not launch_id or not kernel_name:
            continue
        key = (launch_id, kernel_name)
        row = by_launch.setdefault(
            key,
            {
                "run_id": run_id,
                "mode": mode,
                "num_qubits": case.num_qubits,
                "layers": case.layers,
                "profile_index": profile_index + 1,
                "launch_index": launch_id,
                "kernel_name": kernel_name,
            },
        )

        metric_name = metric_row.get("Metric Name", "").strip()
        if metric_name:
            if metric_name in NCU_METRICS:
                row[metric_name] = normalize_ncu_metric_value(metric_row.get("Metric Value", ""))
            continue

        for column_metric in NCU_METRICS:
            if column_metric in metric_row:
                row[column_metric] = normalize_ncu_metric_value(metric_row.get(column_metric, ""))
    return list(by_launch.values())


def build_profile_share_rows(launch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = {}
    for row in launch_rows:
        key = (
            str(row["run_id"]),
            str(row["mode"]),
            int(row["num_qubits"]),
            int(row["layers"]),
            int(row["profile_index"]),
        )
        grouped.setdefault(key, []).append(row)

    profile_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        run_id, mode, num_qubits, layers, profile_index = key
        kernel_totals: dict[str, dict[str, float]] = {}
        total_gpu_time = 0.0
        for row in rows:
            kernel_name = str(row["kernel_name"])
            gpu_time = float(row.get("gpu__time_duration.sum") or 0.0)
            total_gpu_time += gpu_time
            kernel_entry = kernel_totals.setdefault(
                kernel_name,
                {
                    "gpu_time_sum": 0.0,
                    "launch_count": 0.0,
                },
            )
            kernel_entry["gpu_time_sum"] += gpu_time
            kernel_entry["launch_count"] += 1.0

        ordered = sorted(
            kernel_totals.items(),
            key=lambda item: item[1]["gpu_time_sum"],
            reverse=True,
        )
        for rank, (kernel_name, stats) in enumerate(ordered, start=1):
            gpu_time_sum = stats["gpu_time_sum"]
            launch_count = int(stats["launch_count"])
            share_pct = 100.0 * gpu_time_sum / total_gpu_time if total_gpu_time > 0.0 else 0.0
            profile_rows.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "num_qubits": num_qubits,
                    "layers": layers,
                    "profile_index": profile_index,
                    "rank_within_profile": rank,
                    "kernel_name": kernel_name,
                    "kernel_launch_count": launch_count,
                    "kernel_gpu_time_sum": gpu_time_sum,
                    "kernel_gpu_time_pct": share_pct,
                    "profile_total_gpu_time": total_gpu_time,
                    "kernel_gpu_time_avg_per_launch": (
                        gpu_time_sum / launch_count if launch_count > 0 else 0.0
                    ),
                }
            )
    return profile_rows


def build_summary_rows(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = {}
    for row in profile_rows:
        key = (
            str(row["run_id"]),
            str(row["mode"]),
            int(row["num_qubits"]),
            int(row["layers"]),
            str(row["kernel_name"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    by_run: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        run_id, mode, num_qubits, layers, kernel_name = key
        share_values = [float(row["kernel_gpu_time_pct"]) for row in rows]
        time_values = [float(row["kernel_gpu_time_sum"]) for row in rows]
        launch_counts = [int(row["kernel_launch_count"]) for row in rows]
        total_values = [float(row["profile_total_gpu_time"]) for row in rows]
        summary_row = {
            "run_id": run_id,
            "mode": mode,
            "num_qubits": num_qubits,
            "layers": layers,
            "kernel_name": kernel_name,
            "profiles_observed": len(rows),
            "kernel_launch_count_median": int(round(st.median(launch_counts))),
            "kernel_gpu_time_sum_median": st.median(time_values),
            "kernel_gpu_time_sum_mean": st.fmean(time_values),
            "kernel_gpu_time_pct_median": st.median(share_values),
            "kernel_gpu_time_pct_mean": st.fmean(share_values),
            "profile_total_gpu_time_median": st.median(total_values),
        }
        summary_rows.append(summary_row)
        by_run.setdefault((run_id, mode, num_qubits, layers), []).append(summary_row)

    ranked_rows: list[dict[str, Any]] = []
    for key, rows in by_run.items():
        ordered = sorted(
            rows,
            key=lambda row: float(row["kernel_gpu_time_pct_median"]),
            reverse=True,
        )
        for rank, row in enumerate(ordered, start=1):
            ranked = dict(row)
            ranked["rank_by_median_share"] = rank
            ranked_rows.append(ranked)
    return ranked_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def shorten_kernel_name(name: str, limit: int = 72) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def run_ncu_profile_case(
    *,
    args: argparse.Namespace,
    case: BenchmarkCase,
    mode: str,
    profile_index: int,
) -> list[dict[str, Any]]:
    ncu_path = ensure_ncu_available(args.ncu_path)
    run_id = f"q{case.num_qubits}_l{case.layers}_{mode}"
    command = [
        ncu_path,
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--profile-from-start",
        "off",
        "--launch-count",
        str(args.ncu_launch_count),
        "--metrics",
        ",".join(NCU_METRICS),
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-kernel-profile-case",
        "--internal-num-qubits",
        str(case.num_qubits),
        "--internal-layers",
        str(case.layers),
        "--internal-mode",
        mode,
        "--field",
        str(args.field),
        "--seed",
        str(args.seed + profile_index),
        "--init-scale",
        str(args.init_scale),
        "--warmup",
        str(args.warmup),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if "ERR_NVGPUCTRPERM" in completed.stdout or "ERR_NVGPUCTRPERM" in completed.stderr:
            raise RuntimeError(
                "Nsight Compute cannot access NVIDIA GPU Performance Counters. "
                "Enable profiling permissions or run with sufficient privileges. "
                "See https://developer.nvidia.com/ERR_NVGPUCTRPERM\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        raise RuntimeError(
            "Nsight Compute profiling failed for "
            f"{run_id} (profile {profile_index + 1}).\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    rows = parse_ncu_raw_csv(completed.stdout)
    if not rows:
        rows = parse_ncu_raw_csv(completed.stderr)
    if not rows:
        raise RuntimeError(
            f"Nsight Compute produced no parseable CSV metric rows for {run_id}. "
            "Check profiling permissions, metric availability, and whether no kernels "
            "were captured.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return build_launch_rows(
        run_id=run_id,
        mode=mode,
        case=case,
        profile_index=profile_index,
        metric_rows=rows,
    )


def main() -> None:
    args = parse_args()
    if args.internal_kernel_profile_case:
        run_internal_profile_case(args)
        return

    ensure_ncu_available(args.ncu_path)

    launch_rows: list[dict[str, Any]] = []
    total_profiles = len(args.cases) * len(args.modes) * max(0, args.profile_count)
    current_profile = 0

    print("Standalone baseline kernel profiling")
    print(f"  Cases: {', '.join(f'{case.num_qubits}x{case.layers}' for case in args.cases)}")
    print(f"  Modes: {', '.join(args.modes)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Profile count: {args.profile_count}")
    print(f"  Nsight Compute path: {args.ncu_path}")
    print(f"  Launch count: {args.ncu_launch_count}")
    print()

    for case in args.cases:
        for mode in args.modes:
            for profile_index in range(max(0, args.profile_count)):
                current_profile += 1
                run_id = f"q{case.num_qubits}_l{case.layers}_{mode}"
                print(
                    f"[{current_profile}/{total_profiles}] "
                    f"{run_id} profile={profile_index + 1}/{args.profile_count}"
                )
                launch_rows.extend(
                    run_ncu_profile_case(
                        args=args,
                        case=case,
                        mode=mode,
                        profile_index=profile_index,
                    )
                )

    profile_rows = build_profile_share_rows(launch_rows)
    summary_rows = build_summary_rows(profile_rows)

    by_run: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (
            str(row["run_id"]),
            str(row["mode"]),
            int(row["num_qubits"]),
            int(row["layers"]),
        )
        by_run.setdefault(key, []).append(row)

    print()
    for key in sorted(by_run):
        run_id, mode, num_qubits, layers = key
        rows = sorted(
            by_run[key],
            key=lambda row: float(row["kernel_gpu_time_pct_median"]),
            reverse=True,
        )
        print(f"{num_qubits} qubits x {layers} layers, {mode}")
        for row in rows[: max(0, args.top_k)]:
            print(
                f"  {float(row['kernel_gpu_time_pct_median']):7.2f}%  "
                f"launches~{int(row['kernel_launch_count_median']):4d}  "
                f"{shorten_kernel_name(str(row['kernel_name']))}"
            )
        print()

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        profile_path = args.csv_out.with_name(args.csv_out.stem + "_profiles.csv")
        write_csv(profile_path, profile_rows)
        write_csv(args.csv_out, summary_rows)
        print(f"Kernel share summary written to {args.csv_out}")
        print(f"Per-profile kernel shares written to {profile_path}")


if __name__ == "__main__":
    main()
