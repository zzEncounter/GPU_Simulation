"""Shared, resumable end-to-end search support for the added circuits.

The search scripts deliberately measure complete energy-and-gradient calls.  A
row is committed to CSV and the aggregate JSON after every candidate, so a
stopped process never loses completed measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))


SHAPE_GRID = (32, 64, 128, 256)
REGISTER_GRID = (2, 3, 4)
# nonshared_ring_rzz_backward_kernel is instantiated in the shared library for
# every circuit and allocates kMaxQubits * threads doubles.  Values >=256 exceed
# the 48 KiB static shared-memory limit even when the measured circuit is MERA.
ORDINARY_GRID = (64, 128)
LOOKUP_GRID = (6, 8, 10)


def tile_bits(threads: int, register_bits: int) -> int:
    return 5 + register_bits + (threads // 32).bit_length() - 1


def valid_shape(threads: int, register_bits: int) -> bool:
    return threads >= 32 and threads % 32 == 0 and 7 <= tile_bits(threads, register_bits) <= 12


def phase_plan(
    qubits: int,
    threads: int,
    register_bits: int,
    family: str = "compact",
) -> str:
    """Create a valid class plan accepted by runtime/lookups.cuh."""
    capacity = tile_bits(threads, register_bits)
    lane = 5
    reg = register_bits
    warp = capacity - lane - reg
    if qubits <= capacity:
        return ""
    phases: list[tuple[int, int, int]] = []
    remaining = qubits
    first = True
    while remaining:
        if family == "compact" or first:
            counts = [min(lane, remaining), 0, 0]
            remaining -= counts[0]
            counts[1] = min(reg, remaining)
            remaining -= counts[1]
            counts[2] = min(warp, remaining)
            remaining -= counts[2]
        else:
            counts = [0, min(reg, remaining), 0]
            remaining -= counts[1]
            counts[2] = min(warp, remaining)
            remaining -= counts[2]
        if sum(counts) == 0:
            raise ValueError("unable to construct phase plan")
        phases.append(tuple(counts))
        first = False
    encoded = "-".join(f"L{l}R{r}W{w}" for l, r, w in phases)
    return f"{family}:{encoded}"


def candidate_library(flags: dict[str, int]) -> Path:
    normalized = tuple(sorted((str(k), int(v)) for k, v in flags.items()))
    digest = hashlib.sha256(repr(normalized).encode()).hexdigest()[:14]
    path = SAD_ROOT / "build" / f"libsad_search_{digest}.so"
    sources = [SAD_ROOT / "src" / "sad_cuda.cu", SAD_ROOT / "include" / "sad_api.h"]
    sources.extend((SAD_ROOT / "src").glob("**/*.cuh"))
    latest_source = max(source.stat().st_mtime for source in sources)
    if path.exists() and path.stat().st_mtime >= latest_source:
        return path
    defines = [f"-D{k}={v}" for k, v in normalized]
    command = [
        "make", "-C", str(SAD_ROOT),
        f"TARGET=build/libsad_search_{digest}.so",
        f"EXTRA_NVCCFLAGS={' '.join(defines)}",
    ]
    subprocess.run(command, check=True)
    return path


def flags_for(
    *,
    forward_threads: int,
    forward_register_bits: int,
    backward_threads: int,
    backward_register_bits: int,
    ordinary_threads: int = 128,
    diagonal_threads: int = 64,
    shared_diagonal_threads: int = 128,
    lookup_bits: int = 8,
    mailbox_chunks: int = 1,
    legacy_reduction: int = 1,
    rotation_warp_atomic: int = 0,
    diagonal_warp_atomic: int = 0,
) -> dict[str, int]:
    return {
        "SAD_FORWARD_BLOCK_THREADS": forward_threads,
        "SAD_FORWARD_REGISTER_BITS": forward_register_bits,
        "SAD_BLOCK_THREADS": backward_threads,
        "SAD_REGISTER_BITS": backward_register_bits,
        "SAD_ORDINARY_BLOCK_THREADS": ordinary_threads,
        "SAD_DIAGONAL_BLOCK_THREADS": diagonal_threads,
        "SAD_SHARED_DIAGONAL_BLOCK_THREADS": shared_diagonal_threads,
        "SAD_DIAGONAL_LOOKUP_BITS": lookup_bits,
        "SAD_MAILBOX_CHUNKS": mailbox_chunks,
        "SAD_LEGACY_BLOCK_REDUCTION": legacy_reduction,
        "SAD_ROTATION_WARP_ATOMIC": rotation_warp_atomic,
        "SAD_DIAGONAL_WARP_ATOMIC": diagonal_warp_atomic,
    }


def make_candidate(
    stage: str,
    qubits: int,
    config: dict[str, Any],
    *,
    forward_phase_plan: str = "",
    backward_phase_plan: str = "",
) -> dict[str, Any]:
    flag_names = {
        "forward_threads", "forward_register_bits", "backward_threads",
        "backward_register_bits", "ordinary_threads", "diagonal_threads",
        "shared_diagonal_threads", "lookup_bits", "mailbox_chunks",
        "legacy_reduction", "rotation_warp_atomic", "diagonal_warp_atomic",
    }
    flags = flags_for(**{name: config[name] for name in flag_names})
    identity = json.dumps(
        {
            "stage": stage, "qubits": qubits, "config": config,
            "forward_phase_plan": forward_phase_plan,
            "backward_phase_plan": backward_phase_plan,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return {
        "candidate_key": f"{stage}:{qubits}:{digest}",
        "stage": stage,
        "qubits": qubits,
        "config": dict(config),
        "flags": flags,
        "forward_phase_plan": forward_phase_plan,
        "backward_phase_plan": backward_phase_plan,
    }


def best_config(
    csv_path: Path,
    qubits: int,
    stages: str | tuple[str, ...],
    metric: str,
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Return the fastest completed config and its two phase plans."""
    if isinstance(stages, str):
        stages = (stages,)
    best: tuple[float, dict[str, Any], str, str] | None = None
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if (row.get("status") != "ok" or int(row["qubits"]) != qubits
                        or row["stage"] not in stages):
                    continue
                value = float(row[metric])
                item = (
                    value,
                    json.loads(row["config"]),
                    row.get("forward_phase_plan", ""),
                    row.get("backward_phase_plan", ""),
                )
                if best is None or item[0] < best[0]:
                    best = item
    return (dict(fallback), "", "") if best is None else best[1:]


def _key(row: dict[str, Any]) -> str:
    return str(row["candidate_key"])


def _write_summary(csv_path: Path, json_path: Path, metadata: dict[str, Any]) -> None:
    values: list[float] = []
    rows = 0
    failed_rows = 0
    fastest: dict[str, Any] | None = None
    slowest: dict[str, Any] | None = None
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") != "ok":
                    failed_rows += 1
                    continue
                value = float(row["median_ms"])
                values.append(value)
                rows += 1
                item = {"median_ms": value, "candidate_key": row["candidate_key"],
                        "stage": row["stage"], "circuit": row["circuit"],
                        "qubits": int(row["qubits"]),
                        "config": json.loads(row["config"])}
                if fastest is None or value < fastest["median_ms"]:
                    fastest = item
                if slowest is None or value > slowest["median_ms"]:
                    slowest = item
                grouped.setdefault(row["qubits"], []).append((value, item))
    by_qubits: dict[str, Any] = {}
    for qubits, items in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        ordered = sorted(items, key=lambda item: item[0])
        by_qubits[qubits] = {
            "completed_rows": len(ordered),
            "fastest": ordered[0][1],
            "slowest": ordered[-1][1],
            "average_median_ms": statistics.fmean(value for value, _ in ordered),
        }
    payload = {
        **metadata,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_rows": rows,
        "failed_rows": failed_rows,
        "fastest": fastest,
        "slowest": slowest,
        "average_median_ms": statistics.fmean(values) if values else None,
        "by_qubits": by_qubits,
    }
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(json_path)


def run_search(
    *,
    circuit: str,
    candidates: Iterable[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
    layers: int,
    steps: int,
    warmup_steps: int,
    precision: str = "float64",
    seed: int = 42,
) -> None:
    # Delay NumPy/CUDA imports so ``--help`` and candidate inspection work on
    # machines that do not have the runtime environment installed.
    from sad_baseline import energy_and_grad  # noqa: E402

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "timestamp_utc", "candidate_key", "stage", "circuit", "qubits",
        "layers", "config", "flags", "forward_phase_plan",
        "backward_phase_plan", "steps", "warmup_steps", "median_ms",
        "mean_ms", "forward_ms", "hamiltonian_ms", "backward_ms",
        "status", "error",
    )
    done: set[str] = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            done = {_key(row) for row in csv.DictReader(stream)}
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    stream = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    if not exists:
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    metadata = {"circuit": circuit, "csv": str(csv_path), "json": str(json_path),
                "layers": layers, "steps": steps, "warmup_steps": warmup_steps}
    _write_summary(csv_path, json_path, metadata)

    def interrupted(_signum: int, _frame: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        _write_summary(csv_path, json_path, metadata)
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGINT, interrupted)
    try:
        for candidate in candidates:
            key = str(candidate["candidate_key"])
            if key in done:
                continue
            config = candidate["config"]
            flags = candidate["flags"]
            candidate_layers = int(candidate.get("layers", layers))
            try:
                library = candidate_library(flags)
                os.environ["SAD_LIBRARY_PATH"] = str(library)
                os.environ["SAD_EXECUTION_MODE"] = "optimized"
                result = energy_and_grad(
                    circuit=circuit,
                    random_seed=seed,
                    scalability=(int(candidate["qubits"]), candidate_layers),
                    precision=precision,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    forward_phase_plan=candidate.get("forward_phase_plan", ""),
                    backward_phase_plan=candidate.get("backward_phase_plan", ""),
                )
                forward_ms = 1000.0 * statistics.fmean(result.forward_times_s)
                hamiltonian_ms = 1000.0 * statistics.fmean(result.hamiltonian_times_s)
                backward_ms = 1000.0 * statistics.fmean(result.backward_times_s)
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate_key": key, "stage": candidate["stage"],
                    "circuit": circuit, "qubits": candidate["qubits"],
                    "layers": candidate_layers, "config": json.dumps(config, sort_keys=True),
                    "flags": json.dumps(flags, sort_keys=True),
                    "forward_phase_plan": candidate.get("forward_phase_plan", ""),
                    "backward_phase_plan": candidate.get("backward_phase_plan", ""),
                    "steps": steps, "warmup_steps": warmup_steps,
                    "median_ms": 1000.0 * result.median_step_time_s,
                    "mean_ms": 1000.0 * result.mean_step_time_s,
                    "forward_ms": forward_ms, "hamiltonian_ms": hamiltonian_ms,
                    "backward_ms": backward_ms, "status": "ok", "error": "",
                }
            except Exception as exc:  # keep other candidates resumable
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate_key": key, "stage": candidate["stage"],
                    "circuit": circuit, "qubits": candidate["qubits"],
                    "layers": candidate_layers, "config": json.dumps(config, sort_keys=True),
                    "flags": json.dumps(flags, sort_keys=True),
                    "forward_phase_plan": candidate.get("forward_phase_plan", ""),
                    "backward_phase_plan": candidate.get("backward_phase_plan", ""),
                    "steps": steps, "warmup_steps": warmup_steps,
                    "median_ms": "", "mean_ms": "", "forward_ms": "",
                    "hamiltonian_ms": "", "backward_ms": "", "status": "error",
                    "error": repr(exc),
                }
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            done.add(key)
            _write_summary(csv_path, json_path, metadata)
            if row["status"] == "ok":
                print(f"{circuit} {candidate['qubits']}q {candidate['stage']} "
                      f"{float(row['median_ms']):.6f} ms", flush=True)
            else:
                print(f"{circuit} {candidate['qubits']}q {candidate['stage']} ERROR "
                      f"{row['error']}", flush=True)
    except KeyboardInterrupt:
        print("search interrupted; completed rows and summary were preserved", file=sys.stderr)
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        _write_summary(csv_path, json_path, metadata)
