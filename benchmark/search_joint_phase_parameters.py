"""Exhaustive, resumable joint shape/phase/mailbox search.

Each circuit owns an append-only CSV and an atomically replaced JSON summary.
Use --dry-run to inspect the manifest without compiling CUDA libraries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sad" / "python"))
from joint_phase_search_common import (  # noqa: E402
    CIRCUITS, MAILBOX_CHUNKS, PHASE_LEVELS, RZZ_CIRCUITS, Candidate, PhasePlan,
    ShapePair, build_rotation_plan, build_xxz_plan, normalize_circuit,
    repetitions_for_qubits, valid_shape,
)
from search_circuit_common import candidate_library, flags_for  # noqa: E402

SHAPE_PAIRS = (
    ShapePair(32, 2, 32, 2), ShapePair(64, 2, 64, 2),
    ShapePair(128, 2, 128, 2), ShapePair(32, 2, 128, 2),
    ShapePair(128, 2, 32, 2),
)
FIELDS = (
    "timestamp_utc", "run_id", "candidate_key", "status", "error", "circuit",
    "qubits", "layers", "precision", "random_seed", "forward_threads",
    "forward_register_bits", "backward_threads", "backward_register_bits",
    "mailbox_chunks", "partition_level", "phase_family_forward",
    "phase_family_backward", "phase_count_forward", "phase_count_backward",
    "forward_phase_plan", "backward_phase_plan", "forward_pair_counts",
    "backward_pair_counts", "rzz_strategy", "execution_profile", "steps", "warmup_steps",
    "repetitions", "median_ms", "mean_ms", "min_ms", "max_ms", "std_ms",
    "forward_ms", "hamiltonian_ms", "backward_ms", "energy_abs_error",
    "gradient_max_abs_error", "kernel_variant", "library_path",
)


def read_result_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read both pre-execution-profile and current CSV schemas safely."""
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return rows
        current_header = list(header)
        if "execution_profile" not in current_header:
            steps_index = current_header.index("steps")
            extended_header = (
                current_header[:steps_index]
                + ["execution_profile"]
                + current_header[steps_index:]
            )
        else:
            extended_header = current_header
        for values in reader:
            if not values:
                continue
            if len(values) == len(current_header):
                row = dict(zip(current_header, values))
                row.setdefault("execution_profile", "optimized")
            elif len(values) == len(extended_header):
                row = dict(zip(extended_header, values))
            else:
                # Preserve visibility of malformed rows without allowing them
                # to contaminate performance aggregates.
                continue
            rows.append(row)
    return rows


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(x) for x in value.split(",") if x.strip()))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def candidates(circuit: str, qubits: Iterable[int]) -> list[Candidate]:
    circuit = normalize_circuit(circuit)
    result: list[Candidate] = []
    for q in qubits:
        for shape in SHAPE_PAIRS:
            f_valid = valid_shape(shape.forward_threads, shape.forward_register_bits)
            b_valid = valid_shape(shape.backward_threads, shape.backward_register_bits)
            for level in PHASE_LEVELS:
                if circuit == "xxz-hva":
                    fp = build_xxz_plan(q, shape.forward_threads, shape.forward_register_bits, level)
                    bp = build_xxz_plan(q, shape.backward_threads, shape.backward_register_bits, level)
                else:
                    fp = build_rotation_plan(q, shape.forward_threads, shape.forward_register_bits, level)
                    bp = build_rotation_plan(q, shape.backward_threads, shape.backward_register_bits, level)
                strategies = ("merged", "split") if circuit in RZZ_CIRCUITS else ("not_applicable",)
                for mailbox in MAILBOX_CHUNKS:
                    for strategy in strategies:
                        result.append(Candidate(
                            circuit, q, shape, level, fp, bp, mailbox, strategy,
                            "optimized", repetitions_for_qubits(q), 3,
                        ))
        if circuit != "xxz-hva":
            for shape in SHAPE_PAIRS:
                fp = PhasePlan("legacy", 0.0, 0, q, "", ())
                bp = PhasePlan("legacy", 0.0, 0, q, "", ())
                result.append(Candidate(
                    circuit, q, shape, 0.0, fp, bp, 1, "not_applicable",
                    "legacy-generic", repetitions_for_qubits(q), 3,
                ))
    return result


def _base_row(candidate: Candidate, run_id: str) -> dict[str, Any]:
    s = candidate.shape
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "candidate_key": candidate.key, "status": "", "error": "",
        "circuit": candidate.circuit, "qubits": candidate.qubits, "layers": 8,
        "precision": "float64", "random_seed": 42,
        "forward_threads": s.forward_threads, "forward_register_bits": s.forward_register_bits,
        "backward_threads": s.backward_threads, "backward_register_bits": s.backward_register_bits,
        "mailbox_chunks": candidate.mailbox_chunks, "partition_level": candidate.partition_level,
        "phase_family_forward": candidate.forward_plan.family,
        "phase_family_backward": candidate.backward_plan.family,
        "phase_count_forward": candidate.forward_plan.phase_count,
        "phase_count_backward": candidate.backward_plan.phase_count,
        "forward_phase_plan": candidate.forward_plan.encoded,
        "backward_phase_plan": candidate.backward_plan.encoded,
        "forward_pair_counts": "+".join(map(str, candidate.forward_plan.counts)) if candidate.forward_plan.family == "xxz-matching" else "",
        "backward_pair_counts": "+".join(map(str, candidate.backward_plan.counts)) if candidate.backward_plan.family == "xxz-matching" else "",
        "rzz_strategy": candidate.rzz_strategy,
        "execution_profile": candidate.execution_profile, "steps": candidate.steps,
        "warmup_steps": candidate.warmup_steps, "repetitions": candidate.steps,
        "median_ms": "", "mean_ms": "", "min_ms": "", "max_ms": "", "std_ms": "",
        "forward_ms": "", "hamiltonian_ms": "", "backward_ms": "",
        "energy_abs_error": "", "gradient_max_abs_error": "",
        "kernel_variant": "", "library_path": "",
    }


def write_summary(csv_path: Path, json_path: Path, circuit: str, expected: int) -> None:
    rows = read_result_rows(csv_path)
    ok = [r for r in rows if r.get("status") == "ok" and r.get("median_ms")]
    values = [float(r["median_ms"]) for r in ok]
    def selected(fn):
        return fn(ok, key=lambda r: float(r["median_ms"])) if ok else None
    payload = {
        "schema_version": 1, "circuit": circuit,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_candidates": expected, "attempted_candidates": len(rows),
        "completed_ok": len(ok),
        "status_counts": {status: sum(r.get("status") == status for r in rows)
                          for status in sorted({r.get("status", "") for r in rows})},
        "runtime_ms": {
            "best": min(values) if values else None,
            "worst": max(values) if values else None,
            "arithmetic_mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        },
        "best_candidate": selected(min), "worst_candidate": selected(max),
        "by_qubits": {},
    }
    by_qubits: dict[str, list[dict[str, str]]] = {}
    for row in ok:
        by_qubits.setdefault(row["qubits"], []).append(row)
    for qubits, group in sorted(by_qubits.items(), key=lambda item: int(item[0])):
        group_values = [float(row["median_ms"]) for row in group]
        payload["by_qubits"][qubits] = {
            "completed_ok": len(group),
            "runtime_ms": {
                "best": min(group_values),
                "worst": max(group_values),
                "arithmetic_mean": statistics.fmean(group_values),
                "median": statistics.median(group_values),
            },
            "best_candidate": min(group, key=lambda row: float(row["median_ms"])),
            "worst_candidate": max(group, key=lambda row: float(row["median_ms"])),
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, json_path)


def _xxz_binary(candidate: Candidate) -> Path:
    """Build the historical XXZ matching microbenchmark for one shape."""
    nvcc = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")
    cache = ROOT / "sad" / "build" / "joint_xxz"
    flags = (
        f"-DSAD_FORWARD_BLOCK_THREADS={candidate.shape.forward_threads}",
        f"-DSAD_FORWARD_REGISTER_BITS={candidate.shape.forward_register_bits}",
        f"-DSAD_BLOCK_THREADS={candidate.shape.backward_threads}",
        f"-DSAD_REGISTER_BITS={candidate.shape.backward_register_bits}",
        f"-DSAD_MAILBOX_CHUNKS={candidate.mailbox_chunks}",
        "-DSAD_XXZ_COMPONENT_MODE=0", "-DSAD_XXZ_PERSISTENT=0",
    )
    digest = __import__("hashlib").sha256("\0".join(flags).encode()).hexdigest()[:12]
    binary = cache / f"{candidate.shape.name}-m{candidate.mailbox_chunks}-{digest}"
    sources = [ROOT / "benchmark" / "microbench_xxz.cu", *(ROOT / "sad" / "src").glob("**/*.cuh")]
    latest = max(source.stat().st_mtime for source in sources)
    if binary.exists() and binary.stat().st_mtime >= latest:
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        nvcc, "-O3", "-std=c++17", "-arch=native", f"-I{ROOT / 'sad' / 'src'}",
        *flags, str(ROOT / "benchmark" / "microbench_xxz.cu"), "-o", str(binary),
    ], cwd=ROOT, check=True)
    return binary


def _run_xxz_candidate(candidate: Candidate) -> tuple[float, float, float, Path]:
    """Measure even/odd forward/backward pair schedules via microbench_xxz."""
    binary = _xxz_binary(candidate)
    samples: dict[str, list[float]] = {"forward": [], "backward": []}
    for direction, plan in (("forward", candidate.forward_plan), ("backward", candidate.backward_plan)):
        counts = ",".join(map(str, plan.counts))
        for parity in (0, 1):
            completed = subprocess.run(
                [str(binary), str(candidate.qubits), direction, str(parity),
                 str(candidate.steps), counts], cwd=ROOT, check=True,
                capture_output=True, text=True,
            )
            fields = completed.stdout.strip().splitlines()[-1].split(",")
            if len(fields) != 17:
                raise RuntimeError(f"unexpected XXZ microbenchmark output: {completed.stdout}")
            samples[direction].append(float(fields[12]))
    forward = statistics.fmean(samples["forward"])
    backward = statistics.fmean(samples["backward"])
    return forward + backward, forward, backward, binary


def run_circuit(circuit: str, qs: tuple[int, ...], out_dir: Path, seed: int,
                dry_run: bool, retry_failed: bool) -> None:
    jobs = candidates(circuit, qs)
    csv_path, json_path = out_dir / f"{circuit}.csv", out_dir / f"{circuit}.json"
    done: dict[str, str] = {
        row["candidate_key"]: row.get("status", "")
        for row in read_result_rows(csv_path)
        if row.get("candidate_key")
    }
    if dry_run:
        print(f"{circuit}: {len(jobs)} candidates ({sum(q in qs for q in qs)} qubit groups)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    stream = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    if not exists:
        writer.writeheader(); stream.flush(); os.fsync(stream.fileno())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    write_summary(csv_path, json_path, circuit, len(jobs))
    from sad_baseline import energy_and_grad
    ordered = list(jobs); random.Random(seed).shuffle(ordered)
    interrupted = False
    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    old = signal.signal(signal.SIGINT, stop)
    try:
        for c in ordered:
            if interrupted: break
            if c.key in done and (done[c.key] == "ok" or not retry_failed): continue
            row = _base_row(c, run_id)
            try:
                if not valid_shape(c.shape.forward_threads, c.shape.forward_register_bits) or not valid_shape(c.shape.backward_threads, c.shape.backward_register_bits):
                    row["status"] = "invalid_shape"
                elif c.forward_plan.family == "xxz-matching":
                    total, forward, backward, binary = _run_xxz_candidate(c)
                    row.update(status="ok", median_ms=total, mean_ms=total,
                               min_ms=total, max_ms=total, std_ms=0.0,
                               forward_ms=forward, hamiltonian_ms=0.0,
                               backward_ms=backward, library_path=str(binary),
                               kernel_variant="")
                else:
                    flags = flags_for(
                        forward_threads=c.shape.forward_threads,
                        forward_register_bits=c.shape.forward_register_bits,
                        backward_threads=c.shape.backward_threads,
                        backward_register_bits=c.shape.backward_register_bits,
                        mailbox_chunks=c.mailbox_chunks,
                    )
                    if c.execution_profile == "legacy-generic":
                        flags["SAD_REAL_AMPLITUDE"] = 0
                    if c.circuit in RZZ_CIRCUITS:
                        flags.update({"SAD_RZZ_FORWARD_FUSED": 1 if c.rzz_strategy == "merged" else 0,
                                      "SAD_RZZ_BACKWARD_STRATEGY": 1 if c.rzz_strategy == "merged" else 0})
                    library = candidate_library(flags)
                    os.environ["SAD_LIBRARY_PATH"] = str(library)
                    os.environ["SAD_EXECUTION_MODE"] = (
                        "legacy" if c.execution_profile == "legacy-generic" else "optimized"
                    )
                    result = energy_and_grad(
                        circuit=c.circuit, random_seed=42, scalability=(c.qubits, 8),
                        precision="float64", steps=c.steps, warmup_steps=3,
                        forward_phase_plan=c.forward_plan.encoded,
                        backward_phase_plan=c.backward_plan.encoded,
                    )
                    times = [1000.0 * x for x in result.step_times_s]
                    row.update(status="ok", median_ms=statistics.median(times),
                               mean_ms=statistics.fmean(times), min_ms=min(times),
                               max_ms=max(times), std_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
                               forward_ms=1000.0 * statistics.fmean(result.forward_times_s),
                               hamiltonian_ms=1000.0 * statistics.fmean(result.hamiltonian_times_s),
                               backward_ms=1000.0 * statistics.fmean(result.backward_times_s),
                               library_path=str(library), kernel_variant=result.kernel_variant)
            except Exception as exc:
                message = str(exc).lower()
                row["status"] = (
                    "compile_failed" if "nvcc" in message or "compilation" in message
                    else "runtime_failed"
                )
                row["error"] = repr(exc)
            writer.writerow(row); stream.flush(); os.fsync(stream.fileno())
            done[c.key] = row["status"]
            write_summary(csv_path, json_path, circuit, len(jobs))
            print(f"{circuit} {c.qubits}q {c.key} {row['status']}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old); stream.flush(); os.fsync(stream.fileno()); stream.close()
        write_summary(csv_path, json_path, circuit, len(jobs))
    if interrupted:
        print(f"{circuit}: interrupted; previous rows preserved", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", default=",".join(CIRCUITS))
    parser.add_argument("--qubits", type=parse_ints, default=tuple(range(4, 29, 2)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark/results/joint_phase")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    circuits = tuple(normalize_circuit(c) for c in args.circuits.split(","))
    for circuit in circuits:
        run_circuit(circuit, args.qubits, args.output_dir, args.shuffle_seed,
                    args.dry_run, args.retry_failed)


if __name__ == "__main__":
    main()
