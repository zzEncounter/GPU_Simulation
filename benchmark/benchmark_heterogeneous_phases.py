"""Measure heterogeneous RX/RY phase finalists in one multi-shape binary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "sad" / "src"
SHAPE_SOURCE = ROOT / "benchmark" / "heterogeneous_rotation_shape.cu"
MAIN_SOURCE = ROOT / "benchmark" / "microbench_heterogeneous_rotation.cu"
BUILD = ROOT / "sad" / "build" / "heterogeneous_rotation"
DEFAULT_RAW = ROOT / "benchmark" / "results" / "execution_search_final.csv"
DEFAULT_CANDIDATES = (
    ROOT / "benchmark" / "results" / "heterogeneous_phase_candidates.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmark" / "results" / "heterogeneous_phase_paired_raw.csv"
)
NVCC = os.environ.get("SAD_NVCC", "/usr/local/cuda/bin/nvcc")


@dataclass(frozen=True)
class Shape:
    threads: int
    register_bits: int
    mailbox_chunks: int

    @property
    def tile_bits(self) -> int:
        return 5 + self.register_bits + (self.threads // 32).bit_length() - 1

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


SHAPES = {
    "t32r2m1": Shape(32, 2, 1),
    "t32r4m1": Shape(32, 4, 1),
    "t64r2m1": Shape(64, 2, 1),
    "t64r2m2": Shape(64, 2, 2),
    "t64r2m4": Shape(64, 2, 4),
    "t64r3m1": Shape(64, 3, 1),
    "t64r3m2": Shape(64, 3, 2),
    "t64r3m4": Shape(64, 3, 4),
    "t64r3m8": Shape(64, 3, 8),
    "t64r4m1": Shape(64, 4, 1),
    "t64r4m2": Shape(64, 4, 2),
    "t64r4m4": Shape(64, 4, 4),
    "t64r4m8": Shape(64, 4, 8),
    "t64r4m16": Shape(64, 4, 16),
    "t128r2m1": Shape(128, 2, 1),
    "t128r2m2": Shape(128, 2, 2),
    "t128r2m4": Shape(128, 2, 4),
    "t128r3m1": Shape(128, 3, 1),
    "t128r3m2": Shape(128, 3, 2),
    "t128r3m4": Shape(128, 3, 4),
    "t128r3m8": Shape(128, 3, 8),
}

FIELDS = (
    "repetition",
    "order",
    "gate",
    "direction",
    "qubits",
    "candidate",
    "schedule",
    "phase_count",
    "average_ms",
    "uniform_average_ms",
    "uniform_over_candidate",
    "model_predicted_ms",
    "model_predicted_speedup",
    "phi_checksum_abs_error",
    "lambda_checksum_abs_error",
    "gradient_checksum_abs_error",
    "correct",
)


def _latest_source_mtime() -> float:
    return max(
        path.stat().st_mtime
        for path in (SHAPE_SOURCE, MAIN_SOURCE, *INCLUDE.glob("**/*.cuh"))
    )


def _build() -> Path:
    digest = hashlib.sha256(
        "\0".join(
            flag
            for name, shape in sorted(SHAPES.items())
            for flag in (name, *shape.flags)
        ).encode()
    ).hexdigest()[:12]
    binary = BUILD / f"heterogeneous-{digest}"
    if binary.exists() and binary.stat().st_mtime >= _latest_source_mtime():
        return binary
    BUILD.mkdir(parents=True, exist_ok=True)
    objects: list[Path] = []
    common = ("-O3", "-std=c++17", "-arch=native", "-lineinfo", f"-I{INCLUDE}")
    for name, shape in sorted(SHAPES.items()):
        obj = BUILD / f"{name}-{digest}.o"
        subprocess.run(
            [
                NVCC,
                *common,
                "-dc",
                f"-DSAD_SHAPE_NAMESPACE=sad_shape_{name}_ns",
                f"-DSAD_SHAPE_WRAPPER=sad_shape_{name}_launch",
                *shape.flags,
                str(SHAPE_SOURCE),
                "-o",
                str(obj),
            ],
            cwd=ROOT,
            check=True,
        )
        objects.append(obj)
    subprocess.run(
        [NVCC, *common, str(MAIN_SOURCE), *(str(obj) for obj in objects), "-o", str(binary)],
        cwd=ROOT,
        check=True,
    )
    return binary


def _shape_medians(raw: Path) -> dict[tuple[str, str, int], tuple[str, str, float]]:
    samples: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    with raw.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] != "shape" or row["variant"] not in SHAPES:
                continue
            key = (
                row["gate"],
                row["direction"],
                int(row["qubits"]),
                row["variant"],
                row["family"],
            )
            samples[key].append(float(row["average_ms"]))
    best: dict[tuple[str, str, int], tuple[str, str, float]] = {}
    for key, values in samples.items():
        scenario = key[:3]
        candidate = (key[3], key[4], statistics.median(values))
        if scenario not in best or candidate[2] < best[scenario][2]:
            best[scenario] = candidate
    return best


def _uniform_schedule(qubits: int, variant: str, family: str) -> str:
    shape = SHAPES[variant]
    result: list[str] = []
    first = 0
    while first < qubits:
        phase_family = "compact" if first == 0 else family
        capacity = shape.tile_bits
        if phase_family == "fixed":
            capacity -= 5
        elif phase_family == "pairs":
            capacity -= 6
        count = min(capacity, qubits - first)
        result.append(f"{variant}/{phase_family}/{count}")
        first += count
    return ";".join(result)


def _candidate_schedule(row: dict[str, str]) -> str:
    counts = [int(value) for value in row["phase_division"].split("+")]
    parameters = row["phase_parameters"].split(";")
    if len(counts) != len(parameters):
        raise ValueError("phase candidate has inconsistent fields")
    result = []
    for count, parameter in zip(counts, parameters, strict=True):
        match = re.search(r"=([^/]+)/([^/]+)/", parameter)
        if match is None:
            raise ValueError(f"cannot parse {parameter!r}")
        result.append(f"{match.group(1)}/{match.group(2)}/{count}")
    return ";".join(result)


def _finalists(
    candidates: Path, raw: Path, ranks: int
) -> dict[tuple[str, str, int], list[tuple[str, str, float, float]]]:
    uniform = _shape_medians(raw)
    result: dict[tuple[str, str, int], list[tuple[str, str, float, float]]] = defaultdict(list)
    with candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["rank"]) > ranks:
                continue
            scenario = (row["gate"], row["direction"], int(row["qubits"]))
            variant, family, uniform_model_ms = uniform[scenario]
            result[scenario].append(
                (
                    f"heterogeneous-{row['rank']}",
                    _candidate_schedule(row),
                    float(row["predicted_ms"]),
                    uniform_model_ms,
                )
            )
    for scenario, values in result.items():
        variant, family, uniform_model_ms = uniform[scenario]
        values.append(
            (
                "uniform",
                _uniform_schedule(scenario[2], variant, family),
                uniform_model_ms,
                uniform_model_ms,
            )
        )
    return result


def _measure(
    binary: Path,
    scenario: tuple[str, str, int],
    schedule: str,
    iterations: int,
) -> tuple[float, float, float, float, int]:
    gate, direction, qubits = scenario
    completed = subprocess.run(
        [str(binary), str(qubits), gate, direction, str(iterations), schedule],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()[-1].split(",", 8)
    if len(values) != 9:
        raise RuntimeError(f"unexpected heterogeneous output: {completed.stdout}")
    return float(values[4]), float(values[5]), float(values[6]), float(values[7]), int(values[3])


def _measure_pair(
    binary: Path,
    scenario: tuple[str, str, int],
    first_schedule: str,
    second_schedule: str,
    iterations: int,
) -> tuple[
    tuple[float, float, float, float, int],
    tuple[float, float, float, float, int],
]:
    """Alternately time two schedules inside one CUDA process."""

    gate, direction, qubits = scenario
    completed = subprocess.run(
        [
            str(binary),
            str(qubits),
            gate,
            direction,
            str(iterations),
            first_schedule,
            second_schedule,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"unexpected paired heterogeneous output: {completed.stdout}")
    parsed = []
    for line in lines[-2:]:
        values = line.split(",", 8)
        if len(values) != 9:
            raise RuntimeError(
                f"unexpected paired heterogeneous output: {completed.stdout}"
            )
        parsed.append(
            (
                float(values[4]),
                float(values[5]),
                float(values[6]),
                float(values[7]),
                int(values[3]),
            )
        )
    return parsed[0], parsed[1]


def _completed(path: Path) -> set[tuple[int, str, str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (
                int(row["repetition"]),
                row["gate"],
                row["direction"],
                int(row["qubits"]),
                row["candidate"],
            )
            for row in csv.DictReader(stream)
            if row.get("correct") == "1"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    if min(args.ranks, args.rounds, args.iterations) < 1:
        parser.error("ranks, rounds, and iterations must be positive")
    binary = _build()
    finalists = _finalists(args.candidates, args.raw, args.ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.output)
    exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for repetition in range(args.rounds):
            scenarios = list(finalists)
            random.Random(20260817 + repetition).shuffle(scenarios)
            for scenario in scenarios:
                values = finalists[scenario]
                pending = [
                    value
                    for value in values
                    if (repetition, *scenario, value[0]) not in done
                ]
                if not pending:
                    continue
                order = list(pending)
                random.Random(f"heterogeneous-{repetition}-{scenario}").shuffle(order)
                measured = {
                    name: _measure(binary, scenario, schedule, args.iterations)
                    for name, schedule, _, _ in order
                }
                if "uniform" not in measured:
                    uniform_value = next(value for value in values if value[0] == "uniform")
                    measured["uniform"] = _measure(
                        binary, scenario, uniform_value[1], args.iterations
                    )
                uniform_measurement = measured["uniform"]
                for name, schedule, predicted_ms, uniform_model_ms in order:
                    measurement = measured[name]
                    errors = tuple(
                        abs(measurement[index] - uniform_measurement[index])
                        for index in (1, 2, 3)
                    )
                    scale = max(
                        1.0,
                        *(abs(uniform_measurement[index]) for index in (1, 2, 3)),
                    )
                    correct = max(errors) <= 1e-8 * scale
                    writer.writerow(
                        {
                            "repetition": repetition,
                            "order": "-".join(item[0] for item in order),
                            "gate": scenario[0],
                            "direction": scenario[1],
                            "qubits": scenario[2],
                            "candidate": name,
                            "schedule": schedule,
                            "phase_count": measurement[4],
                            "average_ms": measurement[0],
                            "uniform_average_ms": uniform_measurement[0],
                            "uniform_over_candidate": uniform_measurement[0] / measurement[0],
                            "model_predicted_ms": predicted_ms,
                            "model_predicted_speedup": uniform_model_ms / predicted_ms,
                            "phi_checksum_abs_error": errors[0],
                            "lambda_checksum_abs_error": errors[1],
                            "gradient_checksum_abs_error": errors[2],
                            "correct": int(correct),
                        }
                    )
                    stream.flush()
                    done.add((repetition, *scenario, name))
                    print(
                        f"{scenario[0]}/{scenario[1][0]} q={scenario[2]} {name}: "
                        f"{uniform_measurement[0] / measurement[0]:.3f}x "
                        f"correct={correct}",
                        flush=True,
                    )
                    if not correct:
                        raise RuntimeError(
                            f"checksum mismatch for {scenario} {name}: {errors}"
                        )
    print(f"raw CSV written to {args.output}")


if __name__ == "__main__":
    main()
