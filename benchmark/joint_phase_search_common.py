"""Common models and deterministic schedule helpers for joint phase search."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

ALIASES = {
    "ra": "ra-hea", "su2": "su2-hea", "rzz": "rzz-hea",
    "xxz": "xxz-hva", "eqnn": "equivariant-qnn",
    "data-reupload": "data-reuploading", "drqnn": "data-reuploading",
}
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva",
            "equivariant-qnn", "data-reuploading", "qaoa-ns")
RZZ_CIRCUITS = {"rzz-hea", "qaoa", "qaoa-ns"}
PHASE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAILBOX_CHUNKS = (1, 2)


@dataclass(frozen=True)
class ShapePair:
    forward_threads: int
    forward_register_bits: int
    backward_threads: int
    backward_register_bits: int

    @property
    def name(self) -> str:
        return (f"f{self.forward_threads}r{self.forward_register_bits}_"
                f"b{self.backward_threads}r{self.backward_register_bits}")


@dataclass(frozen=True)
class PhasePlan:
    family: str
    partition_level: float
    phase_count: int
    unit_count: int
    encoded: str = ""
    counts: tuple[int, ...] = ()

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    circuit: str
    qubits: int
    shape: ShapePair
    partition_level: float
    forward_plan: PhasePlan
    backward_plan: PhasePlan
    mailbox_chunks: int
    rzz_strategy: str
    execution_profile: str
    steps: int
    warmup_steps: int

    @property
    def key(self) -> str:
        payload = asdict(self)
        payload["shape"] = asdict(self.shape)
        payload["forward_plan"] = asdict(self.forward_plan)
        payload["backward_plan"] = asdict(self.backward_plan)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return f"joint:{self.circuit}:{self.qubits}:{digest}"


def normalize_circuit(value: str) -> str:
    name = value.strip().lower().replace("_", "-")
    name = ALIASES.get(name, name)
    if name not in CIRCUITS:
        raise ValueError(f"unsupported or excluded circuit: {value}")
    return name


def repetitions_for_qubits(qubits: int) -> int:
    if qubits <= 8: return 20
    if qubits <= 14: return 10
    if qubits <= 20: return 5
    if qubits <= 24: return 3
    return 2


def tile_bits(threads: int, register_bits: int) -> int:
    return 5 + register_bits + int(math.log2(threads // 32))


def valid_shape(threads: int, register_bits: int) -> bool:
    return (threads >= 32 and threads % 32 == 0 and register_bits >= 0
            and 7 <= tile_bits(threads, register_bits) <= 12)


def balanced_counts(total: int, parts: int) -> tuple[int, ...]:
    if parts < 1 or parts > total:
        raise ValueError(f"cannot split {total} units into {parts} phases")
    base, remainder = divmod(total, parts)
    return tuple([base + 1] * remainder + [base] * (parts - remainder))


def phase_count(level: float, minimum: int, maximum: int) -> int:
    if not 0.0 <= level <= 1.0:
        raise ValueError("partition level must be in [0, 1]")
    value = round(minimum + level * (maximum - minimum))
    return max(minimum, min(maximum, value))


def encode_compact(counts: tuple[int, ...], threads: int, register_bits: int) -> str:
    capacity = tile_bits(threads, register_bits)
    if any(count <= 0 or count > capacity for count in counts):
        raise ValueError("phase count exceeds tile capacity")
    phases = []
    for count in counts:
        lane = min(5, count)
        remaining = count - lane
        reg = min(register_bits, remaining)
        warp = remaining - reg
        if warp > capacity - 5 - register_bits:
            raise ValueError("phase cannot be encoded for selected shape")
        phases.append(f"L{lane}R{reg}W{warp}")
    return "compact:" + "-".join(phases)


def build_rotation_plan(qubits: int, threads: int, register_bits: int,
                        level: float) -> PhasePlan:
    capacity = tile_bits(threads, register_bits)
    minimum = max(1, math.ceil(qubits / capacity))
    count = phase_count(level, minimum, qubits)
    counts = balanced_counts(qubits, count)
    encoded = encode_compact(counts, threads, register_bits)
    return PhasePlan("rotation", level, count, qubits, encoded, counts)


def build_xxz_plan(qubits: int, threads: int, register_bits: int,
                   level: float) -> PhasePlan:
    bonds = qubits // 2
    capacity = max(1, tile_bits(threads, register_bits) // 2)
    minimum = max(1, math.ceil(bonds / capacity))
    count = phase_count(level, minimum, bonds)
    return PhasePlan("xxz-matching", level, count, bonds, "",
                     balanced_counts(bonds, count))
