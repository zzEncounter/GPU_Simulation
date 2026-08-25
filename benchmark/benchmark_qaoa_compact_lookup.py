"""End-to-end QAOA A/B for chunk-code versus domain-wall lookup."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAD_ROOT = ROOT / "sad"
sys.path.insert(0, str(SAD_ROOT / "python"))
from sad_baseline import energy_and_grad  # noqa: E402


OUTPUT = ROOT / "benchmark" / "results" / "qaoa_compact_lookup.csv"
SHAPES = {
    20: ("f128r2_b128r2", ()),
    24: (
        "f64r3_b64r4",
        (
            "-DSAD_FORWARD_BLOCK_THREADS=64",
            "-DSAD_FORWARD_REGISTER_BITS=3",
            "-DSAD_BLOCK_THREADS=64",
            "-DSAD_REGISTER_BITS=4",
        ),
    ),
    26: (
        "f64r4_b128r3",
        (
            "-DSAD_FORWARD_BLOCK_THREADS=64",
            "-DSAD_FORWARD_REGISTER_BITS=4",
            "-DSAD_BLOCK_THREADS=128",
            "-DSAD_REGISTER_BITS=3",
        ),
    ),
    28: (
        "f64r4_b128r3",
        (
            "-DSAD_FORWARD_BLOCK_THREADS=64",
            "-DSAD_FORWARD_REGISTER_BITS=4",
            "-DSAD_BLOCK_THREADS=128",
            "-DSAD_REGISTER_BITS=3",
        ),
    ),
}


def build(qubits: int, compact: bool, fused_backward: bool) -> Path:
    shape, shape_flags = SHAPES[qubits]
    mode = (
        "compact-fused"
        if compact and fused_backward
        else "compact-split" if compact else "chunk"
    )
    relative = Path("build") / f"libsad_qaoa_{shape}_{mode}.so"
    flags = shape_flags + (
        f"-DSAD_QAOA_COMPACT_LOOKUP={int(compact)}",
        f"-DSAD_QAOA_FUSED_BACKWARD={int(fused_backward)}",
    )
    subprocess.run(
        [
            "make",
            "-C",
            str(SAD_ROOT),
            f"TARGET={relative}",
            f"EXTRA_NVCCFLAGS={' '.join(flags)}",
        ],
        cwd=ROOT,
        check=True,
    )
    return SAD_ROOT / relative


def main() -> None:
    libraries = {
        (qubits, compact, fused_backward): build(
            qubits, compact, fused_backward
        )
        for qubits in SHAPES
        for compact, fused_backward in (
            (False, False),
            (True, False),
            (True, True),
        )
    }
    rows: list[dict[str, object]] = []
    os.environ["SAD_EXECUTION_MODE"] = "optimized"
    for compact, fused_backward in (
        (False, False),
        (True, False),
        (True, True),
    ):
        for qubits in SHAPES:
            os.environ["SAD_LIBRARY_PATH"] = str(
                libraries[qubits, compact, fused_backward]
            )
            steps = 7 if qubits <= 20 else 5 if qubits <= 24 else 3
            result = energy_and_grad(
                circuit="qaoa",
                scalability=(qubits, 8),
                steps=steps,
                warmup_steps=2,
            )
            row = {
                "variant": (
                    "compact-fused"
                    if compact and fused_backward
                    else "compact-split" if compact else "chunk"
                ),
                "qubits": qubits,
                "layers": 8,
                "steps": steps,
                "forward_median_ms": 1e3
                * statistics.median(result.forward_times_s),
                "hamiltonian_median_ms": 1e3
                * statistics.median(result.hamiltonian_times_s),
                "backward_median_ms": 1e3
                * statistics.median(result.backward_times_s),
                "total_median_ms": 1e3
                * statistics.median(result.step_times_s),
                "lookup_kib": result.memory.total_workspace_mib,
                "energy": result.energy,
                "grad_json": json.dumps(result.grad.tolist()),
            }
            rows.append(row)
            print(
                f"{row['variant']:7s} q={qubits} "
                f"F={row['forward_median_ms']:.3f} "
                f"B={row['backward_median_ms']:.3f} "
                f"T={row['total_median_ms']:.3f} ms",
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
