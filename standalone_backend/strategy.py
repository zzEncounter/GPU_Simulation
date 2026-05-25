"""Strategy resolution helpers for standalone backend gradients."""

from __future__ import annotations

import shutil
import subprocess

from .config import RingIsingConfig, StrategyResolution


def query_visible_gpu_memory() -> tuple[int | None, int | None, str | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None, "nvidia-smi not found on PATH"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, None, str(exc)

    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
    if first_line is None:
        return None, None, "nvidia-smi returned no GPU rows"
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        return None, None, "Unable to parse nvidia-smi memory output"
    try:
        return int(parts[0]), int(parts[1]), None
    except ValueError:
        return None, None, "Unable to parse GPU memory values"


def estimate_checkpoint_interval_for_budget(
    config: RingIsingConfig, memory_budget_bytes: int
) -> int | None:
    feasible_intervals: list[int] = []
    for interval in range(1, config.num_ops):
        workspace_bytes = (
            config.estimated_gradient_state_buffers_for("checkpoint", interval)
            * config.statevector_nbytes
        )
        if workspace_bytes <= memory_budget_bytes:
            feasible_intervals.append(interval)
    if not feasible_intervals:
        return None
    # Prefer larger intervals to reduce recomputation when memory budget allows.
    return max(feasible_intervals)


def resolve_strategy(config: RingIsingConfig) -> StrategyResolution:
    requested = config.gradient_strategy
    if requested != "auto":
        checkpoint_interval_ops = (
            0
            if requested in {"save_param_states", "bruteforce_parallel_q6"}
            else config.resolve_checkpoint_interval_ops(strategy=requested)
        )
        return StrategyResolution(
            requested_strategy=requested,
            resolved_strategy=requested,
            checkpoint_interval_ops=checkpoint_interval_ops,
            available_gpu_memory_mib=None,
            total_gpu_memory_mib=None,
            memory_budget_mib=None,
            estimated_workspace_gib=config.estimated_gradient_workspace_gib_for(
                requested, checkpoint_interval_ops
            ),
            note="Explicit strategy requested by user.",
        )

    free_mib, total_mib, note = query_visible_gpu_memory()
    if free_mib is None:
        checkpoint_interval_ops = config.resolve_checkpoint_interval_ops(strategy="checkpoint")
        return StrategyResolution(
            requested_strategy="auto",
            resolved_strategy="checkpoint",
            checkpoint_interval_ops=checkpoint_interval_ops,
            available_gpu_memory_mib=None,
            total_gpu_memory_mib=total_mib,
            memory_budget_mib=None,
            estimated_workspace_gib=config.estimated_gradient_workspace_gib_for(
                "checkpoint", checkpoint_interval_ops
            ),
            note=(
                "Fell back to checkpoint because GPU free-memory probing failed"
                + (f": {note}" if note else ".")
            ),
        )

    budget_mib = max(
        0.0,
        free_mib * config.auto_memory_budget_fraction - config.auto_memory_reserve_mib,
    )
    budget_bytes = int(budget_mib * 1024 * 1024)

    save_param_bytes = int(
        config.estimated_gradient_state_buffers_for("save_param_states")
        * config.statevector_nbytes
    )
    if save_param_bytes <= budget_bytes:
        return StrategyResolution(
            requested_strategy="auto",
            resolved_strategy="save_param_states",
            checkpoint_interval_ops=0,
            available_gpu_memory_mib=free_mib,
            total_gpu_memory_mib=total_mib,
            memory_budget_mib=budget_mib,
            estimated_workspace_gib=config.estimated_gradient_workspace_gib_for(
                "save_param_states"
            ),
            note="Auto-selected save_param_states within the GPU memory budget.",
        )

    checkpoint_interval_ops = estimate_checkpoint_interval_for_budget(config, budget_bytes)
    if checkpoint_interval_ops is None:
        checkpoint_interval_ops = config.default_checkpoint_interval_ops()
        note = (
            "No checkpoint interval fit the memory budget estimate; "
            "falling back to the default checkpoint interval."
        )
    else:
        note = "Auto-selected checkpoint to satisfy the GPU memory budget."
    return StrategyResolution(
        requested_strategy="auto",
        resolved_strategy="checkpoint",
        checkpoint_interval_ops=checkpoint_interval_ops,
        available_gpu_memory_mib=free_mib,
        total_gpu_memory_mib=total_mib,
        memory_budget_mib=budget_mib,
        estimated_workspace_gib=config.estimated_gradient_workspace_gib_for(
            "checkpoint", checkpoint_interval_ops
        ),
        note=note,
    )
