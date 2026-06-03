"""PennyLane GPU device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pennylane as qml


@dataclass(frozen=True)
class DeviceSelection:
    """Selected PennyLane device."""

    device_name: str
    device: Any


def create_device(wires: int) -> DeviceSelection:
    """Create the required PennyLane GPU device."""
    try:
        device_name = "lightning.gpu"
        return DeviceSelection(device_name=device_name, device=qml.device(device_name, wires=wires))
    except Exception as exc:
        raise RuntimeError(
            "PennyLane runs in this repo require an available GPU via the "
            "'lightning.gpu' device.\n"
            f"lightning.gpu: {type(exc).__name__}: {exc}"
        ) from exc
