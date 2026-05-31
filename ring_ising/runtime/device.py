"""PennyLane device-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pennylane as qml


DEVICE_CANDIDATES = {
    "auto": ("lightning.gpu", "lightning.qubit", "default.qubit"),
    "gpu": ("lightning.gpu",),
    "cpu": ("lightning.qubit", "default.qubit"),
    "default": ("default.qubit",),
}


@dataclass(frozen=True)
class DeviceSelection:
    """Selected PennyLane device plus any fallback notes."""

    requested_mode: str
    device_name: str
    device: Any
    selection_errors: tuple[str, ...]


def create_device(requested_mode: str, wires: int) -> DeviceSelection:
    """Create the requested PennyLane device with optional fallback."""
    if requested_mode not in DEVICE_CANDIDATES:
        choices = ", ".join(sorted(DEVICE_CANDIDATES))
        raise ValueError(f"Unknown device mode {requested_mode!r}. Expected one of: {choices}")

    selection_errors: list[str] = []
    for device_name in DEVICE_CANDIDATES[requested_mode]:
        try:
            return DeviceSelection(
                requested_mode=requested_mode,
                device_name=device_name,
                device=qml.device(device_name, wires=wires),
                selection_errors=tuple(selection_errors),
            )
        except Exception as exc:
            selection_errors.append(f"{device_name}: {type(exc).__name__}: {exc}")

    error_text = "\n".join(selection_errors) or "No device candidates were configured."
    raise RuntimeError(
        f"Unable to initialize a PennyLane device for mode {requested_mode!r}.\n{error_text}"
    )
