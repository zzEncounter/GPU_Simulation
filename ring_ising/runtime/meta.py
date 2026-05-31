"""Small metadata and formatting helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version(package_name: str) -> str:
    """Return an installed package version or a readable fallback."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def format_gate_types(gate_types: dict[str, int]) -> str:
    """Format gate counts into a compact human-readable string."""
    return ", ".join(
        f"{gate_name}:{count}" for gate_name, count in sorted(gate_types.items())
    )
