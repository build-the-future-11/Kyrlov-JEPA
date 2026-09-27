"""Manifest helpers for datasets and runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectral_krylov_jepa.data.splits import read_manifest, write_manifest
from spectral_krylov_jepa.utils.io import project_root


def default_manifest_dir() -> Path:
    return project_root() / "experiments" / "manifests"


def save_dataset_manifest(name: str, manifest: dict[str, Any], directory: Path | None = None) -> Path:
    directory = directory or default_manifest_dir()
    path = directory / f"{name}.json"
    write_manifest(path, manifest)
    return path


def load_dataset_manifest(name: str, directory: Path | None = None) -> dict[str, Any]:
    directory = directory or default_manifest_dir()
    return read_manifest(directory / f"{name}.json")
