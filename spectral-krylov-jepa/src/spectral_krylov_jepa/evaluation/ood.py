"""OOD evaluation orchestration."""

from __future__ import annotations

from typing import Any

from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint

OOD_SPLITS = ("test_ID", "test_OOD_narrow", "test_OOD_strong", "test_OOD_double")


def evaluate_ood_suite(
    *,
    checkpoint: str,
    data_path: str,
    manifest_path: str,
    config: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on all ID/OOD splits."""
    results = {}
    for split in OOD_SPLITS:
        results[split] = evaluate_checkpoint(
            checkpoint=checkpoint,
            data_path=data_path,
            manifest_path=manifest_path,
            split=split,
            config=config,
            output_dir=None if output_dir is None else f"{output_dir}/{split}",
        )
    return results
