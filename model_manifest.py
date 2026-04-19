"""Write reproducible model_manifest.json next to joblib artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_model_manifest(
    joblib_path: Path | str,
    manifest_path: Path | str,
    *,
    feature_columns: list[str],
    contamination: float,
    training_samples: int,
    training_seed: int,
    sklearn_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    joblib_path = Path(joblib_path)
    manifest_path = Path(manifest_path)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    h = sha256_file(joblib_path) if joblib_path.exists() else ""
    manifest: dict[str, Any] = {
        "model_file": joblib_path.name,
        "sklearn_version": sklearn_version,
        "training_seed": training_seed,
        "training_samples": training_samples,
        "feature_columns": feature_columns,
        "contamination": contamination,
        "timestamp_created": ts,
        "hash_sha256": h,
    }
    if extra:
        manifest["extra"] = extra
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
