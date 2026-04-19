#!/usr/bin/env python3
"""
Train a fixed Pipeline (Scaler + Isolation Forest) on deterministic synthetic normal traffic,
write models/pretrained.joblib, model_manifest.json, and edge_model.onnx when dependencies allow.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can_generator import generate_normal_traffic  # noqa: E402
from detection_engine import EdgeAIDetector, FEATURE_COLS  # noqa: E402
import config  # noqa: E402


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = generate_normal_traffic(2500, seed=42)
    det = EdgeAIDetector(contamination=0.05)
    det.train(train_df)
    out = config.PRETRAINED_MODEL_PATH
    det.save_model(
        str(out),
        manifest_path=config.MODEL_MANIFEST_PATH,
        training_samples=len(train_df),
        training_seed=42,
    )
    det.export_onnx(str(config.ONNX_MODEL_PATH))
    print(f"[✓] Features: {len(FEATURE_COLS)} cols")
    print(f"[✓] Pretrained edge model written to {out}")


if __name__ == "__main__":
    main()
