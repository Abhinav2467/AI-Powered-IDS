"""
Edge AI Detection Engine (Simulated Gateway ECU)
=================================================
Isolation Forest + StandardScaler in a single Pipeline (ONNX-exportable).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_manifest import write_model_manifest

# ── Feature columns used for detection (must match training data) ──
FEATURE_COLS = [
    "can_id",
    "payload_byte_0",
    "payload_byte_1",
    "payload_byte_2",
    "payload_byte_3",
    "payload_byte_4",
    "payload_byte_5",
    "payload_byte_6",
    "payload_byte_7",
    "inter_arrival_time",
    "iat_rolling_mean",
    "iat_rolling_std",
]


class EdgeAIDetector:
    """Isolation Forest anomaly detector (Pipeline: StandardScaler → IsolationForest)."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.pipeline: Pipeline | None = None
        self._contamination = contamination
        self._n_estimators = n_estimators
        self.is_trained = False
        self.training_time = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=self._n_estimators,
                        contamination=self._contamination,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}. Apply rolling IAT features in can_generator.")
        return df[FEATURE_COLS].values

    def train(self, normal_traffic_df: pd.DataFrame, verbose: bool = True) -> dict:
        if verbose:
            print("[*] Training Edge AI Detection Engine...")
        start = time.time()
        assert self.pipeline is not None
        X_train = self.extract_features(normal_traffic_df)
        self.pipeline.fit(X_train)
        self.is_trained = True
        self.training_time = time.time() - start

        train_scores = self.pipeline.decision_function(X_train)

        stats = {
            "training_samples": len(normal_traffic_df),
            "training_time_ms": round(self.training_time * 1000, 2),
            "mean_normal_score": round(float(np.mean(train_scores)), 4),
            "std_normal_score": round(float(np.std(train_scores)), 4),
            "model_params": {
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
            },
        }
        if verbose:
            print(f"[✓] Model trained in {stats['training_time_ms']}ms")
            print(f"[✓] Normal score baseline: {stats['mean_normal_score']} ± {stats['std_normal_score']}")
        return stats

    def predict(
        self,
        traffic_df: pd.DataFrame,
        default_simulated_stack_us: float = 0.0,
    ) -> pd.DataFrame:
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.extract_features(traffic_df)
        results = traffic_df.copy()

        if "simulated_stack_latency_us" not in results.columns:
            results["simulated_stack_latency_us"] = float(default_simulated_stack_us)

        batch_start = time.perf_counter_ns()
        scores = self.pipeline.decision_function(X)
        labels = self.pipeline.predict(X)
        batch_time_us = (time.perf_counter_ns() - batch_start) / 1000
        per_msg_edge_us = batch_time_us / max(len(X), 1)

        results["edge_processing_latency_us"] = per_msg_edge_us
        results["total_path_latency_us"] = (
            results["simulated_stack_latency_us"].astype(float) + per_msg_edge_us
        )
        results["anomaly_score"] = scores
        results["anomaly_label"] = labels
        results["detection_latency_us"] = results["total_path_latency_us"]

        score_array = np.array(scores)
        min_s, max_s = score_array.min(), score_array.max()
        if max_s - min_s > 0:
            results["confidence"] = 1 - (score_array - min_s) / (max_s - min_s)
        else:
            results["confidence"] = 0.5

        results["detected_anomaly"] = (results["anomaly_label"] == -1).astype(int)
        return results

    def evaluate(self, results_df: pd.DataFrame, verbose: bool = True) -> dict:
        y_true = results_df["is_malicious"]
        y_pred = results_df["detected_anomaly"]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        def q(col: str, qv: float) -> float | None:
            if col not in results_df.columns:
                return None
            return round(float(results_df[col].quantile(qv)), 2)

        metrics: dict[str, Any] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            "detection_rate": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "avg_detection_latency_us": round(results_df["detection_latency_us"].mean(), 2),
            "max_detection_latency_us": round(results_df["detection_latency_us"].max(), 2),
            "p50_detection_latency_us": q("detection_latency_us", 0.5),
            "p99_detection_latency_us": q("detection_latency_us", 0.99),
            "p50_edge_processing_us": q("edge_processing_latency_us", 0.5),
            "p99_edge_processing_us": q("edge_processing_latency_us", 0.99),
            "avg_edge_processing_latency_us": round(results_df["edge_processing_latency_us"].mean(), 2)
            if "edge_processing_latency_us" in results_df.columns
            else None,
            "avg_simulated_stack_latency_us": round(
                results_df["simulated_stack_latency_us"].mean(), 2
            )
            if "simulated_stack_latency_us" in results_df.columns
            else None,
            "avg_total_path_latency_us": round(results_df["total_path_latency_us"].mean(), 2)
            if "total_path_latency_us" in results_df.columns
            else None,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("  EDGE AI DETECTION ENGINE — RESULTS")
            print("=" * 50)
            print(f"  Accuracy:           {metrics['accuracy']:.2%}")
            print(f"  Precision:          {metrics['precision']:.2%}")
            print(f"  Recall:             {metrics['recall']:.2%}")
            print(f"  F1 Score:           {metrics['f1_score']:.2%}")
            print(f"  Detection Rate:     {metrics['detection_rate']:.2%}")
            print(f"  False Positive Rate: {metrics['false_positive_rate']:.2%}")
            print(f"  Avg Total Path:     {metrics['avg_detection_latency_us']:.1f} μs")
            print(f"  p50 / p99 path:     {metrics['p50_detection_latency_us']} / {metrics['p99_detection_latency_us']} μs")
            if metrics.get("avg_edge_processing_latency_us") is not None:
                print(f"  Avg Edge (IF):      {metrics['avg_edge_processing_latency_us']:.1f} μs")
                print(
                    f"  p50 / p99 edge:     {metrics['p50_edge_processing_us']} / {metrics['p99_edge_processing_us']} μs"
                )
            if metrics.get("avg_simulated_stack_latency_us") is not None:
                print(f"  Avg Sim. Stack:     {metrics['avg_simulated_stack_latency_us']:.1f} μs")
            print("=" * 50)

        return metrics

    def save_model(
        self,
        path: str = "edge_model.joblib",
        *,
        manifest_path: str | Path | None = None,
        training_samples: int = 0,
        training_seed: int = 42,
        write_manifest: bool = True,
    ) -> None:
        assert self.pipeline is not None
        payload = {
            "pipeline": self.pipeline,
            "is_trained": self.is_trained,
            "feature_cols": FEATURE_COLS,
            "contamination": self._contamination,
        }
        joblib.dump(payload, path)
        print(f"[✓] Model saved to {path}")

        if write_manifest and manifest_path:
            mp = Path(manifest_path)
            write_model_manifest(
                Path(path),
                mp,
                feature_columns=FEATURE_COLS,
                contamination=self._contamination,
                training_samples=training_samples,
                training_seed=training_seed,
                sklearn_version=sklearn.__version__,
            )
            print(f"[✓] Manifest written to {mp}")

    def load_model(self, path: str = "edge_model.joblib") -> None:
        data = joblib.load(path)
        if "pipeline" in data:
            self.pipeline = data["pipeline"]
            self._contamination = data.get("contamination", self._contamination)
        else:
            # Legacy: separate scaler + model
            self._build_pipeline()
            assert self.pipeline is not None
            self.pipeline.named_steps["scaler"] = data["scaler"]
            self.pipeline.named_steps["iforest"] = data["model"]
        self.is_trained = data.get("is_trained", True)
        print(f"[✓] Model loaded from {path} — Edge AI ready")

    def export_onnx(self, path: str = "edge_model.onnx") -> bool:
        """Export Pipeline to ONNX for edge deployment (optional dependency)."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            print("[!] skl2onnx not installed; skip ONNX export. pip install skl2onnx onnx")
            return False
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Train model before ONNX export.")
        n_features = len(FEATURE_COLS)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(
            self.pipeline,
            initial_types=initial_type,
            target_opset={"": 17, "ai.onnx.ml": 3},
        )
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"[✓] ONNX model written to {path} ({n_features} features)")
        return True


if __name__ == "__main__":
    import config
    from can_generator import generate_dataset

    train_df, test_df = generate_dataset(
        normal_count=1000,
        attack_count=50,
        attack_type="injection",
    )
    detector = EdgeAIDetector(contamination=0.05)
    detector.train(train_df)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    detector.save_model(
        str(config.MODELS_DIR / "edge_model.joblib"),
        manifest_path=config.MODEL_MANIFEST_PATH,
        training_samples=len(train_df),
    )
    detector.export_onnx(str(config.ONNX_MODEL_PATH))
    detector2 = EdgeAIDetector()
    detector2.load_model(str(config.MODELS_DIR / "edge_model.joblib"))
    results = detector2.predict(test_df)
    metrics = detector2.evaluate(results)
    results.to_csv("detection_results.csv", index=False)
    print("\n[✓] Results saved to detection_results.csv")
