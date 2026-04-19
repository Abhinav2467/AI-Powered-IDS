"""
CAN-Guard AI — Main Pipeline
==============================
Complete AI-Driven Intrusion Detection System for Automotive CAN Bus.

Architecture:
    [Threat Model] → [Attack Simulator] → [Edge AI Detection]
    → [Safety Decision] → [Mitigation] → [Demo Output]
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

import config
from can_generator import add_inter_arrival_rolling_features, generate_dataset, generate_normal_traffic
from detection_engine import EdgeAIDetector
from gateway_simulator import apply_gateway_path_delay
from mitigation import MitigationSystem
from safety_layer import SafetyDecisionLayer
from threat_path import ThreatPathLogger


def _tune_contamination_on_validation(
    train_df: pd.DataFrame,
    contamination_default: float,
) -> tuple[float, dict[str, Any]]:
    """
    Hold-out validation on normal-only data: minimize false positive rate (unsupervised IF).
    F1 on all-normal labels is not meaningful; FPR on clean val is the right objective.
    """
    from sklearn.model_selection import train_test_split

    train_sub, val_sub = train_test_split(train_df, test_size=0.2, random_state=42)
    candidates = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
    if contamination_default not in candidates:
        candidates = sorted(set(candidates + [contamination_default]))
    best_c = contamination_default
    best_fpr = float("inf")
    best_metrics: dict[str, Any] = {}
    for c in candidates:
        d = EdgeAIDetector(contamination=c)
        d.train(train_sub, verbose=False)
        vr = d.predict(val_sub)
        m = d.evaluate(vr, verbose=False)
        fpr = m["false_positive_rate"]
        if fpr < best_fpr:
            best_fpr = fpr
            best_c = c
            best_metrics = m
    print(
        f"[✓] Validation tuning: best contamination={best_c} "
        f"(val FPR={best_fpr:.4f}, val accuracy={best_metrics.get('accuracy', 0):.4f})"
    )
    return best_c, {
        "best_contamination": best_c,
        "val_false_positive_rate": best_fpr,
        "val_metrics": best_metrics,
        "candidates_evaluated": candidates,
        "objective": "minimize FPR on normal-only validation split",
    }


def run_full_pipeline(
    normal_count: int = 1000,
    attack_count: int = 50,
    attack_type: str = "injection",
    contamination: float = 0.05,
    mode: str = "synthetic",
    socketcan_interface: str = "can0",
    socketcan_bustype: str = "socketcan",
    socketcan_duration_s: float | None = None,
    socketcan_max_frames: int | None = None,
    use_pretrained: bool = False,
    retrain: bool = False,
    prefer_pqc_signing: bool = False,
    stack_delay_mean_us: float | None = None,
    stack_delay_jitter_us: float | None = None,
    skip_gateway_delay: bool = False,
    tune_contamination: bool = True,
) -> dict[str, Any]:
    """
    Execute the complete CAN-Guard AI pipeline.
    Returns comprehensive results for demo output.

    mode:
      - synthetic: generated normal + injected attacks (labeled evaluation)
      - socketcan_inject: live capture + synthetic labeled injections (needs python-can + CAN iface)
    """
    stack_delay_mean_us = stack_delay_mean_us or config.DEFAULT_SIMULATED_STACK_DELAY_US_MEAN
    stack_delay_jitter_us = stack_delay_jitter_us or config.DEFAULT_SIMULATED_STACK_DELAY_US_JITTER
    socketcan_duration_s = socketcan_duration_s or config.DEFAULT_SOCKETCAN_DURATION_S
    socketcan_max_frames = socketcan_max_frames or config.DEFAULT_SOCKETCAN_MAX_FRAMES

    print("╔" + "═" * 58 + "╗")
    print("║     CAN-GUARD AI — Automotive Intrusion Detection        ║")
    print("║     Edge AI · Safety-First · Quantum-Safe Ready          ║")
    print("╚" + "═" * 58 + "╝")

    pipeline_start = time.time()
    threat_logger = ThreatPathLogger()

    print("\n▶ STAGE 1: Threat Model + Traffic / Attack Simulation")
    print("  Threat Model:")
    print("    Entry:  Infotainment system")
    print("    Path:   Infotainment → Gateway ECU → CAN Bus (lateral movement logged)")
    print("    Target: Braking ECU (CAN ID 0x200)")
    print(f"    Attack: {attack_type}")
    print(f"    Mode:   {mode}")

    train_df = generate_normal_traffic(normal_count, seed=42)

    if mode == "synthetic":
        _, test_df = generate_dataset(
            normal_count, attack_count, attack_type, threat_logger=threat_logger
        )
    elif mode == "socketcan_inject":
        try:
            from can_io import augment_capture_with_injection, capture_socketcan_to_dataframe

            cap = capture_socketcan_to_dataframe(
                interface=socketcan_interface,
                bustype=socketcan_bustype,
                duration_s=socketcan_duration_s,
                max_frames=socketcan_max_frames,
            )
            test_df, mal_df = augment_capture_with_injection(
                cap, num_injections=attack_count, seed=99
            )
            test_df = add_inter_arrival_rolling_features(test_df)
            for _, row in mal_df.iterrows():
                threat_logger.record_attack_chain(int(row["can_id"]), attack_label=attack_type)
        except Exception as e:
            print(
                f"[!] SocketCAN path failed ({e!r}); falling back to synthetic traffic."
            )
            _, test_df = generate_dataset(
                normal_count, attack_count, attack_type, threat_logger=threat_logger
            )
            mode = "synthetic+fallback"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not skip_gateway_delay:
        test_df = apply_gateway_path_delay(
            test_df,
            mean_us=stack_delay_mean_us,
            jitter_us=stack_delay_jitter_us,
            seed=202,
        )

    with open("threat_path_events.json", "w") as tf:
        json.dump(threat_logger.to_json_ready(), tf, indent=2)
    print("[✓] Lateral movement events → threat_path_events.json")

    print("\n▶ STAGE 2: Edge AI Detection Engine")

    pretrained_meta: dict[str, Any] = {}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    use_path = config.PRETRAINED_MODEL_PATH
    runtime_model_path = "edge_model.joblib"
    val_tune_info: dict[str, Any] = {}

    if use_pretrained and use_path.exists() and not retrain:
        runtime_model_path = str(use_path)
        pretrained_meta = {"loaded_from": runtime_model_path, "retrained": False}
        print("[*] Using bundled pretrained model (no training pass).")
    else:
        eff_contamination = contamination
        if tune_contamination:
            eff_contamination, val_tune_info = _tune_contamination_on_validation(
                train_df, contamination
            )
        trainer = EdgeAIDetector(contamination=eff_contamination)
        train_stats = trainer.train(train_df)
        trainer.save_model(
            runtime_model_path,
            manifest_path=config.MODEL_MANIFEST_PATH,
            training_samples=len(train_df),
            training_seed=42,
        )
        trainer.export_onnx(str(config.ONNX_MODEL_PATH))
        pretrained_meta = {
            "retrained": True,
            "train_stats": train_stats,
            "contamination_used": eff_contamination,
            "validation_tuning": val_tune_info if tune_contamination else None,
            "manifest": str(config.MODEL_MANIFEST_PATH),
            "onnx_export": str(config.ONNX_MODEL_PATH),
        }
        if use_pretrained and not use_path.exists():
            print(
                "[!] Pretrained file missing; trained fresh. "
                f"Run: python3 scripts/build_pretrained_model.py → {use_path}"
            )

    edge_runtime = EdgeAIDetector()
    edge_runtime.load_model(runtime_model_path)

    print("[*] Running detection on test traffic (gateway stack + edge IF latency)...")
    t0 = time.time()
    results = edge_runtime.predict(test_df, default_simulated_stack_us=0.0)
    wall_clock_detection_s = time.time() - t0

    detection_metrics = edge_runtime.evaluate(results)
    detection_metrics["wall_clock_batch_detection_s"] = round(wall_clock_detection_s, 4)

    print("\n▶ STAGE 3: Safety Decision Layer")
    safety = SafetyDecisionLayer()
    decisions = safety.process_batch(results)
    safety_summary = safety.get_summary()
    safety.export_log("safety_decisions.json")

    print("\n▶ STAGE 4: Mitigation System")
    mitigation = MitigationSystem(prefer_pqc=prefer_pqc_signing)
    mitigation_summary = mitigation.process_safety_decisions(decisions)
    mitigation.export_incidents("incidents.json")
    mitigation.export_alerts("alerts.json")
    mitigation.verify_all_incidents()

    pipeline_time = time.time() - pipeline_start

    before_state = {
        "status": "VULNERABLE",
        "brake_ecu_protection": "NONE",
        "intrusion_detection": "NONE",
        "attack_success_rate": "100%",
        "response_time": "N/A (no detection)",
    }

    after_state = {
        "status": "PROTECTED",
        "brake_ecu_protection": "AI-monitored + Safety Layer",
        "intrusion_detection": "Edge AI Isolation Forest",
        "attack_detection_rate": f"{detection_metrics['detection_rate']:.1%}",
        "false_positive_rate": f"{detection_metrics['false_positive_rate']:.1%}",
        "avg_total_path_latency_us": f"{detection_metrics['avg_detection_latency_us']:.1f} μs",
        "avg_edge_processing_latency_us": f"{detection_metrics.get('avg_edge_processing_latency_us') or 0:.1f} μs",
        "avg_simulated_stack_latency_us": f"{detection_metrics.get('avg_simulated_stack_latency_us') or 0:.1f} μs",
        "response": "Block + Controlled Braking (Safe Mode)",
    }

    demo_output: dict[str, Any] = {
        "pipeline_execution_time_s": round(pipeline_time, 2),
        "run_mode": mode,
        "pretrained_model": pretrained_meta,
        "threat_model": {
            "entry": "Infotainment system",
            "path": "Infotainment → Gateway ECU → CAN Bus",
            "target": "Braking ECU (0x200)",
            "attack_type": attack_type,
        },
        "lateral_movement": {
            "summary": threat_logger.summary(),
            "events_file": "threat_path_events.json",
        },
        "dataset": {
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "malicious_samples": int(test_df["is_malicious"].sum()),
        },
        "latency_model": {
            "stack_delay_mean_us": stack_delay_mean_us,
            "stack_delay_jitter_us": stack_delay_jitter_us,
            "edge_vs_stack": {
                "avg_edge_processing_us": detection_metrics.get("avg_edge_processing_latency_us"),
                "avg_simulated_stack_us": detection_metrics.get("avg_simulated_stack_latency_us"),
                "avg_total_path_us": detection_metrics.get("avg_total_path_latency_us"),
                "wall_clock_batch_detection_s": detection_metrics.get("wall_clock_batch_detection_s"),
            },
        },
        "detection_metrics": detection_metrics,
        "safety_summary": safety_summary,
        "mitigation_summary": mitigation_summary,
        "before_vs_after": {
            "before": before_state,
            "after": after_state,
        },
    }

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║              DEMO OUTPUT — BEFORE vs AFTER                ║")
    print("╠" + "═" * 58 + "╣")
    print("║  BEFORE (No Protection):                                  ║")
    print("║    Brake ECU Protection:  NONE                            ║")
    print("║    Intrusion Detection:   NONE                            ║")
    print("║    Attack Success Rate:   100%                            ║")
    print("╠" + "═" * 58 + "╣")
    print("║  AFTER (CAN-Guard AI):                                    ║")
    print(f"║    Detection Rate:        {detection_metrics['detection_rate']:.1%}                          ║")
    print(f"║    False Positive Rate:   {detection_metrics['false_positive_rate']:.1%}                          ║")
    ae = detection_metrics.get("avg_edge_processing_latency_us") or 0
    ass = detection_metrics.get("avg_simulated_stack_latency_us") or 0
    print(f"║    Avg Edge (IF) Latency:  {ae:.1f} μs                              ║")
    print(f"║    Avg Sim. Stack Lat.:   {ass:.1f} μs                              ║")
    print(f"║    Avg Total Path:        {detection_metrics['avg_detection_latency_us']:.1f} μs                       ║")
    print(f"║    Safe Mode Activations: {safety_summary['safe_mode_activations']}                              ║")
    print(f"║    Incidents Signed:      {mitigation_summary['total_incidents']}                             ║")
    print(f"║    Signing Algorithm:    {mitigation_summary.get('signing_algorithm', 'n/a')}                      ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Total pipeline execution: {pipeline_time:.2f}s")

    with open("demo_report.json", "w") as f:
        json.dump(demo_output, f, indent=2, default=str)
    print("[✓] Full demo report saved to demo_report.json")

    return demo_output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAN-Guard AI full pipeline")
    p.add_argument("--mode", choices=["synthetic", "socketcan_inject"], default="synthetic")
    p.add_argument("--normal-count", type=int, default=1000)
    p.add_argument("--attack-count", type=int, default=50)
    p.add_argument("--attack-type", default="injection")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--socketcan-interface", default="can0")
    p.add_argument("--socketcan-bustype", default="socketcan")
    p.add_argument("--socketcan-duration", type=float, default=config.DEFAULT_SOCKETCAN_DURATION_S)
    p.add_argument("--socketcan-max-frames", type=int, default=config.DEFAULT_SOCKETCAN_MAX_FRAMES)
    p.add_argument("--pretrained", action="store_true", help=f"Load {config.PRETRAINED_MODEL_PATH} if present")
    p.add_argument("--retrain", action="store_true", help="Ignore pretrained; train a new model")
    p.add_argument("--prefer-pqc-signing", action="store_true", help="Use liboqs if installed")
    p.add_argument("--stack-mean-us", type=float, default=config.DEFAULT_SIMULATED_STACK_DELAY_US_MEAN)
    p.add_argument("--stack-jitter-us", type=float, default=config.DEFAULT_SIMULATED_STACK_DELAY_US_JITTER)
    p.add_argument("--skip-gateway-delay", action="store_true")
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip validation-set contamination tuning (faster; uses --contamination as-is).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_pipeline(
        normal_count=args.normal_count,
        attack_count=args.attack_count,
        attack_type=args.attack_type,
        contamination=args.contamination,
        mode=args.mode,
        socketcan_interface=args.socketcan_interface,
        socketcan_bustype=args.socketcan_bustype,
        socketcan_duration_s=args.socketcan_duration,
        socketcan_max_frames=args.socketcan_max_frames,
        use_pretrained=args.pretrained,
        retrain=args.retrain,
        prefer_pqc_signing=args.prefer_pqc_signing,
        stack_delay_mean_us=args.stack_mean_us,
        stack_delay_jitter_us=args.stack_jitter_us,
        skip_gateway_delay=args.skip_gateway_delay,
        tune_contamination=not args.no_tune,
    )
