"""
CAN-Guard AI — Live Attack Demo
================================
Real-time frame-by-frame simulation for judge presentations.

Run from the hackathon2/ repo root:
    python3 live_demo.py

Controls (press Enter between scenarios):
    1. Injection attack on Brake ECU (0x200) — high-speed scenario
    2. Fuzzing attack (random CAN IDs)
    3. Replay attack (brake disable)
    4. Mixed attack — all types at once
    Q. Quit

Each frame prints live to the terminal so judges can watch packets
arrive, get scored, and get BLOCKED or ALLOWED in real time.
"""

from __future__ import annotations

import sys
import os
import time
import random
import numpy as np

# ── Allow imports from hackathon2/ root ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from can_generator import (
    generate_normal_traffic,
    inject_attack_traffic,
    inject_high_speed_brake_attack,
    add_inter_arrival_rolling_features,
    NORMAL_CAN_IDS,
    ATTACK_PAYLOAD_PATTERNS,
    MALICIOUS_BRAKE_ID,
)
from detection_engine import EdgeAIDetector
from gateway_simulator import apply_gateway_path_delay
from safety_layer import SafetyDecisionLayer, SpeedContextLayer
from mitigation import MitigationSystem
import config

# ── Terminal colour codes ─────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

FRAME_DELAY_S   = 0.06   # seconds between printing each CAN frame (tune for pacing)
ATTACK_DELAY_S  = 0.12   # slightly slower on attack frames so judges can read them
SHOW_NORMAL_MAX = 8      # how many normal frames to show before attacks start (keeps output tight)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def banner(title: str) -> None:
    width = 72
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    pad = (width - len(title)) // 2
    print(f"{BOLD}{CYAN}{' ' * pad}{title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


def action_colour(action: str) -> str:
    colours = {
        "ALLOW":     GREEN,
        "ALERT":     YELLOW,
        "BLOCK":     RED,
        "SAFE_MODE": RED + BOLD,
    }
    return colours.get(action, WHITE)


def print_frame(
    idx: int,
    row: dict,
    action: str,
    confidence: float,
    anomaly_score: float,
    is_malicious_true: int,
    reason: str,
) -> None:
    """Print one CAN frame with detection result."""
    col = action_colour(action)
    mal_tag = f"{RED}[ATTACK]{RESET}" if is_malicious_true else f"{GREEN}[NORMAL]{RESET}"
    ecu     = row.get("ecu_name", "?")
    can_hex = row.get("can_id_hex", hex(int(row["can_id"])))
    iat     = row.get("inter_arrival_time", 0)
    payload = [row.get(f"payload_byte_{i}", 0) for i in range(8)]
    payload_str = " ".join(f"{int(b):02X}" for b in payload)

    print(
        f"  {DIM}#{idx:04d}{RESET}  "
        f"{WHITE}{can_hex:6s}{RESET}  "
        f"{ecu:<22s}  "
        f"IAT={iat*1000:6.3f}ms  "
        f"payload=[{payload_str}]  "
        f"{mal_tag}  "
        f"{col}{BOLD}{action:<10s}{RESET}  "
        f"conf={confidence:.2f}  "
        f"score={anomaly_score:+.4f}"
    )
    if action in ("BLOCK", "SAFE_MODE"):
        print(f"         {DIM}↳ {reason[:90]}{RESET}")


def print_summary(safety, mitigation, elapsed_s: float) -> None:
    """Print a short post-run summary."""
    ss  = safety.get_summary()
    ms  = mitigation.get_summary()
    inc = len(mitigation.incidents)

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  RUN SUMMARY{RESET}")
    print(f"{'─'*72}")
    print(f"  Messages processed : {ss['total_messages']}")
    print(f"  {GREEN}Allowed            : {ss['allowed']}{RESET}")
    print(f"  {YELLOW}Alerts             : {ss['alerts']}{RESET}")
    print(f"  {RED}Blocked            : {ss['blocked']}{RESET}")
    print(f"  {RED}{BOLD}Safe-mode triggers : {ss['safe_mode_activations']}{RESET}")
    print(f"  Incidents signed   : {inc}  (all HMAC-SHA3-256 ✓)" if inc else "  Incidents signed   : 0")
    print(f"  Wall-clock time    : {elapsed_s:.2f}s")
    print(f"{'─'*72}\n")


# ── Model bootstrap ───────────────────────────────────────────────────────────

def load_or_train_model(contamination: float = 0.05) -> EdgeAIDetector:
    """Load pretrained model or train quickly on synthetic normal traffic."""
    pretrained = config.PRETRAINED_MODEL_PATH
    if pretrained.exists():
        print(f"  {GREEN}✓{RESET} Loading pretrained model from {pretrained}")
        det = EdgeAIDetector(contamination=contamination)
        det.load_model(str(pretrained))
        return det

    print(f"  {YELLOW}⚙{RESET}  No pretrained model found — training on 1 000 normal frames…")
    normal_df = generate_normal_traffic(1000, seed=42)
    det = EdgeAIDetector(contamination=contamination)
    det.train(normal_df, verbose=False)
    det.save_model("edge_model.joblib")
    print(f"  {GREEN}✓{RESET} Model ready.\n")
    return det


# ── Core live simulation loop ─────────────────────────────────────────────────

def run_live_scenario(
    detector: EdgeAIDetector,
    scenario_name: str,
    attack_type: str,
    normal_count: int = 500,
    attack_count: int = 40,
    speed_kmh: int = 120,
    frame_delay: float = FRAME_DELAY_S,
) -> None:
    """
    Generate a dataset, then replay it frame-by-frame with live detection output.
    """
    banner(f"SCENARIO: {scenario_name}")

    # ── Generate dataset ──────────────────────────────────────────────────────
    np.random.seed(42)
    normal_df = generate_normal_traffic(normal_count, seed=42)

    if attack_type == "high_speed_brake_injection":
        from can_generator import inject_high_speed_brake_attack
        test_df = inject_high_speed_brake_attack(normal_df, attack_count, speed_kmh=speed_kmh)
    else:
        test_df = inject_attack_traffic(normal_df, attack_count, attack_type)

    test_df = apply_gateway_path_delay(test_df, mean_us=48.0, jitter_us=12.0, seed=202)

    # ── Run ML detection (batch — fast) ─────────────────────────────────────
    results = detector.predict(test_df)

    # ── Safety layer and mitigation (live replay) ────────────────────────────
    speed_ctx = SpeedContextLayer(initial_speed=float(speed_kmh) if "brake" in attack_type else 0.0)
    safety    = SafetyDecisionLayer(zero_trust_enabled=True, speed_context=speed_ctx)
    mitigation = MitigationSystem()

    total        = len(results)
    normal_shown = 0
    attack_idx   = 0
    decisions    = []

    print(
        f"  {DIM}{'#':<6}  {'CAN ID':<6}  {'ECU':<22}  {'IAT':<13}  "
        f"{'payload':>26}  {'truth':<10}  {'ACTION':<10}  {'conf':<9}  score{RESET}"
    )
    print(f"  {DIM}{'─'*110}{RESET}")

    t_start = time.perf_counter()

    for idx, row in results.iterrows():
        row_dict = row.to_dict()
        decision = safety.decide(row_dict)
        decisions.append(decision)

        is_mal = int(row_dict.get("is_malicious", 0))
        action = decision.action.value

        # ── Throttle normal frames to avoid flooding the terminal ─────────────
        if not is_mal:
            normal_shown += 1
            if normal_shown > SHOW_NORMAL_MAX:
                # Only show every 10th normal frame after the initial batch
                if normal_shown % 10 != 0:
                    continue
            delay = frame_delay
        else:
            attack_idx += 1
            delay = ATTACK_DELAY_S

        print_frame(
            idx=idx,
            row=row_dict,
            action=action,
            confidence=float(row_dict.get("confidence", 0)),
            anomaly_score=float(row_dict.get("anomaly_score", 0)),
            is_malicious_true=is_mal,
            reason=decision.reason,
        )

        # Feed blocked/safe-mode decisions to mitigation
        if action in ("BLOCK", "SAFE_MODE", "ALERT"):
            mitigation.process_safety_decisions([decision])

        time.sleep(delay)

    elapsed = time.perf_counter() - t_start
    print_summary(safety, mitigation, elapsed)

    # Verify all signatures
    if mitigation.incidents:
        ok = all(
            mitigation.verify_signature(
                __import__("signing").canonical_incident_json(inc.__dict__ if hasattr(inc, "__dict__") else vars(inc)),
                inc.signature,
            )
            for inc in mitigation.incidents
            if inc.signature
        )
        sig_status = f"{GREEN}✓ All {len(mitigation.incidents)} incident signatures verified{RESET}" if ok else f"{RED}✗ Signature verification failed!{RESET}"
        print(f"  {sig_status}\n")


# ── Menu ──────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "1": (
        "BRAKE INJECTION @ 120 km/h  — spoofed 0xFF emergency-brake frames",
        "high_speed_brake_injection",
        120,
    ),
    "2": (
        "FUZZING ATTACK             — random CAN IDs, random payloads",
        "fuzzing",
        0,
    ),
    "3": (
        "REPLAY ATTACK              — rapid brake-disable replays on 0x200",
        "replay",
        0,
    ),
    "4": (
        "INJECTION ATTACK           — standard spoofed brake injection",
        "injection",
        0,
    ),
}


def main() -> None:
    clear()
    banner("CAN-GUARD AI  ·  Live Attack Demo")

    print(f"  {DIM}Loading detection model…{RESET}")
    detector = load_or_train_model(contamination=0.05)

    while True:
        print(f"\n{BOLD}  Choose a scenario:{RESET}")
        for key, (label, _, _) in SCENARIOS.items():
            print(f"    {CYAN}[{key}]{RESET}  {label}")
        print(f"    {CYAN}[A]{RESET}  Run ALL scenarios back-to-back")
        print(f"    {CYAN}[Q]{RESET}  Quit\n")

        choice = input("  > ").strip().upper()

        if choice == "Q":
            print(f"\n  {DIM}Exiting CAN-Guard demo.{RESET}\n")
            break
        elif choice == "A":
            for key, (name, atype, spd) in SCENARIOS.items():
                run_live_scenario(detector, name, atype, speed_kmh=spd)
                input(f"  {DIM}[Press Enter for next scenario…]{RESET}")
        elif choice in SCENARIOS:
            name, atype, spd = SCENARIOS[choice]
            run_live_scenario(detector, name, atype, speed_kmh=spd)
        else:
            print(f"  {YELLOW}Unknown option — try 1, 2, 3, 4, A or Q.{RESET}")


if __name__ == "__main__":
    main()
