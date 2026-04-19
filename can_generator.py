"""
CAN Traffic Generator + Attack Simulator
=========================================
Generates normal CAN bus traffic and injects malicious brake signals.
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime


# ── CAN Bus Configuration ──
NORMAL_CAN_IDS = {
    0x100: "Engine_RPM",
    0x120: "Throttle_Position",
    0x140: "Steering_Angle",
    0x160: "Wheel_Speed",
    0x180: "Transmission",
    0x1A0: "ABS_Status",
    0x1C0: "Airbag_Status",
    0x200: "Brake_Pedal",       # Legitimate brake ECU
    0x220: "Infotainment",
    0x240: "Climate_Control",
}

# Attack configuration
MALICIOUS_BRAKE_ID = 0x200      # Spoofed brake signal
ATTACK_PAYLOAD_PATTERNS = [
    [0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],  # Emergency brake
    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00],  # Brake disable
    [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x00],  # Fuzzing
]


def generate_normal_payload(can_id: int, speed_kmh: int = 0) -> list:
    """Generate realistic CAN payload for a given ID."""
    if can_id == 0x100:  # Engine RPM (800-6000)
        rpm = np.random.randint(800, 6000)
        return list(rpm.to_bytes(2, 'big')) + [0] * 6
    elif can_id == 0x200:  # Normal brake (0-100 pressure)
        pressure = np.random.randint(0, 100)
        return [pressure, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    elif can_id == 0x160:  # Wheel speed — encode configured speed if given
        speed = int(np.clip(speed_kmh, 0, 300)) if speed_kmh > 0 else np.random.randint(0, 200)
        return list(int(speed).to_bytes(2, 'big')) + [0] * 6
    else:
        return [np.random.randint(0, 100) for _ in range(8)]


def generate_normal_traffic(num_messages: int = 1000, seed: int = 42, speed_kmh: int = 0) -> pd.DataFrame:
    """
    Generate normal CAN bus traffic dataset.
    
    ``speed_kmh`` — when > 0, 0x160 (Wheel_Speed) frames consistently encode this speed
    so the SafetyDecisionLayer correctly reads the driving regime from the start.
    """
    np.random.seed(seed)
    messages = []
    base_time = time.time()
    
    for i in range(num_messages):
        can_id = np.random.choice(list(NORMAL_CAN_IDS.keys()))
        payload = generate_normal_payload(can_id, speed_kmh=speed_kmh)
        # Normal inter-arrival: 1-10ms
        interval = np.random.uniform(0.001, 0.01)
        base_time += interval
        
        messages.append({
            "timestamp": base_time,
            "can_id": can_id,
            "can_id_hex": hex(can_id),
            "ecu_name": NORMAL_CAN_IDS[can_id],
            "dlc": 8,
            "payload": payload,
            **{f"payload_byte_{j}": payload[j] for j in range(8)},
            "inter_arrival_time": interval,
            "is_malicious": 0,
        })
    
    df = pd.DataFrame(messages)
    return add_inter_arrival_rolling_features(df)


def add_inter_arrival_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Time-ordered rolling stats on inter-arrival time (reduces false positives on bursty legit traffic).
    """
    out = df.sort_values("timestamp").reset_index(drop=True)
    s = out["inter_arrival_time"]
    out["iat_rolling_mean"] = s.rolling(window, min_periods=1).mean().fillna(0)
    out["iat_rolling_std"] = s.rolling(window, min_periods=1).std().fillna(0)
    return out


def inject_attack_traffic(
    normal_df: pd.DataFrame,
    num_attacks: int = 50,
    attack_type: str = "injection",
    threat_logger=None,
) -> pd.DataFrame:
    """
    Inject malicious CAN messages into normal traffic.
    
    Attack types:
        - injection: Spoofed brake signals at abnormal frequency
        - replay: Replay captured brake messages rapidly
        - fuzzing: Random payloads on brake CAN ID
    """
    np.random.seed(99)
    attack_messages = []
    
    # Pick random insertion points
    insert_indices = np.random.choice(len(normal_df), num_attacks, replace=False)
    insert_indices.sort()
    
    for idx in insert_indices:
        base_time = normal_df.iloc[idx]["timestamp"]
        
        if attack_type == "injection":
            payload = ATTACK_PAYLOAD_PATTERNS[0]  # Emergency brake
            interval = np.random.uniform(0.0001, 0.001)  # Abnormally fast
            atk_can_id = MALICIOUS_BRAKE_ID
            atk_ecu_name = "SPOOFED_Brake"
        elif attack_type == "replay":
            payload = ATTACK_PAYLOAD_PATTERNS[1]  # Brake disable
            interval = np.random.uniform(0.0001, 0.0005)  # Very rapid replay
            atk_can_id = MALICIOUS_BRAKE_ID
            atk_ecu_name = "SPOOFED_Brake"
        elif attack_type == "fuzzing":
            payload = [np.random.randint(0, 255) for _ in range(8)]
            interval = np.random.uniform(0.0005, 0.002)
            atk_can_id = np.random.randint(0x000, 0x800)  # Pure Fuzzing CAN ID
            atk_ecu_name = "FUZZED_Unknown"
        else:
            payload = ATTACK_PAYLOAD_PATTERNS[0]
            interval = 0.0005
            atk_can_id = MALICIOUS_BRAKE_ID
            atk_ecu_name = "SPOOFED_Unknown"
        
        if threat_logger is not None:
            threat_logger.record_attack_chain(atk_can_id, attack_label=attack_type)

        attack_messages.append({
            "timestamp": base_time + interval,
            "can_id": atk_can_id,
            "can_id_hex": hex(atk_can_id),
            "ecu_name": atk_ecu_name,
            "dlc": 8,
            "payload": payload,
            **{f"payload_byte_{j}": payload[j] for j in range(8)},
            "inter_arrival_time": interval,
            "is_malicious": 1,
        })
    
    attack_df = pd.DataFrame(attack_messages)
    combined = pd.concat([normal_df, attack_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined = add_inter_arrival_rolling_features(combined)

    return combined


def inject_high_speed_brake_attack(
    normal_df: pd.DataFrame,
    num_attacks: int = 50,
    speed_kmh: int = 120,
    threat_logger=None,
) -> pd.DataFrame:
    """
    High-speed brake injection — only injects ``num_attacks`` malicious brake frames
    (0x200, payload 0xFF×8, sub-millisecond IAT).
    Speed context is already encoded in normal 0x160 frames by generate_normal_traffic,
    so no extra speed frames are added here.
    Total output = len(normal_df) + num_attacks (exactly what the UI shows).
    """
    np.random.seed(101)
    n = len(normal_df)
    if n == 0:
        return normal_df

    na = min(num_attacks, n)
    insert_attack = np.random.choice(n, na, replace=False)

    attacks: list[dict] = []
    for idx in sorted(insert_attack):
        base_time = float(normal_df.iloc[idx]["timestamp"])
        interval = np.random.uniform(0.0001, 0.0003)
        ts = base_time + interval
        payload = [0xFF] * 8
        atk_can_id = MALICIOUS_BRAKE_ID
        if threat_logger is not None:
            threat_logger.record_attack_chain(atk_can_id, attack_label="high_speed_brake_injection")
        attacks.append(
            {
                "timestamp": ts,
                "can_id": atk_can_id,
                "can_id_hex": hex(atk_can_id),
                "ecu_name": "SPOOFED_Brake",
                "dlc": 8,
                "payload": payload,
                **{f"payload_byte_{j}": payload[j] for j in range(8)},
                "inter_arrival_time": interval,
                "is_malicious": 1,
            }
        )

    attack_df = pd.DataFrame(attacks)
    combined = pd.concat([normal_df, attack_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return add_inter_arrival_rolling_features(combined)


def generate_dataset(
    normal_count: int = 1000,
    attack_count: int = 50,
    attack_type: str = "injection",
    threat_logger=None,
    speed_kmh: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate complete dataset: train (normal only) + test (normal + attacks).

    ``normal_count`` is the EXACT number of normal frames that appear in the
    test simulation — what you set in the UI is what you see processed.
    Training always uses a fixed 1000-frame baseline (seed=0) so model
    quality is independent of the UI slider.

    Returns:
        train_df: Normal traffic for model training (1000 frames, fixed)
        test_df:  Mixed traffic — exactly normal_count normal + attack_count attack
    """
    # Training: fixed 1000-frame baseline, independent of the UI slider.
    # This keeps model quality stable no matter what the user picks.
    train_df = generate_normal_traffic(1000, seed=0, speed_kmh=speed_kmh)

    # Test: use the FULL normal_count so the frame count matches the UI exactly.
    # attack injection requires at least attack_count slots in normal_df.
    test_normal_rows = max(normal_count, attack_count)
    test_normal = generate_normal_traffic(test_normal_rows, seed=123, speed_kmh=speed_kmh)
    if attack_type == "high_speed_brake_injection":
        test_df = inject_high_speed_brake_attack(
            test_normal, attack_count, speed_kmh=speed_kmh, threat_logger=threat_logger
        )
    else:
        test_df = inject_attack_traffic(
            test_normal, attack_count, attack_type, threat_logger=threat_logger
        )

    print(f"[+] Training set: {len(train_df)} normal messages (fixed baseline)")
    print(f"[+] Test set: {len(test_df)} messages ({normal_count} normal + {attack_count} attack)")
    print(f"[+] Attack type: {attack_type}")

    return train_df, test_df


if __name__ == "__main__":
    train, test = generate_dataset()
    train.to_csv("train_normal.csv", index=False)
    test.to_csv("test_mixed.csv", index=False)
    print("\n[✓] Datasets saved: train_normal.csv, test_mixed.csv")
    print(f"\nSample malicious messages:")
    print(test[test["is_malicious"] == 1].head())
