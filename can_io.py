"""
CAN bus I/O: optional SocketCAN capture → same DataFrame schema as synthetic generator.

Requires: pip install python-can
Linux: ip link set can0 up type can bitrate 500000  (or use virtual CAN vcan0)
macOS: typically no native SocketCAN; use synthetic mode or a USB-CAN adapter with slcan.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


def _require_can() -> Any:
    try:
        import can  # type: ignore
    except ImportError as e:
        raise ImportError(
            "SocketCAN capture requires the 'python-can' package. "
            "Install with: pip install python-can"
        ) from e
    return can


def message_to_row(msg: Any, prev_ts: float | None) -> dict[str, Any]:
    """Map a python-can Message to CAN-Guard row schema."""
    can_id = int(msg.arbitration_id)
    data = list(msg.data)
    while len(data) < 8:
        data.append(0)
    data = data[:8]
    ts = getattr(msg, "timestamp", None)
    if ts is None:
        ts = time.time()
    interval = 0.005 if prev_ts is None else max(ts - prev_ts, 1e-6)
    return {
        "timestamp": ts,
        "can_id": can_id,
        "can_id_hex": hex(can_id),
        "ecu_name": f"CAPTURED_0x{can_id:03X}",
        "dlc": int(msg.dlc),
        "payload": data,
        **{f"payload_byte_{j}": data[j] for j in range(8)},
        "inter_arrival_time": interval,
        "is_malicious": 0,
    }


def capture_socketcan_to_dataframe(
    interface: str = "can0",
    bustype: str = "socketcan",
    duration_s: float = 3.0,
    max_frames: int = 5000,
) -> pd.DataFrame:
    """
    Record live CAN traffic into a DataFrame suitable for EdgeAIDetector.

    Ground-truth label is_malicious=0 (unknown attacks unless you label offline).
    """
    can = _require_can()
    try:
        bus = can.Bus(channel=interface, interface=bustype)
    except Exception:
        bus = can.interface.Bus(channel=interface, bustype=bustype)
    rows: list[dict[str, Any]] = []
    t_end = time.time() + duration_s
    prev_ts: float | None = None
    try:
        while time.time() < t_end and len(rows) < max_frames:
            msg = bus.recv(timeout=0.2)
            if msg is None:
                continue
            row = message_to_row(msg, prev_ts)
            prev_ts = row["timestamp"]
            rows.append(row)
    finally:
        bus.shutdown()
    if not rows:
        raise RuntimeError(
            f"No CAN frames captured on {interface}. Check interface, bitrate, and permissions."
        )
    return pd.DataFrame(rows)


def capture_with_synthetic_labels(
    normal_df: pd.DataFrame,
    attack_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge normal + attack captures for testing (attack rows should have is_malicious=1).
    """
    return pd.concat([normal_df, attack_df], ignore_index=True).sort_values(
        "timestamp"
    ).reset_index(drop=True)


def augment_capture_with_injection(
    capture_df: pd.DataFrame,
    num_injections: int = 20,
    brake_id: int = 0x200,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For demos on live captures with no labels: inject synthetic malicious brake frames.
    Returns (mixed_test_df, labels_only_malicious_df) for metrics.
    """
    from can_generator import ATTACK_PAYLOAD_PATTERNS

    rng = np.random.default_rng(seed)
    base = capture_df.copy()
    n = len(base)
    if n == 0:
        raise ValueError("Empty capture")
    idxs = rng.choice(n, size=min(num_injections, n), replace=False)
    inject_rows = []
    for i in idxs:
        row = base.iloc[i].to_dict()
        t = float(row["timestamp"]) + 1e-4
        payload = list(ATTACK_PAYLOAD_PATTERNS[0])
        inject_rows.append(
            {
                "timestamp": t,
                "can_id": brake_id,
                "can_id_hex": hex(brake_id),
                "ecu_name": "SPOOFED_Brake",
                "dlc": 8,
                "payload": payload,
                **{f"payload_byte_{j}": payload[j] for j in range(8)},
                "inter_arrival_time": 0.0002,
                "is_malicious": 1,
            }
        )
    mal_df = pd.DataFrame(inject_rows)
    mixed = pd.concat([base, mal_df], ignore_index=True).sort_values("timestamp").reset_index(
        drop=True
    )
    return mixed, mal_df
