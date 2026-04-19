"""
Lateral movement model: Infotainment compromise → Gateway ECU → CAN (brake target).

Emits a structured event log for demos and forensics (not a network simulator).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass
class ThreatPathEvent:
    stage: str
    timestamp_iso: str
    detail: str
    can_id_hex: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("metadata") is None:
            d.pop("metadata", None)
        return d


class ThreatPathLogger:
    """
    Records the narrative path for each injected attack (synthetic or captured).
    """

    def __init__(self) -> None:
        self.events: list[ThreatPathEvent] = []

    def log_infotainment_compromise(self, vector: str = "compromised_app_plugin") -> None:
        self.events.append(
            ThreatPathEvent(
                stage="infotainment_compromise",
                timestamp_iso=datetime.now().isoformat(),
                detail=f"Initial foothold on infotainment — vector={vector}",
            )
        )

    def log_gateway_crossing(
        self,
        gateway: str = "Central_Gateway_ECU",
        lateral_technique: str = "diagnostic_tunnel_misuse",
    ) -> None:
        self.events.append(
            ThreatPathEvent(
                stage="gateway_crossing",
                timestamp_iso=datetime.now().isoformat(),
                detail=f"Lateral movement to {gateway} — technique={lateral_technique}",
            )
        )

    def log_can_injection(self, can_id: int, label: str = "brake_spoof") -> None:
        self.events.append(
            ThreatPathEvent(
                stage="can_injection",
                timestamp_iso=datetime.now().isoformat(),
                detail=f"Malicious frame injected toward safety domain — label={label}",
                can_id_hex=hex(can_id),
                metadata={"target": "Braking_ECU", "bus": "powertrain_chassis_CAN"},
            )
        )

    def record_attack_chain(self, can_id: int, attack_label: str = "brake_injection") -> None:
        """Convenience: full path for one attack."""
        self.log_infotainment_compromise()
        self.log_gateway_crossing()
        self.log_can_injection(can_id, label=attack_label)

    def to_json_ready(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def summary(self) -> dict[str, Any]:
        stages = [e.stage for e in self.events]
        return {
            "total_events": len(self.events),
            "stages_seen": list(dict.fromkeys(stages)),
            "attack_chains": stages.count("can_injection"),
        }
