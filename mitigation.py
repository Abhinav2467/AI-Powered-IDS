"""
Mitigation System
=================
- Block suspicious CAN messages
- Log attack events
- Alert system
- Pluggable signing: HMAC-SHA3-256 (default) or liboqs PQC when installed
"""

from __future__ import annotations
import json
from datetime import datetime
from dataclasses import dataclass, asdict

from signing import SignatureProvider, build_signature_provider, canonical_incident_json


@dataclass
class IncidentReport:
    incident_id: str
    timestamp: str
    can_id: str
    ecu_name: str
    attack_type: str
    action_taken: str
    confidence: float
    anomaly_score: float
    is_safety_critical: bool
    signature: str = ""
    signature_algorithm: str = ""


class MitigationSystem:
    """
    Handles blocking, logging, alerting, and cryptographic signing
    of security incidents.
    """

    def __init__(
        self,
        signature_provider: SignatureProvider | None = None,
        prefer_pqc: bool = False,
    ):
        self._provider = signature_provider or build_signature_provider(
            prefer_pqc=prefer_pqc
        )
        self.blocked_ids: set = set()
        self.incidents: list[IncidentReport] = []
        self.alerts: list[dict] = []
        self.incident_counter = 0

    def sign_incident(self, data: str) -> str:
        """Sign canonical incident payload."""
        return self._provider.sign(data)

    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify incident report signature."""
        return self._provider.verify(data, signature)

    def block_can_id(self, can_id: int):
        """Add CAN ID to block list."""
        self.blocked_ids.add(can_id)

    def is_blocked(self, can_id: int) -> bool:
        """Check if CAN ID is currently blocked."""
        return can_id in self.blocked_ids

    def create_incident(
        self,
        can_id: str,
        ecu_name: str,
        action: str,
        confidence: float,
        anomaly_score: float,
        is_safety_critical: bool,
    ) -> IncidentReport:
        """Create and sign an incident report."""
        self.incident_counter += 1

        if confidence >= 0.75:
            attack_type = "CAN_INJECTION_HIGH_CONFIDENCE"
        elif confidence >= 0.50:
            attack_type = "CAN_ANOMALY_MEDIUM_CONFIDENCE"
        else:
            attack_type = "CAN_ANOMALY_LOW_CONFIDENCE"

        incident = IncidentReport(
            incident_id=f"INC-{self.incident_counter:04d}",
            timestamp=datetime.now().isoformat(),
            can_id=can_id,
            ecu_name=ecu_name,
            attack_type=attack_type,
            action_taken=action,
            confidence=round(confidence, 4),
            anomaly_score=round(anomaly_score, 4),
            is_safety_critical=is_safety_critical,
        )

        unsigned = asdict(incident)
        unsigned.pop("signature", None)
        unsigned.pop("signature_algorithm", None)
        payload = canonical_incident_json(unsigned)
        incident.signature = self._provider.sign(payload)
        incident.signature_algorithm = self._provider.algorithm_name

        self.incidents.append(incident)
        return incident

    def trigger_alert(self, incident: IncidentReport):
        """Trigger alert for incident."""
        alert = {
            "alert_id": f"ALT-{len(self.alerts) + 1:04d}",
            "incident_id": incident.incident_id,
            "severity": "CRITICAL" if incident.is_safety_critical else "HIGH",
            "timestamp": datetime.now().isoformat(),
            "message": (
                f"[{incident.attack_type}] Anomalous CAN message detected on "
                f"{incident.ecu_name} (ID: {incident.can_id}). "
                f"Action: {incident.action_taken}. "
                f"Confidence: {incident.confidence:.2%}."
            ),
        }
        self.alerts.append(alert)
        return alert

    def process_safety_decisions(self, decisions: list) -> dict:
        """
        Process all safety decisions into incidents and alerts.

        Args:
            decisions: List of SafetyDecision objects from SafetyDecisionLayer
        """
        for d in decisions:
            if d.action.value in ("BLOCK", "SAFE_MODE", "ALERT"):
                incident = self.create_incident(
                    can_id=d.can_id,
                    ecu_name=d.ecu_name,
                    action=d.action.value,
                    confidence=d.confidence,
                    anomaly_score=d.anomaly_score,
                    is_safety_critical=d.is_safety_critical,
                )

                if d.action.value in ("BLOCK", "SAFE_MODE"):
                    self.block_can_id(int(d.can_id, 16))

                self.trigger_alert(incident)

        return self.get_summary()

    def get_summary(self) -> dict:
        """Get mitigation system summary."""
        summary = {
            "total_incidents": len(self.incidents),
            "total_alerts": len(self.alerts),
            "blocked_can_ids": [hex(x) for x in self.blocked_ids],
            "critical_alerts": sum(1 for a in self.alerts if a["severity"] == "CRITICAL"),
            "all_signed": all(i.signature != "" for i in self.incidents),
            "signing_algorithm": self._provider.algorithm_name,
        }

        print("\n" + "=" * 50)
        print("  MITIGATION SYSTEM — SUMMARY")
        print("=" * 50)
        print(f"  Total Incidents:    {summary['total_incidents']}")
        print(f"  Total Alerts:       {summary['total_alerts']}")
        print(f"  Critical Alerts:    {summary['critical_alerts']}")
        print(f"  Blocked CAN IDs:    {summary['blocked_can_ids']}")
        print(f"  All Reports Signed: {summary['all_signed']}")
        print(f"  Signing Algorithm:  {summary['signing_algorithm']}")
        print("=" * 50)

        return summary

    def export_incidents(self, path: str = "incidents.json"):
        """Export all incidents as signed JSON."""
        data = [asdict(i) for i in self.incidents]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] {len(data)} incidents exported to {path}")

    def export_alerts(self, path: str = "alerts.json"):
        """Export all alerts."""
        with open(path, "w") as f:
            json.dump(self.alerts, f, indent=2)
        print(f"[✓] {len(self.alerts)} alerts exported to {path}")

    def verify_all_incidents(self) -> bool:
        """Verify signatures on all incidents (integrity check)."""
        for incident in self.incidents:
            saved_sig = incident.signature
            d = asdict(incident)
            d.pop("signature", None)
            d.pop("signature_algorithm", None)
            payload = canonical_incident_json(d)
            if not self._provider.verify(payload, saved_sig):
                print(f"[✗] Signature verification FAILED for {incident.incident_id}")
                return False
        print(f"[✓] All {len(self.incidents)} incident signatures verified")
        return True
