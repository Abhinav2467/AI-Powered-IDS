"""
Safety Decision Layer
=====================
Implements fail-operational logic:
- High confidence anomaly → Block + safe mode
- Low confidence → Alert only
- False positive guard → Never abrupt shutdown
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json


class ActionType(Enum):
    ALLOW = "ALLOW"
    ALERT = "ALERT"
    BLOCK = "BLOCK"
    SAFE_MODE = "SAFE_MODE"


@dataclass
class SafetyDecision:
    timestamp: str
    can_id: str
    ecu_name: str
    anomaly_score: float
    confidence: float
    action: ActionType
    reason: str
    is_safety_critical: bool = False


class SpeedContextLayer:
    """
    Layer-4 style vehicle-speed context: adjusts anomaly confidence threshold by driving regime.
    Parked 0–5 km/h → 0.75, urban 5–60 → 0.65, highway 60+ → 0.50.
    """

    def __init__(self, initial_speed: float = 0.0) -> None:
        self.current_speed_kmh = max(0.0, float(initial_speed))

    def update_speed(self, speed_kmh: float) -> None:
        self.current_speed_kmh = max(0.0, float(speed_kmh))

    @staticmethod
    def extract_speed_from_frame(message: dict) -> float:
        """Decode speed from CAN 0x160 payload bytes 0–1 (big-endian km/h), matching can_generator."""
        b0 = int(message.get("payload_byte_0", 0) or 0)
        b1 = int(message.get("payload_byte_1", 0) or 0)
        return float((b0 << 8) | b1)

    def get_adjusted_threshold(self) -> float:
        s = self.current_speed_kmh
        if s <= 5:
            return 0.75
        if s <= 60:
            return 0.65
        return 0.50

    def get_band_label(self) -> str:
        s = self.current_speed_kmh
        if s <= 5:
            return "parked"
        if s <= 60:
            return "urban"
        return "highway"


class SafetyDecisionLayer:
    """
    Safety-first decision engine.
    
    Thresholds:
        - confidence >= HIGH_THRESHOLD → Block + Safe Mode
        - confidence >= ALERT_THRESHOLD → Alert only
        - confidence < ALERT_THRESHOLD → Allow (normal)
    
    Safety-critical ECUs (brake, steering) get extra protection:
        - Never abrupt shutdown
        - Fallback to last known safe value
    """
    
    HIGH_THRESHOLD = 0.75       # Block + safe mode
    ALERT_THRESHOLD = 0.50      # Alert only
    
    SAFETY_CRITICAL_ECUS = {
        0x200: "Brake_Pedal",
        0x140: "Steering_Angle",
        0x1A0: "ABS_Status",
    }
    
    def __init__(
        self,
        zero_trust_enabled: bool = False,
        speed_context: SpeedContextLayer | None = None,
    ):
        self.decisions: list[SafetyDecision] = []
        self.zero_trust_enabled = zero_trust_enabled
        self.speed_context = speed_context if speed_context is not None else SpeedContextLayer()
        self.blocked_count = 0
        self.alert_count = 0
        self.safe_mode_count = 0
        self.allowed_count = 0
        self.last_safe_brake_value = 0x00  # Last known safe brake pressure
        
        # Explicit Identity Allowlist for Zero Trust Context
        self.TRUSTED_ECUS = {
            "Infotainment_System",
            "Gateway",
            "Brake_Pedal",
            "Steering_Angle",
            "Engine_Control",
            "Transmission",
            "ABS_Status",
            "Airbag_Control",
            "Door_Locks",
            "Wheel_Speed",
        }
    
    def decide(self, message: dict) -> SafetyDecision:
        """
        Make safety decision for a single CAN message.
        
        Args:
            message: Dict with keys: can_id, ecu_name, anomaly_score,
                     confidence, detected_anomaly, payload_byte_0
        """
        can_raw = message["can_id"]
        try:
            can_id = int(can_raw)
        except (TypeError, ValueError):
            s = str(can_raw).strip().lower()
            can_id = int(s, 16) if s.startswith("0x") else int(float(can_raw))
        if can_id == 0x160:
            sp = self.speed_context.extract_speed_from_frame(message)
            self.speed_context.update_speed(sp)

        confidence = message.get("confidence", 0)
        detected = message.get("detected_anomaly", 0)
        ecu_name = message.get("ecu_name", "Unknown")
        
        is_critical = can_id in self.SAFETY_CRITICAL_ECUS
        
        # ── Zero Trust Authentication Perimeter ──
        if self.zero_trust_enabled:
            is_spoofed = str(ecu_name).startswith("SPOOFED_")
            is_unknown = ecu_name not in self.TRUSTED_ECUS
            
            if is_spoofed or is_unknown:
                action = ActionType.BLOCK
                reason = f"ZTA Default Deny — Unauthorized Identity ({ecu_name})."
                self.blocked_count += 1
                
                decision = SafetyDecision(
                    timestamp=datetime.now().isoformat(),
                    can_id=hex(can_id),
                    ecu_name=ecu_name,
                    anomaly_score=round(message.get("anomaly_score", 0), 4),
                    confidence=1.0,  # Absolute confidence due to identity failure
                    action=action,
                    reason=reason,
                    is_safety_critical=is_critical,
                )
                self.decisions.append(decision)
                return decision
        
        # ── ML Decision Logic ──
        if detected == 0:
            # Normal traffic — allow and update safe values
            action = ActionType.ALLOW
            reason = "Normal traffic — passed"
            if can_id == 0x200:
                self.last_safe_brake_value = message.get("payload_byte_0", 0)
            self.allowed_count += 1
            
        elif confidence >= self.speed_context.get_adjusted_threshold():
            if is_critical:
                # SAFETY CRITICAL + HIGH CONFIDENCE
                # Block malicious message, activate safe mode
                # Fallback to last known safe brake value (NOT full shutdown)
                action = ActionType.SAFE_MODE
                reason = (
                    f"HIGH confidence anomaly on safety-critical ECU ({ecu_name}). "
                    f"Message blocked. Controlled braking activated — "
                    f"falling back to last safe value: {self.last_safe_brake_value}. "
                    f"No abrupt shutdown. "
                    f"Speed context: {self.speed_context.get_band_label()} "
                    f"({self.speed_context.current_speed_kmh:.0f} km/h) — "
                    f"threshold lowered to {self.speed_context.get_adjusted_threshold()}."
                )
                self.safe_mode_count += 1
            else:
                action = ActionType.BLOCK
                reason = f"HIGH confidence anomaly — message blocked"
                self.blocked_count += 1
                
        elif confidence >= self.ALERT_THRESHOLD:
            # Medium confidence — alert but don't block
            # FALSE POSITIVE GUARD: don't disrupt vehicle operation
            action = ActionType.ALERT
            reason = (
                f"Medium confidence ({confidence:.2f}). "
                f"Alert raised — message allowed to prevent false positive disruption."
            )
            self.alert_count += 1
            
        else:
            # Low confidence detection — likely false positive
            action = ActionType.ALLOW
            reason = "Low confidence detection — classified as likely false positive"
            self.allowed_count += 1
        
        decision = SafetyDecision(
            timestamp=datetime.now().isoformat(),
            can_id=hex(can_id),
            ecu_name=ecu_name,
            anomaly_score=round(message.get("anomaly_score", 0), 4),
            confidence=round(confidence, 4),
            action=action,
            reason=reason,
            is_safety_critical=is_critical,
        )
        
        self.decisions.append(decision)
        return decision
    
    def process_batch(self, results_df) -> list[SafetyDecision]:
        """Process all detection results through safety layer."""
        decisions = []
        for _, row in results_df.iterrows():
            d = self.decide(row.to_dict())
            decisions.append(d)
        return decisions
    
    def get_summary(self) -> dict:
        """Get summary of all safety decisions."""
        total = len(self.decisions)
        summary = {
            "total_messages": total,
            "allowed": self.allowed_count,
            "alerts": self.alert_count,
            "blocked": self.blocked_count,
            "safe_mode_activations": self.safe_mode_count,
            "safety_critical_events": sum(
                1 for d in self.decisions if d.is_safety_critical and d.action != ActionType.ALLOW
            ),
        }
        
        print("\n" + "=" * 50)
        print("  SAFETY DECISION LAYER — SUMMARY")
        print("=" * 50)
        print(f"  Total Messages:         {summary['total_messages']}")
        print(f"  Allowed:                {summary['allowed']}")
        print(f"  Alerts (low conf):      {summary['alerts']}")
        print(f"  Blocked:                {summary['blocked']}")
        print(f"  Safe Mode Activations:  {summary['safe_mode_activations']}")
        print(f"  Safety Critical Events: {summary['safety_critical_events']}")
        print("=" * 50)
        
        return summary
    
    def export_log(self, path: str = "safety_decisions.json"):
        """Export all decisions as JSON log."""
        log = [
            {
                "timestamp": d.timestamp,
                "can_id": d.can_id,
                "ecu_name": d.ecu_name,
                "anomaly_score": d.anomaly_score,
                "confidence": d.confidence,
                "action": d.action.value,
                "reason": d.reason,
                "is_safety_critical": d.is_safety_critical,
            }
            for d in self.decisions
        ]
        
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        
        print(f"[✓] Safety log exported to {path}")


if __name__ == "__main__":
    # Quick test
    layer = SafetyDecisionLayer()
    
    # Test high confidence brake attack
    test_msg = {
        "can_id": 0x200,
        "ecu_name": "SPOOFED_Brake",
        "anomaly_score": -0.35,
        "confidence": 0.92,
        "detected_anomaly": 1,
        "payload_byte_0": 0xFF,
    }
    
    d = layer.decide(test_msg)
    print(f"Action: {d.action.value}")
    print(f"Reason: {d.reason}")
