"""
Shared plain-language summary of top mitigated threat patterns (CustomTkinter + Streamlit).
"""

from __future__ import annotations

from mitigation import MitigationSystem

TOP_PREVENTED_THREATS_COUNT = 6

_ACTION_EXPOSURE_WEIGHT = {"SAFE_MODE": 3.0, "BLOCK": 2.0, "ALERT": 1.0}


def attack_type_to_human(attack_type: str) -> str:
    mapping = {
        "CAN_INJECTION_HIGH_CONFIDENCE": "a high-confidence injection or spoofing attempt",
        "CAN_ANOMALY_MEDIUM_CONFIDENCE": "a medium-confidence anomaly",
        "CAN_ANOMALY_LOW_CONFIDENCE": "a low-confidence anomaly",
    }
    return mapping.get(attack_type, attack_type.replace("_", " ").lower())


def format_prevented_threats_summary(
    mitigation: MitigationSystem, top_n: int = TOP_PREVENTED_THREATS_COUNT
) -> str:
    """Short plain-language summary: top threat patterns by frequency and risk avoided."""
    incidents = mitigation.incidents
    if not incidents:
        return (
            "\nTop prevented threats (by frequency × severity)\n"
            "  None — no traffic required blocking, alerts, or safe mode in this run.\n"
        )

    groups: dict[tuple[str, str, str, str], list] = {}
    for inc in incidents:
        action = (inc.action_taken or "").strip().upper()
        key = (inc.ecu_name, str(inc.can_id), inc.attack_type, action)
        groups.setdefault(key, []).append(inc)

    scored: list[tuple[float, tuple[str, str, str, str], list]] = []
    for key, grp in groups.items():
        _ecu, _cid, _atk, action = key
        n_ev = len(grp)
        mean_conf = sum(i.confidence for i in grp) / n_ev
        mean_anom = sum(abs(i.anomaly_score) for i in grp) / n_ev
        critical = any(i.is_safety_critical for i in grp)
        aw = _ACTION_EXPOSURE_WEIGHT.get(action, 1.0)
        crit_m = 1.45 if critical else 1.0
        exposure = (
            n_ev
            * mean_conf
            * (1.0 + min(2.0, mean_anom))
            * aw
            * crit_m
        )
        scored.append((exposure, key, grp))

    scored.sort(key=lambda x: x[0], reverse=True)
    show = scored[: min(top_n, len(scored))]

    lines: list[str] = [
        "\nTop prevented threats (by frequency × confidence × action severity)",
        f"  {len(incidents)} mitigated events, {len(scored)} distinct patterns — "
        f"showing top {len(show)}.",
    ]
    for _exposure, key, grp in show:
        ecu_name, can_id, attack_type, action = key
        n_ev = len(grp)
        mean_conf = sum(i.confidence for i in grp) / n_ev
        mean_anom = sum(abs(i.anomaly_score) for i in grp) / n_ev
        critical = any(i.is_safety_critical for i in grp)
        human_at = attack_type_to_human(attack_type)
        crit = " (safety-critical ECU)" if critical else ""
        conf_pct = f"{mean_conf * 100:.0f}%"

        if action == "BLOCK":
            verb = f"Blocked {human_at}"
        elif action == "SAFE_MODE":
            verb = f"Safe mode after {human_at}"
        elif action == "ALERT":
            verb = f"Alert for {human_at} (traffic still allowed)"
        else:
            verb = f"{action}: {human_at}"

        lines.append(
            f"  • {n_ev}× {verb} — {ecu_name}{crit}, CAN {can_id}. "
            f"Avg. confidence {conf_pct}, avg. |anomaly score| {mean_anom:.2f}."
        )

    if len(scored) > len(show):
        lines.append(
            f"  … {len(scored) - len(show)} other pattern(s); full audit in Incidents tab."
        )

    return "\n".join(lines) + "\n"
