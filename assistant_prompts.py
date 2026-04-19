"""
Quick Assistant prompts — shared by CustomTkinter (sends to Ollama) and Streamlit (copy/reference).

User-controlled or telemetry-derived text sent to the local LLM is passed through
`sanitize_can_data_for_prompt` in `insight_engine` to reduce prompt-injection risk from CAN-derived
or log content (the IDS decision path does not depend on the LLM).
"""

from __future__ import annotations

import re

CHAT_SUGGESTIONS: tuple[tuple[str, str], ...] = (
    (
        "Feasibility on a real ECU?",
        "Given our detection accuracy, precision/recall, and average inference latency in this run, "
        "how feasible is deploying this Isolation Forest on a production gateway ECU? "
        "What is the tightest bottleneck: CPU, memory, or false-positive risk?",
    ),
    (
        "Charts 1–6 for judges",
        "Walk me through each of the six dashboard charts in order: what it shows, why it matters, "
        "and what a judge should look for in 2–3 sentences per chart.",
    ),
    (
        "Top threats mitigated",
        "Summarize the most serious mitigated threats from this run using the incident index: "
        "which ECUs and CAN IDs were involved, and what actions (alert, block, safe mode) were taken?",
    ),
    (
        "Complexity: frame → decision",
        "Explain the conceptual complexity of the pipeline from one CAN frame’s features to a safety "
        "decision and mitigation step. Where is most of the ‘thinking’ vs simple thresholds?",
    ),
    (
        "False positives vs safety",
        "What trade-off exists between false positives and protecting safety-critical ECUs? "
        "How does our fail-operational design try to balance that?",
    ),
    (
        "Latency thought experiment",
        "If gateway stack + inference latency doubled, which parts of the story (detection, mitigation, "
        "UX) would break first, and what would you monitor in production?",
    ),
    (
        "Zero-trust: worth it?",
        "Based on this run’s context, when does zero-trust style ECU identity checking add real value, "
        "and when might it be noisy for a demo?",
    ),
    (
        "30-second pitch",
        "Give me a punchy 30-second pitch for judges: problem, what we built, one metric, and why it matters "
        "for automotive security—bullet points only.",
    ),
)


def sanitize_can_data_for_prompt(raw: str, *, max_chars: int = 120_000) -> str:
    """
    Defensive normalization before sending telemetry/user text to Ollama (explanation-only path).

    - Drops common instruction-injection phrases and angle-bracket blobs.
    - Redacts long hex runs (possible dumps); short tokens like 0x200 are unchanged.
    - Caps total length to avoid runaway prompts (default generous for dashboard context).

    For a one-line incident summary only, pass a smaller ``max_chars`` (e.g. 500).
    """
    if not raw:
        return ""
    s = re.sub(
        r"(ignore|forget|override|pretend|system:|<<.*?>>)",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"\b[0-9a-fA-F]{32,}\b", "[HEX_REDACTED]", s)
    s = re.sub(r"\b[0-9a-fA-F]{16,31}\b", "[HEX_REDACTED]", s)
    s = s.strip()[:max_chars]
    return s


def build_safe_incident_prompt_line(
    *,
    ecu_name: str,
    action_taken: str,
    is_safety_critical: bool,
    confidence: float,
) -> str:
    """
    Structured, minimal line for narratives — no signatures, hashes, or raw feature blobs.
    """
    severity = "CRITICAL" if is_safety_critical else "HIGH"
    conf_band = "high" if confidence > 0.75 else "medium"
    ecu = ecu_name or "unknown"
    action = action_taken or "unknown"
    return (
        f"A {severity} security event occurred on {ecu}. "
        f"Action taken: {action}. Confidence band: {conf_band}. "
        "Summarise the risk in 2 sentences."
    )
