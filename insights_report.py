"""
Offline insight engine for CAN-Guard edge detection (no APIs, no cloud).

Combines:
  - Rule-based narrative tied to the Isolation Forest + StandardScaler pipeline
  - Lightweight statistics on decision_function scores (separation, overlap proxy)
  - Operational risk band from metrics (FPR / detection rate / F1)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from detection_engine import FEATURE_COLS


def _score_separation_label(normal_scores: pd.Series, mal_scores: pd.Series) -> tuple[str, float]:
    """Offline proxy for class separation using IF decision scores (more negative = more anomalous)."""
    if len(normal_scores) == 0 or len(mal_scores) == 0:
        return "insufficient data", 0.0
    mn = float(normal_scores.mean())
    mm = float(mal_scores.mean())
    gap = abs(mn - mm)
    # Heuristic: larger gap => easier separation in score space
    if gap >= 0.08:
        return "strong", gap
    if gap >= 0.04:
        return "moderate", gap
    return "weak (overlapping distributions likely)", gap


def _overlap_proxy(normal_scores: pd.Series, mal_scores: pd.Series) -> str:
    """Simple overlap description using quantiles (offline, no ML API)."""
    if len(normal_scores) < 5 or len(mal_scores) < 3:
        return "n/a (too few samples)"
    n75 = float(normal_scores.quantile(0.75))
    m25 = float(mal_scores.quantile(0.25))
    if m25 < n75:
        return "high overlap in score range (attacks and normal share similar IF scores)"
    return "moderate separation between typical normal and typical attack scores"


def _risk_band(metrics: dict[str, Any]) -> str:
    fpr = float(metrics.get("false_positive_rate", 0))
    dr = float(metrics.get("detection_rate", 0))
    f1 = float(metrics.get("f1_score", 0))
    if fpr <= 0.05 and dr >= 0.85:
        return "GREEN — low nuisance alarms, strong catch rate"
    if fpr <= 0.10 and dr >= 0.70:
        return "AMBER — acceptable for demo; tune contamination / features if deploying"
    return "RED — revisit training data, contamination, or safety thresholds"


def _offline_recommendations(metrics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    fpr = float(metrics.get("false_positive_rate", 0))
    fn = int(metrics.get("false_negatives", 0))
    fp = int(metrics.get("false_positives", 0))
    if fpr > 0.08:
        out.append("High FPR: consider validation-driven contamination tuning or richer features (e.g. IAT rolling stats).")
    if fn > 0:
        out.append(f"Missed attacks (FN={fn}): check for score overlap; attacks may mimic normal brake timing.")
    if fp > 0 and fpr <= 0.05:
        out.append("Low FPR but some FP: review safety layer thresholds on medium-confidence alerts.")
    if not out:
        out.append("Metrics look balanced for the synthetic scenario; validate on real CAN captures next.")
    return out


def build_explanation_report(
    *,
    metrics: dict[str, Any],
    results_df: pd.DataFrame,
    safety_summary: dict[str, Any],
    mit_summary: dict[str, Any],
    mitigation: Any,
    decisions: list[Any],
) -> str:
    """
    Full offline report: edge detector semantics, chart meanings, incidents, heuristics.
    """
    m = metrics
    n_normal = int((results_df["is_malicious"] == 0).sum())
    n_mal = int((results_df["is_malicious"] == 1).sum())
    n_total = len(results_df)

    normal_scores = results_df.loc[results_df["is_malicious"] == 0, "anomaly_score"]
    mal_scores = results_df.loc[results_df["is_malicious"] == 1, "anomaly_score"]
    mn_n = float(normal_scores.mean()) if len(normal_scores) else float("nan")
    mn_m = float(mal_scores.mean()) if len(mal_scores) else float("nan")
    sep_label, sep_gap = _score_separation_label(normal_scores, mal_scores)
    overlap = _overlap_proxy(normal_scores, mal_scores)
    band = _risk_band(m)
    recs = _offline_recommendations(m)

    action_counts: Counter[str] = Counter()
    for d in decisions:
        action_counts[d.action.value] += 1

    inc_by_attack: Counter[str] = Counter()
    inc_by_action: Counter[str] = Counter()
    critical_n = 0
    for inc in mitigation.incidents:
        inc_by_attack[inc.attack_type] += 1
        inc_by_action[inc.action_taken] += 1
        if getattr(inc, "is_safety_critical", False):
            critical_n += 1

    lines: list[str] = []
    lines.append("# CAN-Guard AI — Offline edge insight report\n")
    lines.append("## Edge detector (on-device, no API)\n")
    lines.append(
        f"- **Model:** `StandardScaler` → **Isolation Forest** on **{len(FEATURE_COLS)}** numeric features "
        f"(same path as training and optional ONNX export).\n"
        f"- **Features:** `{', '.join(FEATURE_COLS[:4])}`, … `iat_rolling_mean`, `iat_rolling_std` "
        f"(rolling inter-arrival stats help the edge separate bursty legit traffic from injected frames).\n"
        f"- **Score:** `decision_function` output per frame — typically **more negative** ⇒ more anomalous "
        f"for this sklearn IF setup.\n"
        f"- **Offline separation check:** **{sep_label}** (|mean_normal − mean_malicious| ≈ **{sep_gap:.4f}**).\n"
        f"- **Overlap proxy:** {overlap}\n"
        f"- **Operational band (heuristic from metrics):** {band}\n"
    )

    lines.append("\n## Executive summary\n")
    lines.append(
        f"- **Frames:** {n_total} ({n_normal} normal, {n_mal} malicious).\n"
        f"- **Accuracy / F1:** {m.get('accuracy', 0):.1%} / {m.get('f1_score', 0):.1%}.\n"
        f"- **Detection rate (recall on attacks):** {m.get('detection_rate', 0):.1%}; "
        f"**FPR on normal:** {m.get('false_positive_rate', 0):.1%}.\n"
        f"- **Mean IF score:** normal **{mn_n:.4f}** vs malicious **{mn_m:.4f}**.\n"
    )

    lines.append("\n## What each chart shows (edge view)\n")
    lines.append(
        "**1–2. Histograms** — Distribution of IF scores: less overlap ⇒ easier edge decision.\n\n"
        "**3. Timeline** — Score vs message index; malicious points should skew anomalous when injected.\n\n"
        "**4. CAN IDs** — Volume by ID; brake spoofing uses **0x200** in this simulator.\n\n"
        "**5. Safety pie** — Policy outcomes after scoring: ALLOW / ALERT / BLOCK / SAFE_MODE.\n\n"
        f"**6. Confusion matrix** — TN={m.get('true_negatives')}, FP={m.get('false_positives')}, "
        f"FN={m.get('false_negatives')}, TP={m.get('true_positives')}.\n"
    )

    lines.append("\n## Latency (gateway + edge inference)\n")
    p50 = m.get("p50_detection_latency_us")
    p99 = m.get("p99_detection_latency_us")
    ae = m.get("avg_edge_processing_latency_us")
    lines.append(
        f"- **Path:** {m.get('avg_detection_latency_us', 0):.1f} μs avg (p50 {p50} / p99 {p99} μs).\n"
        f"- **IF amortized edge time:** ~{ae} μs per frame in batch inference.\n"
    )

    lines.append("\n## Safety & incidents\n")
    lines.append(
        f"- **Counts:** allowed {safety_summary.get('allowed', 0)}, alerts {safety_summary.get('alerts', 0)}, "
        f"blocked {safety_summary.get('blocked', 0)}, safe_mode {safety_summary.get('safe_mode_activations', 0)}.\n"
        f"- **Decisions:** {dict(action_counts)}\n"
        f"- **Incidents:** {mit_summary.get('total_incidents', 0)} signed ({mit_summary.get('signing_algorithm', 'n/a')}); "
        f"critical rows ≈ {critical_n}.\n"
        f"- **By action:** {dict(inc_by_action)}\n"
        f"- **By attack label:** {dict(inc_by_attack)}\n"
        f"- **Blocked IDs:** {mit_summary.get('blocked_can_ids', [])}\n"
    )

    lines.append("\n## Offline recommendations (rule + statistics)\n")
    for r in recs:
        lines.append(f"- {r}\n")

    lines.append("\n## Demo script (30 seconds)\n")
    lines.append(
        "1. Point to **ONNX** (`models/edge_model.onnx`) for non-Python edge runtimes.  \n"
        "2. Explain **IF scores** + **rolling IAT** as the on-gateway signal.  \n"
        "3. Walk **confusion matrix** → **incidents** → **signed logs**.\n"
    )

    return "".join(lines)


# Backward compatibility: no network; callers may still import this name
def maybe_enhance_with_llm(base_report: str, metrics: dict[str, Any]) -> str:
    """Deprecated: insights are fully offline. Returns the report unchanged."""
    return base_report
