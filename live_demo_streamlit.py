"""
CAN-Guard AI — Live Attack Demo (Streamlit Edition)
====================================================
TRUE frame-by-frame real-time simulation for judge presentations.
Every CAN frame streams in live with a configurable delay so judges
can watch attacks arrive, get scored, and be BLOCKED in real time.

Run with:
    .venv/bin/streamlit run live_demo_streamlit.py --server.port 8502
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from can_generator import (
    generate_normal_traffic,
    inject_attack_traffic,
    inject_high_speed_brake_attack,
)
from detection_engine import EdgeAIDetector
from gateway_simulator import apply_gateway_path_delay
from safety_layer import SafetyDecisionLayer, SpeedContextLayer
from mitigation import MitigationSystem
import signing as _signing
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAN-Guard AI · Live Attack Demo",
    page_icon="🛡️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hdr {
    background: linear-gradient(135deg,#0d1117,#161b22);
    border:1px solid #30363d; border-radius:14px;
    padding:1.8rem 2rem; margin-bottom:1.5rem; text-align:center;
}
.hdr h1 {
    font-size:2.2rem; font-weight:700; margin:0;
    background:linear-gradient(90deg,#58a6ff,#3fb950,#f78166);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hdr .sub { color:#8b949e; font-size:.9rem; margin-top:.3rem; }

/* live counter cards */
.card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:.9rem 1rem; text-align:center; height:100%;
}
.card .lbl { color:#8b949e; font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; }
.card .val { font-size:1.9rem; font-weight:700; margin-top:.25rem; line-height:1; }
.val.blue   { color:#58a6ff; }
.val.green  { color:#3fb950; }
.val.yellow { color:#d29922; }
.val.red    { color:#f85149; }
.val.white  { color:#c9d1d9; }

/* frame rows */
.frame-row {
    font-family:'JetBrains Mono',monospace; font-size:.75rem;
    padding:.35rem .7rem; border-radius:6px; margin-bottom:3px;
    display:flex; gap:.8rem; align-items:center; flex-wrap:wrap;
}
.frame-normal { background:#0d1f14; border-left:3px solid #3fb950; color:#8b949e; }
.frame-attack { background:#1c0e0e; border-left:3px solid #f85149; color:#c9d1d9; }
.frame-safemode { background:#200e0e; border-left:4px solid #ff7b72; color:#c9d1d9; }
.frame-alert  { background:#1c1608; border-left:3px solid #d29922; color:#c9d1d9; }

.badge {
    border-radius:999px; padding:1px 8px; font-size:.68rem; font-weight:700;
    white-space:nowrap;
}
.b-atk  { background:#3d1f1e; color:#f85149; border:1px solid #5a1010; }
.b-norm { background:#1a2f1f; color:#3fb950; border:1px solid #0f3d20; }
.b-blk  { background:#3d1f1e; color:#ff7b72; border:1px solid #5a1010; }
.b-allow{ background:#1a2f1f; color:#3fb950; border:1px solid #0f3d20; }
.b-alert{ background:#2d2309; color:#d29922; border:1px solid #3d3000; }
.b-safe { background:#3d1f1e; color:#ff7b72; border:1px solid #5a1010; }

.reason { color:#6e7681; font-size:.68rem; font-style:italic; }

/* sig ok / fail */
.sig-ok   { background:#1a2f1f; border:1px solid #0f3d20; color:#3fb950;
            border-radius:8px; padding:.5rem 1rem; font-weight:600; }
.sig-fail { background:#3d1f1e; border:1px solid #5a1010; color:#f85149;
            border-radius:8px; padding:.5rem 1rem; font-weight:600; }
.sig-none { background:#161b22; border:1px solid #30363d; color:#8b949e;
            border-radius:8px; padding:.5rem 1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Scenarios definition ──────────────────────────────────────────────────────
SCENARIOS: dict[str, dict] = {
    "🔴  Brake Injection @ 120 km/h": {
        "attack_type":   "high_speed_brake_injection",
        "default_speed": 120,
        "normal_count":  400,
        "attack_count":  50,
        "desc": "Spoofed 0xFF emergency-brake frames on CAN ID 0x200 at highway speed. "
                "Speed-aware threshold drops to 0.50 — system is most aggressive.",
    },
    "🟠  Fuzzing Attack": {
        "attack_type":   "fuzzing",
        "default_speed": 0,
        "normal_count":  350,
        "attack_count":  50,
        "desc": "Random CAN IDs and random payloads flooding the bus to confuse the ECU.",
    },
    "🟡  Replay Attack": {
        "attack_type":   "replay",
        "default_speed": 0,
        "normal_count":  350,
        "attack_count":  40,
        "desc": "Rapid brake-disable frames replayed on 0x200.",
    },
    "🟣  Injection Attack": {
        "attack_type":   "injection",
        "default_speed": 0,
        "normal_count":  350,
        "attack_count":  40,
        "desc": "Standard spoofed brake injection frames.",
    },
}

ACTION_BADGE   = {"ALLOW":"b-allow", "ALERT":"b-alert", "BLOCK":"b-blk", "SAFE_MODE":"b-safe"}
ACTION_EMOJI   = {"ALLOW":"✅ ALLOW", "ALERT":"⚠️ ALERT", "BLOCK":"🚫 BLOCK", "SAFE_MODE":"🔴 SAFE MODE"}
ROW_CLASS      = {"ALLOW":"frame-normal", "ALERT":"frame-alert", "BLOCK":"frame-attack", "SAFE_MODE":"frame-safemode"}

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Edge AI detection model…")
def get_detector() -> EdgeAIDetector:
    det = EdgeAIDetector(contamination=0.05)
    p = config.PRETRAINED_MODEL_PATH
    if p.exists():
        det.load_model(str(p))
    else:
        df = generate_normal_traffic(1000, seed=42)
        det.train(df, verbose=False)
        det.save_model("edge_model.joblib")
    return det

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <h1>🛡️ CAN-Guard AI · Live Attack Demo</h1>
  <div class="sub">Real-time frame-by-frame CAN bus attack simulation · MAHE Mobility Challenge 2026</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    scenario_name = st.selectbox("Attack Scenario", list(SCENARIOS.keys()))
    cfg = SCENARIOS[scenario_name]
    st.caption(cfg["desc"])
    st.divider()

    speed_kmh    = st.slider("Vehicle Speed (km/h)", 0, 200, cfg["default_speed"], 10)
    normal_count = st.slider("Normal Frames",  100, 800, cfg["normal_count"],  50)
    attack_count = st.slider("Attack Frames",   10, 100, cfg["attack_count"],   5)

    st.divider()
    st.markdown("**⏱ Replay Speed**")
    normal_delay_ms = st.slider("Normal frame delay (ms)", 0, 200, 30, 5)
    attack_delay_ms = st.slider("Attack frame delay (ms)", 0, 500, 120, 10,
                                 help="Slower = judges can read the BLOCK decision")
    show_all_normal = st.checkbox("Show all normal frames", value=False,
                                   help="Uncheck to only show every 10th normal frame (less noise)")

    st.divider()
    run_btn = st.button("▶ START LIVE SIMULATION", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
**Legend**
- 🚫 BLOCK — packet dropped, CAN ID blacklisted  
- 🔴 SAFE MODE — safety-critical ECU, fallback activated  
- ⚠️ ALERT — medium confidence, logged & allowed  
- ✅ ALLOW — normal traffic  
""")

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("running", False), ("frames", []),
    ("allowed", 0), ("alerts", 0), ("blocked", 0), ("safe_mode", 0),
    ("incidents", 0), ("sig_ok", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper: render one frame as HTML ─────────────────────────────────────────
def frame_html(f: dict) -> str:
    truth_cls   = "b-atk"  if f["is_attack"] else "b-norm"
    truth_txt   = "ATTACK" if f["is_attack"] else "NORMAL"
    action_cls  = ACTION_BADGE[f["action"]]
    action_txt  = ACTION_EMOJI[f["action"]]
    row_cls     = ROW_CLASS[f["action"]]
    score_col   = "#f85149" if f["anomaly_score"] < 0 else "#3fb950"
    reason_span = f'<span class="reason">↳ {f["reason"][:95]}</span>' if f["reason"] else ""
    return (
        f'<div class="{row_cls} frame-row">'
        f'<span style="color:#6e7681">#{f["idx"]:04d}</span>'
        f'<span style="color:#79c0ff">{f["can_id"]}</span>'
        f'<span style="color:#c9d1d9;min-width:140px">{f["ecu"]}</span>'
        f'<span style="color:#6e7681">{f["iat_ms"]:.2f}ms</span>'
        f'<span style="color:#484f58">[{f["payload"]}]</span>'
        f'<span class="badge {truth_cls}">{truth_txt}</span>'
        f'<span class="badge {action_cls}">{action_txt}</span>'
        f'<span style="color:#8b949e">conf {f["confidence"]:.2f}</span>'
        f'<span style="color:{score_col}">{f["anomaly_score"]:+.4f}</span>'
        f'{reason_span}'
        f'</div>'
    )

# ── Helper: live metric card ──────────────────────────────────────────────────
def card(label, value, color="white"):
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val {color}">{value}</div></div>')

# ── Main layout placeholders ──────────────────────────────────────────────────
metric_ph  = st.empty()          # top 6 metric cards
prog_ph    = st.empty()          # progress bar
status_ph  = st.empty()          # "▶ Running…" / "✅ Done" status text
feed_ph    = st.empty()          # live frame feed box
chart_ph   = st.empty()          # anomaly score chart (updates at end)
sig_ph     = st.empty()          # signature verification result

# ── Idle splash ───────────────────────────────────────────────────────────────
if not st.session_state.running and not st.session_state.frames:
    feed_ph.markdown("""
    <div style="text-align:center;padding:3.5rem 2rem;background:#0d1117;
         border:1px solid #30363d;border-radius:14px">
        <div style="font-size:2.5rem;margin-bottom:.8rem">📡</div>
        <h2 style="color:#c9d1d9;margin:0 0 .5rem">Waiting for simulation…</h2>
        <p style="color:#8b949e;margin:0">Select a scenario in the sidebar and click
        <strong>▶ START LIVE SIMULATION</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

# ── LIVE SIMULATION ───────────────────────────────────────────────────────────
if run_btn:
    # Reset state
    st.session_state.update(
        running=True, frames=[],
        allowed=0, alerts=0, blocked=0, safe_mode=0,
        incidents=0, sig_ok=None,
    )

    det = get_detector()

    # ── 1. Generate & detect (fast batch) ────────────────────────────────────
    with st.spinner("Generating CAN traffic and running AI detection batch…"):
        np.random.seed(42)
        normal_df = generate_normal_traffic(normal_count, seed=42)
        atype = cfg["attack_type"]
        if atype == "high_speed_brake_injection":
            test_df = inject_high_speed_brake_attack(normal_df, attack_count, speed_kmh=speed_kmh)
        else:
            test_df = inject_attack_traffic(normal_df, attack_count, atype)
        test_df = apply_gateway_path_delay(test_df, mean_us=48.0, jitter_us=12.0, seed=202)
        results = det.predict(test_df)

    total_frames = len(results)

    # ── 2. Frame-by-frame live replay ───────────────────────────────────────
    speed_ctx   = SpeedContextLayer(initial_speed=float(speed_kmh))
    safety      = SafetyDecisionLayer(zero_trust_enabled=True, speed_context=speed_ctx)
    mitigation  = MitigationSystem()

    visible_html: list[str] = []
    normal_shown = 0
    score_data: list[dict] = []

    status_ph.markdown(
        '<p style="color:#3fb950;font-weight:600;font-size:.95rem">▶ Simulation running…</p>',
        unsafe_allow_html=True,
    )

    for step, (idx, row) in enumerate(results.iterrows(), 1):
        row_dict = row.to_dict()
        decision = safety.decide(row_dict)
        is_mal   = int(row_dict.get("is_malicious", 0))
        action   = decision.action.value
        payload  = [row_dict.get(f"payload_byte_{i}", 0) for i in range(8)]

        f = {
            "idx":           idx,
            "can_id":        row_dict.get("can_id_hex", hex(int(row_dict.get("can_id", 0)))),
            "ecu":           row_dict.get("ecu_name", "?"),
            "iat_ms":        round(float(row_dict.get("inter_arrival_time", 0)) * 1000, 3),
            "payload":       " ".join(f"{int(b):02X}" for b in payload),
            "is_attack":     is_mal,
            "action":        action,
            "confidence":    round(float(row_dict.get("confidence", 0)), 3),
            "anomaly_score": round(float(row_dict.get("anomaly_score", 0)), 4),
            "reason":        decision.reason[:95] if action in ("BLOCK", "SAFE_MODE") else "",
        }
        st.session_state.frames.append(f)
        score_data.append({"idx": idx, "score": f["anomaly_score"], "is_attack": is_mal, "action": action})

        # Update mitigation
        if action in ("BLOCK", "SAFE_MODE", "ALERT"):
            mitigation.process_safety_decisions([decision])

        # Update counters
        if action == "ALLOW":     st.session_state.allowed   += 1
        elif action == "ALERT":   st.session_state.alerts    += 1
        elif action == "BLOCK":   st.session_state.blocked   += 1
        elif action == "SAFE_MODE": st.session_state.safe_mode += 1
        st.session_state.incidents = len(mitigation.incidents)

        # ── Decide whether to display this frame ──────────────────────────
        show_frame = True
        delay = normal_delay_ms / 1000.0
        if not is_mal:
            normal_shown += 1
            if not show_all_normal and normal_shown > 8 and normal_shown % 10 != 0:
                show_frame = False
        else:
            delay = attack_delay_ms / 1000.0

        if show_frame:
            visible_html.append(frame_html(f))
            # Keep only latest 80 rows in feed (performance)
            display_rows = visible_html[-80:]

        # ── Update metric cards ───────────────────────────────────────────
        metric_ph.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.7rem;margin-bottom:.5rem">
            {card("📡 Total", step, "blue")}
            {card("✅ Allowed", st.session_state.allowed, "green")}
            {card("⚠️ Alerts", st.session_state.alerts, "yellow")}
            {card("🚫 Blocked", st.session_state.blocked, "red")}
            {card("🔴 Safe Mode", st.session_state.safe_mode, "red")}
            {card("📋 Incidents", st.session_state.incidents, "blue")}
        </div>
        """, unsafe_allow_html=True)

        # ── Progress bar ──────────────────────────────────────────────────
        pct = int(step / total_frames * 100)
        bar_color = "#f85149" if is_mal else "#3fb950"
        prog_ph.markdown(
            f'<div style="background:#21262d;border-radius:999px;height:6px;margin-bottom:.5rem">'
            f'<div style="background:{bar_color};width:{pct}%;height:100%;border-radius:999px;'
            f'transition:width .1s"></div></div>',
            unsafe_allow_html=True,
        )

        # ── Live frame feed ───────────────────────────────────────────────
        if show_frame:
            feed_ph.markdown(
                f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;'
                f'padding:.7rem;max-height:460px;overflow-y:auto">'
                f'{"".join(display_rows)}</div>',
                unsafe_allow_html=True,
            )

        time.sleep(delay)

    # ── 3. Final summary + charts ────────────────────────────────────────────
    ss  = safety.get_summary()
    ms  = mitigation.get_summary()

    status_ph.markdown(
        f'<p style="color:#58a6ff;font-weight:600;font-size:.95rem">'
        f'✅ Simulation complete — {total_frames} frames processed in {scenario_name}</p>',
        unsafe_allow_html=True,
    )
    prog_ph.markdown(
        '<div style="background:#3fb950;height:6px;border-radius:999px;width:100%"></div>',
        unsafe_allow_html=True,
    )

    # Final metric row (confirmed totals)
    metric_ph.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.7rem;margin-bottom:.5rem">
        {card("📡 Total", ss['total_messages'], "blue")}
        {card("✅ Allowed", ss['allowed'], "green")}
        {card("⚠️ Alerts", ss['alerts'], "yellow")}
        {card("🚫 Blocked", ss['blocked'], "red")}
        {card("🔴 Safe Mode", ss['safe_mode_activations'], "red")}
        {card("📋 Incidents", ms['total_incidents'], "blue")}
    </div>
    """, unsafe_allow_html=True)

    # Signature verification
    sig_all_ok = True
    n_signed   = len(mitigation.incidents)
    for inc in mitigation.incidents:
        if inc.signature:
            d = {k: v for k, v in vars(inc).items() if k not in ("signature", "signature_algorithm")}
            if not mitigation.verify_signature(_signing.canonical_incident_json(d), inc.signature):
                sig_all_ok = False

    if n_signed == 0:
        sig_ph.markdown('<div class="sig-none">ℹ️ No incidents to sign in this run.</div>', unsafe_allow_html=True)
    elif sig_all_ok:
        sig_ph.markdown(f'<div class="sig-ok">🔐 All {n_signed} incident signatures verified — HMAC-SHA3-256 ✓</div>', unsafe_allow_html=True)
    else:
        sig_ph.markdown(f'<div class="sig-fail">❌ Signature verification FAILED on one or more incidents.</div>', unsafe_allow_html=True)

    # Anomaly score chart
    df_s = pd.DataFrame(score_data)
    fig  = go.Figure()
    if not df_s.empty:
        norm_df = df_s[df_s["is_attack"] == 0]
        att_df  = df_s[df_s["is_attack"] == 1]
        blk_df  = df_s[df_s["action"].isin(["BLOCK", "SAFE_MODE"])]

        fig.add_trace(go.Scatter(x=norm_df["idx"], y=norm_df["score"], mode="markers",
                                  name="Normal", marker=dict(color="#3fb950", size=4, opacity=0.5)))
        fig.add_trace(go.Scatter(x=att_df["idx"], y=att_df["score"], mode="markers",
                                  name="Attack", marker=dict(color="#f85149", size=7, symbol="x")))
        fig.add_trace(go.Scatter(x=blk_df["idx"], y=blk_df["score"], mode="markers",
                                  name="Blocked", marker=dict(color="#ff7b72", size=11,
                                  symbol="x-open", line=dict(width=2))))
        fig.add_hline(y=0, line_dash="dot", line_color="#30363d",
                      annotation_text="Detection boundary", annotation_position="bottom right")

    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter"),
        xaxis=dict(title="Frame index", gridcolor="#21262d"),
        yaxis=dict(title="IF anomaly score", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0, r=0, t=30, b=0), height=300,
        title=dict(text="📈 Anomaly Score Timeline", font=dict(color="#c9d1d9")),
    )
    chart_ph.plotly_chart(fig, use_container_width=True)

    # CSV export
    df_export = pd.DataFrame(st.session_state.frames)
    st.download_button(
        "⬇️ Download Full Frame Log (CSV)",
        df_export.to_csv(index=False).encode("utf-8"),
        file_name="can_guard_live_demo.csv",
        mime="text/csv",
    )

    st.session_state.running = False
