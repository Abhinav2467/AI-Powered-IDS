"""
CAN-Guard AI — Demo Dashboard (Streamlit)
==========================================
Visual demo output with Before/After comparison.
Run: streamlit run dashboard.py
   or: python3 dashboard.py   (re-invokes Streamlit automatically)
"""

import sys
from pathlib import Path

# Running `python3 dashboard.py` executes outside Streamlit and breaks session state + spams
# ScriptRunContext warnings. If we are not inside Streamlit, hand off to `streamlit run`.
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _cg_sl_ctx = get_script_run_ctx()
    except Exception:
        _cg_sl_ctx = None
    if _cg_sl_ctx is None:
        import subprocess

        r = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        )
        raise SystemExit(r.returncode)

import html
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config
from can_generator import generate_dataset
from detection_engine import EdgeAIDetector
from gateway_simulator import apply_gateway_path_delay
from mitigation import MitigationSystem
from safety_layer import SafetyDecisionLayer, SpeedContextLayer
from threat_path import ThreatPathLogger
from insights_report import build_explanation_report
from prevented_threats_summary import format_prevented_threats_summary
from assistant_prompts import CHAT_SUGGESTIONS
from insight_engine import LLMInsightEngine, build_distilled_dashboard_context

# ── Page config — paired with CustomTkinter desktop (no decorative icons) ──
st.set_page_config(
    page_title="CAN-Guard — vehicle network shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark-blue / slate, close to CustomTkinter dark theme) ──
st.markdown(
    """
<style>
    .block-container { padding-top: 0.75rem; max-width: 100%; }
    .cg-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #e6edf3;
        margin: 0 0 2px 0;
    }
    .cg-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.45;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #141414 100%);
        border-right: 1px solid #2d2d2d;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
    .stMetric > div {
        background: linear-gradient(145deg, #1c2128 0%, #161b22 100%);
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid #30363d;
    }
    .cg-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .cg-panel h4 { margin: 0 0 10px 0; font-size: 0.95rem; color: #c9d1d9; font-weight: 600; }
    .cg-mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #e6edf3;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px;
        white-space: pre-wrap;
    }
    [data-testid="stTabs"] button { font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header (text-only — matches desktop sidebar branding) ──
st.markdown(
    '<p class="cg-title">CAN-Guard</p><p class="cg-subtitle">AI Based In-House IDS</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ──
with st.sidebar:
    page = st.radio(
        "Navigate",
        ["📊 Analysis Dashboard", "🎯 Live Attack Simulation"],
        label_visibility="collapsed",
    )
    st.divider()

    if page == "📊 Analysis Dashboard":
        st.caption("Configure the run, then run the pipeline — same options as the desktop app.")
        st.subheader("What this demo shows")
        st.info(
            "Someone enters through the screen system → moves toward the car network "
            "→ aims at brakes (safety-critical)."
        )
        st.subheader("Demo settings")
        normal_count = st.selectbox(
            "Normal traffic size",
            list(config.NORMAL_MESSAGE_PRESETS),
            index=0,
        )
        attack_count = st.selectbox(
            "Injected attack messages",
            list(config.ATTACK_COUNT_PRESETS),
            index=0,
        )
        attack_type = st.selectbox(
            "Attack style",
            ["injection", "replay", "fuzzing", "high_speed_brake_injection"],
        )
        contam_choice = st.selectbox(
            "Detector sensitivity",
            ["auto", "0.01", "0.05", "0.10", "0.15"],
            index=0,
        )
        contamination = 0.05 if contam_choice == "auto" else float(contam_choice)
        stack_mean = st.slider(
            "Network delay average (μs)",
            5.0, 120.0,
            float(config.DEFAULT_SIMULATED_STACK_DELAY_US_MEAN), 1.0,
        )
        stack_jitter = st.slider(
            "Delay variation (μs)",
            0.0, 40.0,
            float(config.DEFAULT_SIMULATED_STACK_DELAY_US_JITTER), 1.0,
        )
        pretrained_ok = config.PRETRAINED_MODEL_PATH.exists()
        use_pretrained = pretrained_ok   # auto: use pretrained if available
        prefer_pqc    = False            # always HMAC-SHA3-256
        use_zta       = True             # Zero Trust always on
        st.divider()
        st.subheader("🚗 Vehicle Speed Context")
        vehicle_speed = st.slider("Vehicle Speed (km/h)", 0, 200, 0, step=10)
        # Show the speed band the slider maps to in real time
        if vehicle_speed >= 100:
            _band_preview = "🛣️ highway  (threshold → 0.50)"
        elif vehicle_speed >= 30:
            _band_preview = "🏙️ urban  (threshold → 0.55)"
        else:
            _band_preview = "🅿️ parked  (threshold → 0.65)"
        st.caption(f"**Selected band:** {_band_preview}")
        st.caption(
            "0x160 Wheel_Speed frames will encode this speed — "
            "SafetyDecisionLayer threshold adapts from first frame."
        )
        run_btn = st.button("▶ Run protection demo", type="primary", use_container_width=True, key="run_demo")
    else:
        # Live sim sidebar controls (defined below with the page)
        normal_count = 400
        attack_count = 50
        attack_type  = "injection"
        contamination = 0.05
        stack_mean   = float(config.DEFAULT_SIMULATED_STACK_DELAY_US_MEAN)
        stack_jitter = float(config.DEFAULT_SIMULATED_STACK_DELAY_US_JITTER)
        pretrained_ok = config.PRETRAINED_MODEL_PATH.exists()
        use_pretrained = pretrained_ok
        prefer_pqc   = False
        use_zta      = True
        vehicle_speed = 0
        run_btn      = False


# ── Main Pipeline ──
if run_btn:
    # Stage 1
    with st.status("Running CAN-Guard pipeline...", expanded=True) as status:
        st.write("Generating CAN traffic, attack simulation, and lateral-movement log...")
        threat_logger = ThreatPathLogger()
        train_df, test_df = generate_dataset(
            normal_count,
            attack_count,
            attack_type,
            threat_logger=threat_logger,
            speed_kmh=vehicle_speed,
        )
        test_df = apply_gateway_path_delay(
            test_df, mean_us=stack_mean, jitter_us=stack_jitter, seed=202
        )
        time.sleep(0.3)

        st.write("Edge AI model (train or bundled pretrained)...")
        model_path = "edge_model.joblib"
        if use_pretrained and pretrained_ok:
            model_path = str(config.PRETRAINED_MODEL_PATH)
            edge = EdgeAIDetector(contamination=contamination)
            edge.load_model(model_path)
        else:
            detector = EdgeAIDetector(contamination=contamination)
            detector.train(train_df)
            detector.save_model("edge_model.joblib")
            edge = EdgeAIDetector()
            edge.load_model("edge_model.joblib")
        time.sleep(0.2)

        st.write("Running detection (Isolation Forest + simulated stack latency)...")
        t_det = time.time()
        results = edge.predict(test_df, default_simulated_stack_us=0.0)
        metrics = edge.evaluate(results)
        metrics["wall_clock_batch_detection_s"] = round(time.time() - t_det, 4)
        time.sleep(0.2)

        st.write("Safety Decision Layer processing...")
        _speed_ctx = SpeedContextLayer(initial_speed=vehicle_speed)
        safety = SafetyDecisionLayer(zero_trust_enabled=use_zta, speed_context=_speed_ctx)
        decisions = safety.process_batch(results)
        safety_summary = safety.get_summary()
        time.sleep(0.2)

        st.write("Mitigation system and signing...")
        mitigation = MitigationSystem(prefer_pqc=prefer_pqc)
        mit_summary = mitigation.process_safety_decisions(decisions)
        mitigation.verify_all_incidents()

        status.update(label="Pipeline complete", state="complete")

    st.session_state["cg_last_run"] = {
        "metrics": metrics,
        "results": results,
        "safety_summary": safety_summary,
        "mit_summary": mit_summary,
        "mitigation": mitigation,
        "decisions": decisions,
        "threat_logger": threat_logger,
        "configured_speed_kmh": vehicle_speed,               # ← slider value (exact)
        "speed_context_kmh": safety.speed_context.current_speed_kmh,
        "speed_band_label": safety.speed_context.get_band_label(),
    }
    st.session_state.pop("sl_chat_messages", None)
    st.session_state.pop("sl_ai_insight", None)

cg = st.session_state.get("cg_last_run")
if page == "📊 Analysis Dashboard":
    if cg is not None:
        metrics = cg["metrics"]
        results = cg["results"]
        safety_summary = cg["safety_summary"]
        mit_summary = cg["mit_summary"]
        mitigation = cg["mitigation"]
        decisions = cg["decisions"]
        threat_logger = cg["threat_logger"]
        configured_speed = cg.get("configured_speed_kmh", 0)
        speed_context_kmh = cg.get("speed_context_kmh", 0.0)
        speed_band_label  = cg.get("speed_band_label", "parked")

    st.divider()

    # ── Before / After (neutral panels — same content as desktop Summary, no alert styling) ──
    st.header("Snapshot")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="cg-panel"><h4>Before (no protection)</h4></div>',
            unsafe_allow_html=True,
        )
        st.metric("Brake ECU protection", "NONE")
        st.metric("Intrusion detection", "NONE")
        st.metric("Attack success rate", "100%")
    with col2:
        st.markdown(
            '<div class="cg-panel"><h4>After (CAN-Guard AI)</h4></div>',
            unsafe_allow_html=True,
        )
        st.metric("Detection rate", f"{metrics['detection_rate']:.1%}")
        st.metric("False positive rate", f"{metrics['false_positive_rate']:.1%}")
        ae_snap = metrics.get("avg_edge_processing_latency_us") or 0
        ass_snap = metrics.get("avg_simulated_stack_latency_us") or 0
        st.metric("Avg total path latency", f"{metrics['avg_detection_latency_us']:.1f} μs")
        st.metric("Avg edge (IF) latency", f"{ae_snap:.1f} μs")
        st.metric("Avg sim. stack latency", f"{ass_snap:.1f} μs")
        st.metric("Safe mode activations", safety_summary["safe_mode_activations"])
        st.metric(
            "Speed Context",
            f"{configured_speed} km/h (configured) — {speed_band_label}",
        )

    st.divider()

    lm1, lm2, lm3 = st.columns(3)
    lm1.metric("Attacks Detected", int(metrics["true_positives"]))
    lm2.metric("Safe Mode Triggers", int(safety_summary["safe_mode_activations"]))
    lm3.metric("Speed Band", str(speed_band_label).upper())

    st.subheader("Live CAN Frame Monitor")
    st.caption(
        "Frame-by-frame view: ML anomaly score and safety action per message (time order). "
        "Green = allow, yellow = alert, orange/red = block or safe mode."
    )
    _rows: list[dict] = []
    _dec_list = decisions if isinstance(decisions, list) else list(decisions)
    for i, (_, frame) in enumerate(results.iterrows()):
        action = (
            _dec_list[i].action.value
            if i < len(_dec_list)
            else "ALLOW"
        )
        _band = (
            str(speed_band_label).upper()
            if action in ("SAFE_MODE", "BLOCK", "ALERT")
            else "—"
        )
        _rows.append(
            {
                "ECU": frame.get("ecu_name", ""),
                "CAN ID": frame.get("can_id_hex", ""),
                "Anomaly score": round(float(frame.get("anomaly_score", 0.0)), 4),
                "Confidence": f"{float(frame.get('confidence', 0.0)):.0%}",
                "Action": action,
                "Speed band": _band,
            }
        )
    df_live = pd.DataFrame(_rows)

    def _color_action(val: str) -> str:
        if val == "SAFE_MODE":
            return "background-color: #ff4444; color: white"
        if val == "BLOCK":
            return "background-color: #ff8800; color: white"
        if val == "ALERT":
            return "background-color: #ffcc00; color: black"
        if val == "ALLOW":
            return "background-color: #143d2a; color: #9ae6b4"
        return "background-color: #1a1a1a; color: #aaaaaa"

    try:
        _styled = df_live.style.map(_color_action, subset=["Action"])
    except AttributeError:
        _styled = df_live.style.applymap(_color_action, subset=["Action"])  # type: ignore[attr-defined]

    st.dataframe(_styled, height=400, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Detection metrics")
    
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
    m5.metric("True Positives", metrics['true_positives'])
    m6.metric("False Positives", metrics['false_positives'])
    m7.metric("Batch wall time (s)", f"{metrics.get('wall_clock_batch_detection_s', 0):.4f}")
    
    st.divider()
    
    st.header("Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Summary", "Charts", "Incidents", "Path", "Insights"]
    )

    with tab1:
        ae = metrics.get("avg_edge_processing_latency_us") or 0
        ass = metrics.get("avg_simulated_stack_latency_us") or 0
        overview = (
            f"BEFORE (no protection)\n"
            f"  Brake ECU protection: NONE\n"
            f"  Intrusion detection: NONE\n"
            f"  Attack success rate: 100%\n\n"
            f"AFTER (CAN-Guard AI)\n"
            f"  Detection rate:        {metrics['detection_rate']:.2%}\n"
            f"  False positive rate:   {metrics['false_positive_rate']:.2%}\n"
            f"  Accuracy:              {metrics['accuracy']:.2%}\n"
            f"  Precision / Recall / F1: {metrics['precision']:.2%} / {metrics['recall']:.2%} / {metrics['f1_score']:.2%}\n"
            f"  Avg total path latency: {metrics['avg_detection_latency_us']:.1f} μs\n"
            f"  Avg edge (IF) latency:  {ae:.1f} μs\n"
            f"  Avg sim. stack latency: {ass:.1f} μs\n"
            f"  Batch wall time:       {metrics.get('wall_clock_batch_detection_s', 0):.4f} s\n\n"
            f"Safety layer\n"
            f"  Allowed: {safety_summary['allowed']}  Alerts: {safety_summary['alerts']}  "
            f"Blocked: {safety_summary['blocked']}\n"
            f"  Safe mode activations: {safety_summary['safe_mode_activations']}\n\n"
            f"Mitigation\n"
            f"  Incidents: {mit_summary['total_incidents']}  Alerts: {mit_summary['total_alerts']}\n"
            f"  Signing: {mit_summary.get('signing_algorithm', 'n/a')}\n"
        )
        overview += format_prevented_threats_summary(mitigation)
        st.markdown(f'<div class="cg-mono">{html.escape(overview)}</div>', unsafe_allow_html=True)

    with tab2:
        # IF decision_function: same bin range for both histograms (comparable)
        rp = results.reset_index(drop=True)
        normal_scores = rp.loc[rp["is_malicious"] == 0, "anomaly_score"]
        attack_scores = rp.loc[rp["is_malicious"] == 1, "anomaly_score"]
        all_s = rp["anomaly_score"].values.astype(float)
        s_min = float(np.min(all_s))
        s_max = float(np.max(all_s))
        if s_max - s_min < 1e-12:
            s_min -= 0.5
            s_max += 0.5
        nbins = 30
        bin_size = (s_max - s_min) / nbins

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Ground-truth normal — score distribution",
                "Ground-truth malicious — score distribution",
            ),
        )
        fig.add_trace(
            go.Histogram(
                x=normal_scores,
                name="Normal",
                marker_color="#00CC96",
                opacity=0.75,
                xbins=dict(start=s_min, end=s_max + 1e-9, size=bin_size),
                autobinx=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=attack_scores,
                name="Malicious",
                marker_color="#FF4B4B",
                opacity=0.75,
                xbins=dict(start=s_min, end=s_max + 1e-9, size=bin_size),
                autobinx=False,
            ),
            row=1,
            col=2,
        )
        _xl = "IF decision_function (↓ more anomalous, ↑ more inlier-like)"
        fig.update_xaxes(title_text=_xl, row=1, col=1)
        fig.update_xaxes(title_text=_xl, row=1, col=2)
        fig.update_layout(height=420, showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        nm = rp["is_malicious"] == 0
        mk = rp["is_malicious"] == 1
        fig2.add_trace(
            go.Scatter(
                x=rp.index[nm],
                y=rp.loc[nm, "anomaly_score"],
                mode="markers",
                name="Normal",
                marker=dict(color="#00CC96", size=4, opacity=0.5),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=rp.index[mk],
                y=rp.loc[mk, "anomaly_score"],
                mode="markers",
                name="Malicious",
                marker=dict(color="#FF4B4B", size=10, symbol="x"),
            )
        )
        fig2.update_layout(
            title="Score vs message order",
            xaxis_title="Message index",
            yaxis_title="Isolation Forest score",
            height=420,
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("CAN ID counts")
        # Same CAN-ID categories for both traces (grouped — not stacked on misaligned x)
        id_counts_normal = results[results["is_malicious"] == 0]["can_id_hex"].value_counts()
        id_counts_attack = results[results["is_malicious"] == 1]["can_id_hex"].value_counts()
        keys = sorted(set(id_counts_normal.index) | set(id_counts_attack.index), key=str)

        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=keys,
                y=[id_counts_normal.get(k, 0) for k in keys],
                name="Normal subset",
                marker_color="#00CC96",
            )
        )
        fig3.add_trace(
            go.Bar(
                x=keys,
                y=[id_counts_attack.get(k, 0) for k in keys],
                name="Malicious subset",
                marker_color="#FF4B4B",
            )
        )
        fig3.update_layout(
            title="Message counts by CAN ID (normal vs malicious subsets)",
            barmode="group",
            height=420,
            template="plotly_dark",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Safety decisions")
        # Safety decision breakdown — colors must match labels (not list order)
        action_counts: dict[str, int] = {}
        for d in decisions:
            action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1

        color_map = {
            "allow": "#00CC96",
            "alert": "#FFD700",
            "block": "#FF4B4B",
            "safe_mode": "#FF6B00",
        }
        pie_labels = sorted(action_counts.keys(), key=str)
        pie_vals = [action_counts[k] for k in pie_labels]
        pie_colors = [color_map.get(str(l).lower(), "#888888") for l in pie_labels]

        fig4 = go.Figure(
            data=[
                go.Pie(
                    labels=pie_labels,
                    values=pie_vals,
                    hole=0.4,
                    marker=dict(colors=pie_colors),
                )
            ]
        )
        fig4.update_layout(title="Safety Decision Distribution", height=400, template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
        
        # Summary stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Messages Allowed", safety_summary["allowed"])
        col_b.metric("Alerts Raised", safety_summary["alerts"])
        col_c.metric("Blocked + Safe Mode", safety_summary["blocked"] + safety_summary["safe_mode_activations"])

        st.subheader("Confusion matrix")
        ztn, zfp, zfn, ztp = (
            metrics["true_negatives"],
            metrics["false_positives"],
            metrics["false_negatives"],
            metrics["true_positives"],
        )
        zmax = max(1, ztn, zfp, zfn, ztp)
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=[[ztn, zfp], [zfn, ztp]],
                x=["Predicted normal (inlier)", "Predicted anomaly"],
                y=["Actual normal", "Actual malicious"],
                text=[[ztn, zfp], [zfn, ztp]],
                texttemplate="%{text}",
                colorscale="RdBu_r",
                zmin=0,
                zmax=zmax,
                showscale=False,
            )
        )
        cm_fig.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(cm_fig, use_container_width=True)

    with tab3:
        # Incident table
        st.subheader(f"{mit_summary['total_incidents']} incidents logged")
        
        incident_data = []
        for inc in mitigation.incidents[: config.INCIDENT_TABLE_MAX_ROWS]:
            incident_data.append({
                "ID": inc.incident_id,
                "CAN ID": inc.can_id,
                "ECU": inc.ecu_name,
                "Attack Type": inc.attack_type,
                "Action": inc.action_taken,
                "Confidence": f"{inc.confidence:.2%}",
                "Critical": "yes" if inc.is_safety_critical else "",
                "Algorithm": getattr(inc, "signature_algorithm", "") or mit_summary.get("signing_algorithm", ""),
                "Signed": "yes" if inc.signature else "no",
            })
        
        st.dataframe(pd.DataFrame(incident_data), use_container_width=True, hide_index=True)
        
        st.info(
            f"Signing: {mit_summary.get('signing_algorithm', 'n/a')} — "
            f"{mit_summary['total_incidents']} incident reports signed."
        )

    with tab4:
        st.subheader("Path")
        st.caption("Threat path sequence (synthetic lateral-movement log).")
        st.json(threat_logger.to_json_ready())

    with tab5:
        st.subheader("Insights")
        llm = LLMInsightEngine()
        st.caption(
            "On-device explanation report (same baseline as the desktop Insights tab). "
            f"Ollama HTTP API: `{llm.base_url}` (set `OLLAMA_BASE_URL` or `OLLAMA_HOST` if not local)."
        )
        rep = build_explanation_report(
            metrics=metrics,
            results_df=results,
            safety_summary=safety_summary,
            mit_summary=mit_summary,
            mitigation=mitigation,
            decisions=decisions,
        )
        st.markdown(rep)

        st.divider()
        _ollama_ok = llm.is_server_reachable()
        st.caption(
            f"Ollama: **{'reachable' if _ollama_ok else 'offline'}** — model `{llm.model_name}` "
            f"(`OLLAMA_MODEL`). Start the Ollama app or run `ollama serve`, then `ollama pull {llm.model_name}`."
        )
        b1, _ = st.columns([1, 3])
        with b1:
            if st.button("Generate AI narrative (Ollama)", key="sl_ollama_narrative"):
                st.subheader("Ollama narrative")
                with st.chat_message("assistant"):
                    streamed = st.write_stream(llm.stream_insight(metrics, safety_summary, mit_summary))
                st.session_state.sl_ai_insight = streamed
        if st.session_state.get("sl_ai_insight") and not st.session_state.get("_insight_just_shown"):
            st.subheader("Ollama narrative")
            st.markdown(st.session_state.sl_ai_insight)

        st.divider()
        st.subheader("Assistant (Ollama)")
        if "sl_chat_messages" not in st.session_state:
            st.session_state.sl_chat_messages = []
        for msg in st.session_state.sl_chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        _chat_q = st.chat_input("Ask about this run…")
        if _chat_q:
            st.session_state.sl_chat_messages.append({"role": "user", "content": _chat_q})
            with st.chat_message("user"):
                st.markdown(_chat_q)
            with st.chat_message("assistant"):
                _streamed = st.write_stream(
                    llm.stream_chat(metrics, safety_summary, mit_summary, _chat_q)
                )
            st.session_state.sl_chat_messages.append({"role": "assistant", "content": _streamed})
            st.rerun()

        st.divider()
        st.subheader("Quick prompts")
        st.caption(
            "Same suggested questions as the desktop Assistant. "
            "Pick one to load the full prompt for copy/paste, or type in the assistant above."
        )
        qcols = st.columns(2)
        for i, (short, full) in enumerate(CHAT_SUGGESTIONS):
            with qcols[i % 2]:
                if st.button(short, key=f"sl_qp_{i}", use_container_width=True):
                    st.session_state.sl_prompt = full
        st.text_area(
            "Full prompt",
            height=180,
            key="sl_prompt",
            help="Updates when you tap a quick prompt above.",
        )

# ────────────────────────────────────────────────────────────────────────
# LIVE ATTACK SIMULATION PAGE
# ────────────────────────────────────────────────────────────────────────
elif page == "🎯 Live Attack Simulation":
    import time as _time

    from can_generator import (
        generate_normal_traffic as _gen_normal,
        inject_attack_traffic as _inject_atk,
        inject_high_speed_brake_attack as _inject_hspeed,
    )
    from gateway_simulator import apply_gateway_path_delay as _gw_delay
    from safety_layer import SafetyDecisionLayer as _SDL, SpeedContextLayer as _SCL
    from mitigation import MitigationSystem as _MS
    import signing as _signing_mod

    _SCENARIOS: dict[str, dict] = {
        "🔴  Brake Injection @ Speed": {
            "attack_type": "high_speed_brake_injection",
            "default_speed": 120, "normal_count": 400, "attack_count": 50,
            "desc": "Spoofed 0xFF emergency-brake frames on CAN ID 0x200 at highway speed.",
        },
        "🟠  Fuzzing Attack": {
            "attack_type": "fuzzing",
            "default_speed": 0, "normal_count": 350, "attack_count": 50,
            "desc": "Random CAN IDs and random payloads flooding the bus.",
        },
        "🟡  Replay Attack": {
            "attack_type": "replay",
            "default_speed": 0, "normal_count": 350, "attack_count": 40,
            "desc": "Rapid brake-disable frames replayed on 0x200.",
        },
        "🟣  Injection Attack": {
            "attack_type": "injection",
            "default_speed": 0, "normal_count": 350, "attack_count": 40,
            "desc": "Standard spoofed brake injection frames.",
        },
    }

    _ACTION_BADGE = {"ALLOW": "b-allow", "ALERT": "b-alert", "BLOCK": "b-blk", "SAFE_MODE": "b-safe"}
    _ACTION_EMOJI = {"ALLOW": "✅ ALLOW", "ALERT": "⚠️ ALERT", "BLOCK": "🚫 BLOCK", "SAFE_MODE": "🔴 SAFE MODE"}
    _ROW_CLASS    = {"ALLOW": "frame-normal", "ALERT": "frame-alert", "BLOCK": "frame-attack", "SAFE_MODE": "frame-safemode"}

    st.markdown("""
    <style>
    .frame-row{font-family:'Courier New',monospace;font-size:.76rem;padding:.35rem .7rem;border-radius:6px;
    margin-bottom:3px;display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
    .frame-normal{background:#0d1f14;border-left:3px solid #3fb950;color:#8b949e}
    .frame-attack{background:#1c0e0e;border-left:3px solid #f85149;color:#c9d1d9}
    .frame-safemode{background:#200e0e;border-left:4px solid #ff7b72;color:#c9d1d9}
    .frame-alert{background:#1c1608;border-left:3px solid #d29922;color:#c9d1d9}
    .badge{border-radius:999px;padding:1px 8px;font-size:.68rem;font-weight:700;white-space:nowrap}
    .b-atk{background:#3d1f1e;color:#f85149;border:1px solid #5a1010}
    .b-norm{background:#1a2f1f;color:#3fb950;border:1px solid #0f3d20}
    .b-blk{background:#3d1f1e;color:#ff7b72;border:1px solid #5a1010}
    .b-allow{background:#1a2f1f;color:#3fb950;border:1px solid #0f3d20}
    .b-alert{background:#2d2309;color:#d29922;border:1px solid #3d3000}
    .b-safe{background:#3d1f1e;color:#ff7b72;border:1px solid #5a1010}
    .lscard{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:.9rem 1rem;text-align:center}
    .lscard .lbl{color:#8b949e;font-size:.68rem;text-transform:uppercase;letter-spacing:.07em}
    .lscard .val{font-size:1.9rem;font-weight:700;margin-top:.25rem}
    .v-blue{color:#58a6ff}.v-green{color:#3fb950}.v-yellow{color:#d29922}.v-red{color:#f85149}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar controls for live sim
    with st.sidebar:
        st.markdown("### ⏱ Live Sim Settings")
        _scen_name   = st.selectbox("Attack Scenario", list(_SCENARIOS.keys()))
        _scfg        = _SCENARIOS[_scen_name]
        st.caption(_scfg["desc"])
        st.divider()
        _sim_speed   = st.slider("🚗 Vehicle Speed (km/h)", 0, 200, _scfg["default_speed"], 10)
        if _sim_speed >= 100:
            st.caption(f"**Band:** 🛣️ highway (threshold → 0.50)")
        elif _sim_speed >= 30:
            st.caption(f"**Band:** 🏙️ urban (threshold → 0.55)")
        else:
            st.caption(f"**Band:** 🅿️ parked (threshold → 0.65)")
        _norm_cnt    = st.slider("Normal Frames",  100, 800, _scfg["normal_count"],  50)
        _atk_cnt     = st.slider("Attack Frames",   10, 100, _scfg["attack_count"],   5)
        _norm_delay  = st.slider("Normal frame delay (ms)", 0, 200, 30, 5)
        _atk_delay   = st.slider("Attack frame delay (ms)", 0, 500, 150, 10)
        _show_all    = st.checkbox("Show all normal frames", value=False)
        st.divider()
        _run_sim_btn = st.button("▶ START LIVE SIMULATION", type="primary", use_container_width=True, key="live_sim_run")

    # Init state
    for _k, _v in [("ls_frames", []), ("ls_running", False), ("ls_done", False),
                   ("ls_allowed", 0), ("ls_alerts", 0), ("ls_blocked", 0),
                   ("ls_safe", 0), ("ls_incidents", 0), ("ls_sig_ok", None)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    @st.cache_resource(show_spinner="Loading Edge AI model…")
    def _get_det_live():
        _d = EdgeAIDetector(contamination=0.05)
        _p = config.PRETRAINED_MODEL_PATH
        if _p.exists():
            _d.load_model(str(_p))
        else:
            _d.train(_gen_normal(1000, seed=42), verbose=False)
        return _d

    def _frame_html(f: dict) -> str:
        tc  = "b-atk"  if f["is_attack"] else "b-norm"
        tt  = "ATTACK" if f["is_attack"] else "NORMAL"
        ac  = _ACTION_BADGE[f["action"]]
        at  = _ACTION_EMOJI[f["action"]]
        rc  = _ROW_CLASS[f["action"]]
        sc  = "#f85149" if f["anomaly_score"] < 0 else "#3fb950"
        rs  = f'<span style="color:#6e7681;font-size:.68rem">&#x21B3; {f["reason"][:90]}</span>' if f["reason"] else ""
        return (
            f'<div class="{rc} frame-row">'
            f'<span style="color:#6e7681">#{f["idx"]:04d}</span>'
            f'<span style="color:#79c0ff">{f["can_id"]}</span>'
            f'<span style="color:#c9d1d9;min-width:140px">{f["ecu"]}</span>'
            f'<span style="color:#6e7681">{f["iat_ms"]:.2f}ms</span>'
            f'<span style="color:#484f58">[{f["payload"]}]</span>'
            f'<span class="badge {tc}">{tt}</span>'
            f'<span class="badge {ac}">{at}</span>'
            f'<span style="color:#8b949e">conf {f["confidence"]:.2f}</span>'
            f'<span style="color:{sc}">{f["anomaly_score"]:+.4f}</span>'
            f'{rs}</div>'
        )

    def _card(label, value, color):
        return (f'<div class="lscard"><div class="lbl">{label}</div>'
                f'<div class="val v-{color}">{value}</div></div>')

    # Placeholders
    _met_ph  = st.empty()
    _prog_ph = st.empty()
    _stat_ph = st.empty()
    _feed_ph = st.empty()
    _chart_ph = st.empty()
    _sig_ph  = st.empty()

    # Idle splash
    if not st.session_state.ls_running and not st.session_state.ls_frames:
        _feed_ph.markdown("""
        <div style="text-align:center;padding:3rem 2rem;background:#0d1117;
             border:1px solid #30363d;border-radius:14px">
            <div style="font-size:2.5rem">📡</div>
            <h2 style="color:#c9d1d9;margin:.5rem 0">CAN-Guard AI · Live Attack Simulator</h2>
            <p style="color:#8b949e;max-width:480px;margin:0 auto">Select a scenario in the sidebar
            and click <strong>▶ START LIVE SIMULATION</strong> to watch frames arrive in real time.</p>
        </div>""", unsafe_allow_html=True)

    if _run_sim_btn:
        st.session_state.update(
            ls_running=True, ls_frames=[],
            ls_allowed=0, ls_alerts=0, ls_blocked=0, ls_safe=0,
            ls_incidents=0, ls_sig_ok=None,
        )
        _det = _get_det_live()
        _atype = _scfg["attack_type"]

        with st.spinner("Generating CAN traffic and running AI detection…"):
            _norm_df = _gen_normal(_norm_cnt, seed=42, speed_kmh=_sim_speed)
            if _atype == "high_speed_brake_injection":
                _test_df = _inject_hspeed(_norm_df, _atk_cnt, speed_kmh=_sim_speed)
            else:
                _test_df = _inject_atk(_norm_df, _atk_cnt, _atype)
            _test_df = _gw_delay(_test_df, mean_us=48.0, jitter_us=12.0, seed=202)
            _results = _det.predict(_test_df)

        _total   = len(_results)
        _spd_ctx = _SCL(initial_speed=float(_sim_speed))
        _safety  = _SDL(zero_trust_enabled=True, speed_context=_spd_ctx)
        _mit     = _MS()

        _vis_html: list[str] = []
        _norm_shown = 0
        _scores: list[dict] = []

        _stat_ph.markdown(
            '<p style="color:#3fb950;font-weight:600">▶ Simulation running…</p>',
            unsafe_allow_html=True,
        )

        for _step, (_idx, _row) in enumerate(_results.iterrows(), 1):
            _rd = _row.to_dict()
            _dec = _safety.decide(_rd)
            _is_mal = int(_rd.get("is_malicious", 0))
            _action = _dec.action.value
            _pld = [_rd.get(f"payload_byte_{i}", 0) for i in range(8)]

            _f = {
                "idx":           _idx,
                "can_id":        _rd.get("can_id_hex", hex(int(_rd.get("can_id", 0)))),
                "ecu":           _rd.get("ecu_name", "?"),
                "iat_ms":        round(float(_rd.get("inter_arrival_time", 0)) * 1000, 3),
                "payload":       " ".join(f"{int(b):02X}" for b in _pld),
                "is_attack":     _is_mal,
                "action":        _action,
                "confidence":    round(float(_rd.get("confidence", 0)), 3),
                "anomaly_score": round(float(_rd.get("anomaly_score", 0)), 4),
                "reason":        _dec.reason[:90] if _action in ("BLOCK", "SAFE_MODE") else "",
            }
            st.session_state.ls_frames.append(_f)
            _scores.append({"idx": _idx, "score": _f["anomaly_score"],
                            "is_attack": _is_mal, "action": _action})

            if _action in ("BLOCK", "SAFE_MODE", "ALERT"):
                _mit.process_safety_decisions([_dec])

            if _action == "ALLOW":     st.session_state.ls_allowed   += 1
            elif _action == "ALERT":   st.session_state.ls_alerts    += 1
            elif _action == "BLOCK":   st.session_state.ls_blocked   += 1
            elif _action == "SAFE_MODE": st.session_state.ls_safe    += 1
            st.session_state.ls_incidents = len(_mit.incidents)

            # Throttle display
            _show = True
            _delay = _norm_delay / 1000.0
            if not _is_mal:
                _norm_shown += 1
                if not _show_all and _norm_shown > 8 and _norm_shown % 10 != 0:
                    _show = False
            else:
                _delay = _atk_delay / 1000.0

            if _show:
                _vis_html.append(_frame_html(_f))
                _disp = _vis_html[-80:]

            # Live metrics
            _met_ph.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem;margin-bottom:.4rem">
                {_card("Total", _step, "blue")}
                {_card("✅ Allowed", st.session_state.ls_allowed, "green")}
                {_card("⚠️ Alerts", st.session_state.ls_alerts, "yellow")}
                {_card("🚫 Blocked", st.session_state.ls_blocked, "red")}
                {_card("🔴 Safe Mode", st.session_state.ls_safe, "red")}
                {_card("📋 Incidents", st.session_state.ls_incidents, "blue")}
            </div>""", unsafe_allow_html=True)

            # Progress bar
            _pct = int(_step / _total * 100)
            _bc  = "#f85149" if _is_mal else "#3fb950"
            _prog_ph.markdown(
                f'<div style="background:#21262d;border-radius:999px;height:6px;margin-bottom:.4rem">'
                f'<div style="background:{_bc};width:{_pct}%;height:100%;border-radius:999px"></div></div>',
                unsafe_allow_html=True,
            )

            if _show:
                _feed_ph.markdown(
                    f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;'
                    f'padding:.6rem;max-height:460px;overflow-y:auto">{" ".join(_disp)}</div>',
                    unsafe_allow_html=True,
                )

            _time.sleep(_delay)

        # ─ Final summary ───────────────────────────────────────────────────────────────
        _ss  = _safety.get_summary()
        _ms_s = _mit.get_summary()
        _stat_ph.markdown(
            f'<p style="color:#58a6ff;font-weight:600">✅ Done — {_total} frames · {_scen_name} · '
            f'{_sim_speed} km/h</p>',
            unsafe_allow_html=True,
        )
        _prog_ph.markdown(
            '<div style="background:#3fb950;height:6px;border-radius:999px"></div>',
            unsafe_allow_html=True,
        )
        _met_ph.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem;margin-bottom:.4rem">
            {_card("Total", _ss['total_messages'], "blue")}
            {_card("✅ Allowed", _ss['allowed'], "green")}
            {_card("⚠️ Alerts", _ss['alerts'], "yellow")}
            {_card("🚫 Blocked", _ss['blocked'], "red")}
            {_card("🔴 Safe Mode", _ss['safe_mode_activations'], "red")}
            {_card("📋 Incidents", _ms_s['total_incidents'], "blue")}
        </div>""", unsafe_allow_html=True)

        # Signature check
        _all_ok = True
        _n_sig  = len(_mit.incidents)
        for _inc in _mit.incidents:
            if _inc.signature:
                _d2 = {k: v for k, v in vars(_inc).items() if k not in ("signature", "signature_algorithm")}
                if not _mit.verify_signature(_signing_mod.canonical_incident_json(_d2), _inc.signature):
                    _all_ok = False
        if _n_sig == 0:
            _sig_ph.markdown('<div style="background:#161b22;border:1px solid #30363d;color:#8b949e;border-radius:8px;padding:.5rem 1rem">ℹ️ No incidents to sign.</div>', unsafe_allow_html=True)
        elif _all_ok:
            _sig_ph.markdown(f'<div style="background:#1a2f1f;border:1px solid #0f3d20;color:#3fb950;border-radius:8px;padding:.5rem 1rem">🔐 All {_n_sig} incident signatures verified — HMAC-SHA3-256 ✓</div>', unsafe_allow_html=True)
        else:
            _sig_ph.markdown(f'<div style="background:#3d1f1e;border:1px solid #5a1010;color:#f85149;border-radius:8px;padding:.5rem 1rem">❌ Signature FAILED on one or more of {_n_sig} incidents.</div>', unsafe_allow_html=True)

        # Anomaly score chart
        _df_s = pd.DataFrame(_scores)
        if not _df_s.empty:
            import plotly.graph_objects as _go
            _fig = _go.Figure()
            _nd  = _df_s[_df_s["is_attack"] == 0]
            _ad  = _df_s[_df_s["is_attack"] == 1]
            _bd  = _df_s[_df_s["action"].isin(["BLOCK", "SAFE_MODE"])]
            _fig.add_trace(_go.Scatter(x=_nd["idx"], y=_nd["score"], mode="markers",
                                       name="Normal", marker=dict(color="#3fb950", size=4, opacity=0.5)))
            _fig.add_trace(_go.Scatter(x=_ad["idx"], y=_ad["score"], mode="markers",
                                       name="Attack", marker=dict(color="#f85149", size=7, symbol="x")))
            _fig.add_trace(_go.Scatter(x=_bd["idx"], y=_bd["score"], mode="markers",
                                       name="Blocked", marker=dict(color="#ff7b72", size=11,
                                       symbol="x-open", line=dict(width=2))))
            _fig.add_hline(y=0, line_dash="dot", line_color="#30363d")
            _fig.update_layout(
                title="📈 Anomaly Score Timeline",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=300,
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            _chart_ph.plotly_chart(_fig, use_container_width=True)

        _df_exp = pd.DataFrame(st.session_state.ls_frames)
        st.download_button(
            "⬇️ Download Frame Log (CSV)",
            _df_exp.to_csv(index=False).encode("utf-8"),
            file_name="can_guard_live_sim.csv", mime="text/csv",
        )
        st.session_state.ls_running = False

