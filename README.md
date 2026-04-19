# 🛡️ CAN-Guard AI — Automotive Intrusion Detection System

> **Edge AI security gateway for in-vehicle CAN bus networks**  
> MAHE Mobility Challenge 2026

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is CAN-Guard AI?

Modern vehicles connect 70+ ECUs (Engine, Brakes, Steering, Airbags…) over a shared **CAN bus** with **no authentication**. An attacker who gains access — through infotainment, OBD-II, or Bluetooth — can send spoofed brake commands or disable safety systems.

**CAN-Guard AI** is a software gateway that sits on the CAN bus and:
1. **Scores every frame** using an Isolation Forest anomaly detector trained on normal traffic
2. **Decides** allow / alert / block / safe-mode based on AI score + Zero Trust rules + vehicle speed
3. **Signs** every incident report with HMAC-SHA3-256 for tamper-evident forensics
4. **Encrypts** all AI audit logs at rest with AES-256 (Fernet)

---

## Architecture

```
 CAN Bus Frames
       │
       ▼
┌─────────────────────────┐
│   Gateway Simulator     │  ← adds realistic network latency (configurable μs)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Edge AI Detector       │  ← Isolation Forest, <30 μs per frame
│  (detection_engine.py)  │
└──────────┬──────────────┘
           │  anomaly score + confidence
           ▼
┌─────────────────────────┐
│  Safety Decision Layer  │  ← Zero Trust + speed-aware thresholds
│  (safety_layer.py)      │     parked 0.65 / urban 0.55 / highway 0.50
└──────────┬──────────────┘
           │  ALLOW / ALERT / BLOCK / SAFE_MODE
           ▼
┌─────────────────────────┐
│  Mitigation System      │  ← HMAC-SHA3-256 signed incident reports
│  (mitigation.py)        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Encrypted AI Logger    │  ← AES-256 encrypted stores at rest
│  (ai_security.py)       │
└─────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| 🤖 **Edge AI Detection** | Isolation Forest — unsupervised anomaly scoring, no labelled data needed |
| 🔒 **Zero Trust Architecture** | Every ECU identity verified; unknown senders default-denied |
| 🚗 **Speed-Aware Thresholds** | Adaptive detection sensitivity: parked / urban / highway |
| 🛡️ **Cryptographic Signing** | Every blocked frame's incident report signed with HMAC-SHA3-256 |
| 🔐 **Encrypted AI Logs** | All LLM prompts + responses encrypted to `.enc` files via AES-256 |
| 📡 **Live Attack Simulation** | Real-time Streamlit dashboard — watch attacks arrive and get blocked frame-by-frame |
| 💬 **On-device AI Assistant** | Streaming Ollama chat (llama3.2:1b) — ask questions about any run |
| 📊 **Analytics Dashboard** | Confusion matrix, anomaly score timeline, CAN ID breakdown, safety pie chart |

---

## Attack Types Supported

| Attack | Description |
|---|---|
| **Brake Injection** | Spoofed `0xFF×8` emergency-brake frames on CAN ID `0x200` |
| **High-Speed Brake Injection** | Same, at highway speed — tightest detection threshold (0.50) |
| **Replay Attack** | Rapid replay of captured brake-disable frames |
| **Fuzzing** | Random CAN IDs and random payloads flooding the bus |

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/<your-username>/can-guard-ai.git
cd can-guard-ai

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Set up Ollama for AI chat

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:1b
ollama serve
```

### 3. Run the dashboard

```bash
# Streamlit only (recommended)
streamlit run dashboard.py

# Both Streamlit + CustomTkinter launcher
python start_can_guard.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
can-guard-ai/
├── dashboard.py              # Unified Streamlit app (Analysis + Live Sim)
├── can_generator.py          # Synthetic CAN traffic + attack injection
├── detection_engine.py       # Isolation Forest edge AI detector
├── safety_layer.py           # Zero Trust + speed-aware decision layer
├── mitigation.py             # Incident logging + HMAC-SHA3-256 signing
├── gateway_simulator.py      # Network latency simulation
├── insight_engine.py         # Ollama LLM integration (streaming)
├── ai_security.py            # AES-256 encrypted AI audit logger
├── signing.py                # Cryptographic signing utilities
├── config.py                 # Global configuration
├── live_demo.py              # Terminal-based frame-by-frame demo
├── start_can_guard.py        # Desktop launcher (CustomTkinter + Streamlit)
├── models/
│   └── pretrained.joblib     # Pre-trained Isolation Forest model
├── scripts/
│   └── build_pretrained_model.py
└── requirements.txt
```

---

## Dashboard Pages

### 📊 Analysis Dashboard
Run the full pipeline → see batch metrics, confusion matrix, incident table, AI narrative

### 🎯 Live Attack Simulation
Frame-by-frame real-time demo:
- Choose scenario (Brake Injection / Fuzzing / Replay / Injection)
- Set vehicle speed → watch the safety threshold change live
- Each frame appears with colour-coded action: ✅ ALLOW / ⚠️ ALERT / 🚫 BLOCK / 🔴 SAFE MODE
- HMAC signature verified at the end of every run

---

## Security Model

### Encryption at Rest
All AI agent data is encrypted before hitting disk:

| File | Contents |
|---|---|
| `ai_secure_store.enc` | All LLM prompts + AI-generated responses |
| `ai_metadata_store.enc` | Model endpoint, call counts, runtime stats |
| `ai_session_store.enc` | Session init data, platform info |

> ⚠️ `ai_crypto.key` is **excluded** from git (see `.gitignore`). In production, use a HSM or secrets manager.

### Incident Signing
Every blocked/alerted incident is signed:
```
HMAC-SHA3-256(canonical_incident_json) → stored alongside incident
```
All signatures are verified at end of run. Tampered logs are detected.

---

## Performance

| Metric | Value |
|---|---|
| Avg edge (IF) latency | ~25 μs per frame |
| Avg total path latency | ~73 μs (incl. simulated stack) |
| Detection rate | ~87–100% depending on attack type |
| False positive rate | ~9–21% (ZTA strict mode) |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) (optional — for AI chat/brief)
- macOS / Linux (CustomTkinter for desktop launcher)

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built for the MAHE Mobility Challenge 2026 · CAN-Guard AI Team*
