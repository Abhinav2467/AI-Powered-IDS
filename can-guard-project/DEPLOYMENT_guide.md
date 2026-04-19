# CAN-Guard AI Deployment Guide

This guide is designed for deploying the real-time anomaly detection and mitigation engine onto a simulated Edge Gateway/ECU for the MAHE Mobility Challenge 2026.

## Prerequisites

Ensure your edge/simulation device has Python 3.9+ installed and the necessary libraries (from the **repository root**, one level above this folder):
```bash
cd ..
pip install -r requirements.txt
```
*(Core packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`, `customtkinter`, `streamlit`, `plotly`, etc.)*

## Local LLM Setup (Optimized for Hackathon Edge Simulation)

To use the AI-assisted LLM Insight Engine, no cloud APIs are needed! We now use an ultra-fast local edge model via **Ollama**.

1. Download and install [Ollama](https://ollama.ai/)
2. Open a terminal and run your edge model:
   ```bash
   ollama run llama3.2:1b
   ```
*(Ollama binds to `localhost:11434` automatically). If Ollama isn't running, the system will perform perfectly using graceful offline heuristics—ensuring your demo never crashes!*

## Executing the Simulation

Use the main integration bootstrap script to start everything automatically:

```bash
cd can-guard-project
python3 04_main_integration.py
```

This script loads the CustomTkinter dashboard (`03_tkinter_dashboard.py`), which trains or loads the edge Isolation Forest via `detection_engine.py` (and optional `models/pretrained.joblib` from the repo root).

## Performance Tuning
Isolation Forest hyperparameters (e.g. **contamination**, tree count) live in **`detection_engine.py`** and the dashboard’s Run settings—not in a separate legacy script. 
