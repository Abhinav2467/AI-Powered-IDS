#!/usr/bin/env python3
import time
import sys
import os

def print_banner():
    print("""
============================================================
██████╗  █████╗ ███╗   ██╗     ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
██╔════╝ ██╔══██╗████╗  ██║    ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██║      ███████║██╔██╗ ██║    ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║      ██╔══██║██║╚██╗██║    ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
╚██████╗██║  ██║██║ ╚████║    ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
        MAHE Mobility Challenge 2026 - Edge AI Defense Sandbox
============================================================
    """)

def main():
    print_banner()
    print("[SYSTEM] Initializing Master Deployment Sequence...")
    time.sleep(0.5)

    if not os.path.exists("03_tkinter_dashboard.py"):
        print("[WARNING] Dashboard module missing!")
    else:
        print("[OK] Dashboard module (03_tkinter_dashboard.py) present.")
    time.sleep(0.3)

    print("[SYSTEM] Loading LLM Insight Engine module…")
    try:
        from insight_engine import LLMInsightEngine
    except ImportError:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("insight_engine", "02_insight_engine.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[WARNING] 02_insight_engine.py error: {e}")

    time.sleep(0.3)
    print(
        "[INFO] Optional local LLM: install Ollama and run `ollama pull llama3.2:1b` "
        "(API at http://127.0.0.1:11434). Offline heuristics apply if Ollama is down."
    )
    
    time.sleep(0.5)
    print("\n[SYSTEM] Launching Professional Monitoring Dashboard...")
    print("Press Ctrl+C in this terminal to gracefully terminate the system.")
    
    # Launch Dashboard
    sys.path.append(os.path.abspath('..'))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tk_dashboard", "03_tkinter_dashboard.py")
        tk_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tk_mod)
        app = tk_mod.CanGuardTkApp()
        app.run()
    except Exception as e:
        print(f"[ERROR] Failed to launch dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
