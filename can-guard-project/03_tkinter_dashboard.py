"""
CAN-Guard AI — Tkinter Demo Dashboard
=====================================
Same pipeline as Streamlit; native desktop UI with embedded matplotlib charts.

Run from this folder:  python3 04_main_integration.py
  (or: python3 03_tkinter_dashboard.py)

Charts require matplotlib. If you see ImportError, from the repo root install:
  python3 -m pip install matplotlib
  python3 -m pip install -r ../requirements.txt
"""

from __future__ import annotations

import json
import re
import threading
import tkinter as tk

import numpy as np
from tkinter import ttk, messagebox
import sys
import os
import customtkinter as ctk

# Configure global CustomTkinter themes (modern dark + blue accent)
ctk.set_appearance_mode("dark")
try:
    ctk.set_default_color_theme("dark-blue")
except Exception:
    ctk.set_default_color_theme("blue")

sys.path.append(os.path.abspath('..'))
try:
    from insight_engine import LLMInsightEngine, build_distilled_dashboard_context
except ImportError:
    # Try importing with 02_ prefix
    import importlib.util
    spec = importlib.util.spec_from_file_location("insight_engine", "02_insight_engine.py")
    insight_engine_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(insight_engine_mod)
    LLMInsightEngine = insight_engine_mod.LLMInsightEngine
    build_distilled_dashboard_context = insight_engine_mod.build_distilled_dashboard_context


try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    plt.style.use("dark_background")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    FigureCanvasTkAgg = None  # type: ignore[assignment]
    NavigationToolbar2Tk = None  # type: ignore[assignment]

import config
from can_generator import generate_dataset
from detection_engine import EdgeAIDetector
from gateway_simulator import apply_gateway_path_delay
from insights_report import build_explanation_report
from mitigation import MitigationSystem
from safety_layer import SafetyDecisionLayer
from threat_path import ThreatPathLogger

from assistant_prompts import CHAT_SUGGESTIONS
from prevented_threats_summary import format_prevented_threats_summary

# LLM panels use tk.Text (tags for headings / bold). Summary uses CTkTextbox like other metric tabs.
_LLM_FONT = "Helvetica"
_LLM_SIZE_BODY = 15
_LLM_SIZE_HEAD = 17


def _configure_llm_text_tags(w: tk.Text) -> None:
    w.tag_configure(
        "h2",
        font=(_LLM_FONT, _LLM_SIZE_HEAD, "bold"),
        foreground="#f9fafb",
        spacing1=12,
        spacing3=8,
    )
    w.tag_configure(
        "h3",
        font=(_LLM_FONT, _LLM_SIZE_HEAD - 2, "bold"),
        foreground="#e5e7eb",
        spacing1=8,
        spacing3=4,
    )
    w.tag_configure(
        "body",
        font=(_LLM_FONT, _LLM_SIZE_BODY),
        foreground="#d1d5db",
        spacing1=3,
        spacing3=2,
    )
    w.tag_configure(
        "bold",
        font=(_LLM_FONT, _LLM_SIZE_BODY, "bold"),
        foreground="#f3f4f6",
    )
    w.tag_configure(
        "bullet",
        font=(_LLM_FONT, _LLM_SIZE_BODY),
        foreground="#d1d5db",
        lmargin1=18,
        lmargin2=36,
        spacing1=2,
    )
    w.tag_configure("meta", font=(_LLM_FONT, 13), foreground="#8e8ea0")
    w.tag_configure(
        "chat_you",
        font=(_LLM_FONT, 13, "bold"),
        foreground="#93c5fd",
        spacing1=10,
    )
    w.tag_configure(
        "chat_user_msg",
        font=(_LLM_FONT, _LLM_SIZE_BODY),
        foreground="#ececec",
    )
    w.tag_configure(
        "chat_assistant_label",
        font=(_LLM_FONT, 13, "bold"),
        foreground="#10a37f",
        spacing1=8,
    )


def _insert_line_with_bold(w: tk.Text, line: str, *base_tags: str) -> None:
    tags = base_tags if base_tags else ("body",)
    for part in re.split(r"(\*\*.+?\*\*)", line):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            w.insert(tk.END, part[2:-2], tags + ("bold",))
        else:
            w.insert(tk.END, part, tags)


def _heading_plain(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", s)


def _insert_formatted_llm(w: tk.Text, text: str) -> None:
    """Render assistant markdown: ## / ###, bullets, **bold** (headings strip inline ** for simplicity)."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return
    lines = text.split("\n")
    prev_empty = True
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                w.insert(tk.END, "\n", ("body",))
            prev_empty = True
            continue
        prev_empty = False
        if stripped.startswith("### "):
            w.insert(tk.END, _heading_plain(stripped[4:]) + "\n", ("h3",))
        elif stripped.startswith("## "):
            w.insert(tk.END, _heading_plain(stripped[3:]) + "\n", ("h2",))
        elif stripped.startswith("# ") and not stripped.startswith("##"):
            w.insert(tk.END, _heading_plain(stripped[2:]) + "\n", ("h2",))
        elif stripped.startswith(("- ", "* ")):
            w.insert(tk.END, "• ", ("bullet",))
            _insert_line_with_bold(w, stripped[2:].strip(), "bullet")
            w.insert(tk.END, "\n", ("bullet",))
        elif re.match(r"^\d+\.\s", stripped):
            w.insert(tk.END, "   ", ("body",))
            _insert_line_with_bold(w, stripped, "body")
            w.insert(tk.END, "\n", ("body",))
        else:
            _insert_line_with_bold(w, line.rstrip(), "body")
            w.insert(tk.END, "\n", ("body",))


def _pack_styled_llm_text(
    parent,
    *,
    bg: str,
    padx: int,
    pady: int,
) -> tk.Text:
    outer = ctk.CTkFrame(parent, fg_color="transparent")
    outer.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
    inner = ctk.CTkFrame(outer, fg_color=bg, corner_radius=12, border_width=1, border_color="#333333")
    inner.pack(fill=tk.BOTH, expand=True)
    text = tk.Text(
        inner,
        wrap=tk.WORD,
        font=(_LLM_FONT, _LLM_SIZE_BODY),
        bg=bg,
        fg="#d1d5db",
        relief=tk.FLAT,
        borderwidth=0,
        highlightthickness=0,
        padx=20,
        pady=18,
        state=tk.DISABLED,
        cursor="arrow",
        selectbackground="#404040",
        selectforeground="#ffffff",
    )
    sb = tk.Scrollbar(inner, command=text.yview, width=14)
    text.configure(yscrollcommand=sb.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    _configure_llm_text_tags(text)
    return text


def _run_pipeline(
    normal_count: int,
    attack_count: int,
    attack_type: str,
    contamination: float,
    stack_mean: float,
    stack_jitter: float,
    use_pretrained: bool,
    prefer_pqc: bool,
    use_zta: bool,
) -> dict:
    """Execute pipeline; returns serializable context for the UI."""
    threat_logger = ThreatPathLogger()
    train_df, test_df = generate_dataset(
        normal_count, attack_count, attack_type, threat_logger=threat_logger
    )
    test_df = apply_gateway_path_delay(
        test_df, mean_us=stack_mean, jitter_us=stack_jitter, seed=202
    )

    pretrained_ok = config.PRETRAINED_MODEL_PATH.exists()
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

    import time as _t

    t0 = _t.time()
    results = edge.predict(test_df, default_simulated_stack_us=0.0)
    metrics = edge.evaluate(results)
    metrics["wall_clock_batch_detection_s"] = round(_t.time() - t0, 4)

    safety = SafetyDecisionLayer(zero_trust_enabled=use_zta)
    decisions = safety.process_batch(results)
    safety_summary = safety.get_summary()

    mitigation = MitigationSystem(prefer_pqc=prefer_pqc)
    mit_summary = mitigation.process_safety_decisions(decisions)
    mitigation.verify_all_incidents()

    return {
        "results": results,
        "metrics": metrics,
        "decisions": decisions,
        "safety_summary": safety_summary,
        "mitigation": mitigation,
        "mit_summary": mit_summary,
        "threat_logger": threat_logger,
    }


class CanGuardTkApp:
    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title("CAN-Guard — vehicle network shield")
        self.root.geometry("1480x880")
        self.root.minsize(1120, 740)

        self._status = tk.StringVar(value="Pick options on the left, then press Run to start the demo.")
        if not HAS_MPL:
            self._status.set(
                "Charts disabled (matplotlib not installed). Overview & tables still work. "
                "Install: python3 -m pip install matplotlib"
            )
        self._build_ui()

    def _build_ui(self) -> None:
        main = ctk.CTkFrame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Draggable sashes between: controls | tabs | AI chat (all grow with the window)
        paned = tk.PanedWindow(
            main,
            orient=tk.HORIZONTAL,
            sashwidth=8,
            sashpad=2,
            relief=tk.FLAT,
            bg="#2b2b2b",
            bd=0,
        )
        paned.pack(fill=tk.BOTH, expand=True)

        # ── Left: controls (plain language) ──
        left = ctk.CTkFrame(paned, corner_radius=12)

        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill=tk.X, padx=12, pady=(12, 8))
        title_box = ctk.CTkFrame(head, fg_color="transparent")
        title_box.pack(side=tk.LEFT, fill=tk.X)
        ctk.CTkLabel(
            title_box,
            text="CAN-Guard",
            font=("Helvetica", 20, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="AI Based In-House IDS",
            font=("Helvetica", 12),
            text_color="gray70",
        ).pack(anchor="w")

        scroll = ctk.CTkScrollableFrame(left)
        scroll.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        f_threat = ctk.CTkFrame(scroll, corner_radius=10)
        f_threat.pack(fill=tk.X, pady=(0, 8))
        ctk.CTkLabel(f_threat, text="What this demo shows", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(
            f_threat,
            text="Someone enters through the screen system\n→ moves toward the car network\n→ aims at brakes (safety-critical).",
            justify=tk.LEFT,
            font=("Helvetica", 12),
            text_color="gray70",
        ).pack(anchor="w", padx=10, pady=(4, 10))

        f_params = ctk.CTkFrame(scroll, fg_color="transparent")
        f_params.pack(fill=tk.BOTH, expand=True)
        ctk.CTkLabel(f_params, text="Demo settings", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=4, pady=(0, 6))

        self.var_normal = tk.StringVar(value=str(config.NORMAL_MESSAGE_PRESETS[0]))
        self.var_attack = tk.StringVar(value=str(config.ATTACK_COUNT_PRESETS[0]))
        self.var_attack_type = tk.StringVar(value="injection")
        self.var_contamination = tk.StringVar(value="auto")
        self.var_stack_mean = tk.DoubleVar(value=float(config.DEFAULT_SIMULATED_STACK_DELAY_US_MEAN))
        self.var_stack_jitter = tk.DoubleVar(value=float(config.DEFAULT_SIMULATED_STACK_DELAY_US_JITTER))
        self.var_pretrained = tk.BooleanVar(value=config.PRETRAINED_MODEL_PATH.exists())
        self.var_pqc = tk.BooleanVar(value=False)
        self.var_zta = tk.BooleanVar(value=True)

        row_n = ctk.CTkFrame(f_params, fg_color="transparent")
        row_n.pack(fill=tk.X, pady=4)
        ctk.CTkLabel(row_n, text="Normal traffic size", width=158, anchor="w").pack(side=tk.LEFT)
        ctk.CTkComboBox(
            row_n,
            variable=self.var_normal,
            values=[str(x) for x in config.NORMAL_MESSAGE_PRESETS],
            state="readonly",
            width=100,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        row_a = ctk.CTkFrame(f_params, fg_color="transparent")
        row_a.pack(fill=tk.X, pady=4)
        ctk.CTkLabel(row_a, text="Injected attack messages", width=158, anchor="w").pack(side=tk.LEFT)
        ctk.CTkComboBox(
            row_a,
            variable=self.var_attack,
            values=[str(x) for x in config.ATTACK_COUNT_PRESETS],
            state="readonly",
            width=100,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        row = ctk.CTkFrame(f_params, fg_color="transparent")
        row.pack(fill=tk.X, pady=4)
        ctk.CTkLabel(row, text="Attack style", width=158, anchor="w").pack(side=tk.LEFT)
        ctk.CTkComboBox(
            row,
            variable=self.var_attack_type,
            values=["injection", "replay", "fuzzing"],
            state="readonly",
            width=100,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        row_contam = ctk.CTkFrame(f_params, fg_color="transparent")
        row_contam.pack(fill=tk.X, pady=4)
        ctk.CTkLabel(row_contam, text="Detector sensitivity", width=158, anchor="w").pack(side=tk.LEFT)
        ctk.CTkComboBox(
            row_contam,
            variable=self.var_contamination,
            values=["auto", "0.01", "0.05", "0.10", "0.15"],
            state="readonly",
            width=100,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self._row_scale(f_params, "Network delay average (μs)", self.var_stack_mean, 5.0, 120.0)
        self._row_scale(f_params, "Delay variation (μs)", self.var_stack_jitter, 0.0, 40.0)

        pretrained_ok = config.PRETRAINED_MODEL_PATH.exists()
        cb_pre = ctk.CTkSwitch(
            f_params,
            text="Use pre-trained model file (skip training)",
            variable=self.var_pretrained,
            state=tk.NORMAL if pretrained_ok else tk.DISABLED,
        )
        cb_pre.pack(anchor="w", padx=8, pady=(12, 4))
        if not pretrained_ok:
            ctk.CTkLabel(
                f_params,
                text=f"(Add a model via scripts/build_pretrained_model.py → {config.PRETRAINED_MODEL_PATH.name})",
                font=("Helvetica", 10),
                text_color="gray",
            ).pack(anchor="w", padx=24)

        ctk.CTkSwitch(
            f_params,
            text="Stronger future-proof signing (when available)",
            variable=self.var_pqc,
        ).pack(anchor="w", padx=8, pady=4)

        ctk.CTkSwitch(
            f_params,
            text="Verify every step (zero-trust style)",
            variable=self.var_zta,
        ).pack(anchor="w", padx=8, pady=4)
        ctk.CTkLabel(
            f_params,
            text="Insights uses the on-device report (no cloud required for the baseline).",
            font=("Helvetica", 11),
            text_color="gray",
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self.btn_run = ctk.CTkButton(
            left,
            text="Run protection demo",
            command=self._on_run,
            height=46,
            font=("Helvetica", 15, "bold"),
            corner_radius=12,
        )
        self.btn_run.pack(fill=tk.X, padx=12, pady=16)

        ctk.CTkLabel(left, textvariable=self._status, wraplength=280, text_color="gray").pack(anchor="w", padx=12, pady=4)

        paned.add(left, minsize=220)

        # ── Center: notebook ──
        center = ctk.CTkFrame(paned)

        self.nb = ctk.CTkTabview(center)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Names must match exactly for CTkTabview.set() (CustomTkinter has no .select())
        self._TAB_SUMMARY = "Summary"
        self._TAB_CHARTS = "Charts"
        self._TAB_INCIDENTS = "Incidents"
        self._TAB_PATH = "Path"
        self._TAB_INSIGHTS = "Insights"

        self.tab_overview = self.nb.add(self._TAB_SUMMARY)
        self.tab_charts = self.nb.add(self._TAB_CHARTS)
        self.tab_incidents = self.nb.add(self._TAB_INCIDENTS)
        self.tab_threat = self.nb.add(self._TAB_PATH)
        self.tab_insights = self.nb.add(self._TAB_INSIGHTS)

        # Same monospace size as Path / Incidents tables (12pt), not the old 10pt ScrolledText
        self._overview_text = ctk.CTkTextbox(
            self.tab_overview,
            font=("Menlo", 12),
            text_color="#e6edf3",
            corner_radius=10,
        )
        self._overview_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Charts: matplotlib figure in a frame (optional)
        self._fig = None
        self._canvas = None
        if HAS_MPL and Figure is not None and FigureCanvasTkAgg is not None:
            self._fig = Figure(figsize=(10, 9), dpi=100, facecolor="#0e1117")
            self._fig.subplots_adjust(
                left=0.08, right=0.98, top=0.95, bottom=0.06, hspace=0.4, wspace=0.28
            )
            self._canvas = FigureCanvasTkAgg(self._fig, master=self.tab_charts)
            self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            if NavigationToolbar2Tk is not None:
                self._toolbar = NavigationToolbar2Tk(self._canvas, self.tab_charts)
                self._toolbar.update()
        else:
            install_msg = (
                "Charts need matplotlib.\n\n"
                "Install:\n  python3 -m pip install matplotlib\n\n"
                "Or everything:\n  python3 -m pip install -r ../requirements.txt"
            )
            ctk.CTkLabel(
                self.tab_charts,
                text=install_msg,
                justify=tk.LEFT,
                wraplength=720,
            ).pack(anchor="nw", padx=12, pady=12)

        # Style the Treeview to blend into CustomTkinter dark mode!
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", 
                        background="#2b2b2b",
                        foreground="white",
                        rowheight=25,
                        fieldbackground="#2b2b2b",
                        font=("Helvetica", 11),
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        style.configure("Treeview.Heading",
                        background="#565b5e",
                        foreground="white",
                        font=("Helvetica", 11, "bold"),
                        relief="flat")
        style.map("Treeview.Heading", background=[('active', '#3484F0')])

        cols = ("id", "can", "ecu", "attack", "action", "conf", "alg")
        self._tree = ttk.Treeview(self.tab_incidents, columns=cols, show="headings", height=18)
        for c, w, t in zip(
            cols,
            (90, 70, 140, 220, 90, 70, 100),
            ("ID", "CAN ID", "ECU", "Attack", "Action", "Conf.", "Algorithm"),
        ):
            self._tree.heading(c, text=t)
            self._tree.column(c, width=w)
        vsb = ttk.Scrollbar(self.tab_incidents, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._threat_text = ctk.CTkTextbox(
            self.tab_threat, font=("Menlo", 12)
        )
        self._threat_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ctk.CTkLabel(
            self.tab_insights,
            text="Insights",
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            self.tab_insights,
            text="On-device explanation report. Ollama Assistant + quick prompts are in the right column.",
            text_color="gray70",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkLabel(self.tab_insights, text="Current readout:", text_color="gray70").pack(anchor="w", padx=8)
        self._insights_text = _pack_styled_llm_text(
            self.tab_insights, bg="#1e1e1e", padx=8, pady=4
        )

        paned.add(center, minsize=320)

        # ── Right: assistant chat ──
        right_chat = ctk.CTkFrame(paned, corner_radius=12)

        ctk.CTkLabel(
            right_chat,
            text="Assistant",
            font=("Helvetica", 15, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(
            right_chat,
            text="Chat-style readout — larger type, clear sections (after you run the pipeline).",
            font=("Helvetica", 11),
            text_color="gray70",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self._chat_history = _pack_styled_llm_text(
            right_chat, bg="#1a1a1a", padx=12, pady=4
        )

        ctk.CTkLabel(
            right_chat,
            text="Quick prompts — tap to ask (works from any tab)",
            font=("Helvetica", 11, "bold"),
            text_color="gray70",
        ).pack(anchor="w", padx=12, pady=(2, 4))
        sug_wrap = ctk.CTkFrame(right_chat, fg_color="transparent")
        sug_wrap.pack(fill=tk.X, padx=12, pady=(0, 6))
        sug_grid = ctk.CTkFrame(sug_wrap, fg_color="transparent")
        sug_grid.pack(fill=tk.X)
        for col in (0, 1):
            sug_grid.grid_columnconfigure(col, weight=1, uniform="sug")
        for idx, (short, full_q) in enumerate(CHAT_SUGGESTIONS):
            r, c = divmod(idx, 2)
            ctk.CTkButton(
                sug_grid,
                text=short,
                font=("Helvetica", 11),
                height=32,
                corner_radius=10,
                fg_color="#243044",
                hover_color="#31425a",
                anchor="w",
                command=lambda q=full_q: self._apply_suggested_question(q),
            ).grid(row=r, column=c, padx=3, pady=3, sticky="ew")

        input_frame = ctk.CTkFrame(right_chat, fg_color="transparent")
        input_frame.pack(fill=tk.X, padx=12, pady=(4, 12))
        self._chat_input = ctk.CTkEntry(
            input_frame,
            font=("Helvetica", 14),
            placeholder_text="Ask about metrics, incidents, or charts…",
        )
        self._chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._chat_input.bind("<Return>", lambda e: self._on_chat_send())
        self.btn_send = ctk.CTkButton(input_frame, text="Send", command=self._on_chat_send, width=56, font=("Helvetica", 13))
        self.btn_send.pack(side=tk.RIGHT)

        paned.add(right_chat, minsize=200)

        # Prefer extra width for the main tabs + AI column (sidebar stays narrow)
        try:
            paned.paneconfigure(left, stretch="never")
            paned.paneconfigure(center, stretch="always")
            paned.paneconfigure(right_chat, stretch="always")
        except tk.TclError:
            pass

        # Sash positions default to even splits; drag sashes to resize (tk on some macOS builds
        # has no PanedWindow.sashpos in the Python wrapper — do not call it here).

        self._ctx: dict | None = None
        self.llm_engine = LLMInsightEngine()

    def _row_scale(
        self,
        parent,
        label: str,
        variable: tk.Variable,
        vmin: float,
        vmax: float,
    ) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, pady=4)
        ctk.CTkLabel(row, text=label, width=158, anchor="w").pack(side=tk.LEFT)
        lbl_val = ctk.CTkLabel(row, width=40, anchor="e")
        lbl_val.pack(side=tk.RIGHT)

        is_float = isinstance(variable, tk.DoubleVar)

        def refresh(_: str | float | None = None) -> None:
            v = variable.get()
            if is_float:
                lbl_val.configure(text=f"{float(v):.2f}")
            else:
                lbl_val.configure(text=str(int(round(v))))

        sc = ctk.CTkSlider(
            row,
            from_=vmin,
            to=vmax,
            variable=variable,
            command=lambda _: refresh(),
        )
        sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        refresh()

    def _on_run(self) -> None:
        self.btn_run.configure(state=tk.DISABLED)
        self._status.set("Running pipeline…")
        self.root.update_idletasks()

        def work() -> None:
            try:
                ctx = _run_pipeline(
                    normal_count=int(self.var_normal.get().strip()),
                    attack_count=int(self.var_attack.get().strip()),
                    attack_type=self.var_attack_type.get(),
                    contamination="auto" if self.var_contamination.get() == "auto" else float(self.var_contamination.get()),
                    stack_mean=float(self.var_stack_mean.get()),
                    stack_jitter=float(self.var_stack_jitter.get()),
                    use_pretrained=bool(self.var_pretrained.get()),
                    prefer_pqc=bool(self.var_pqc.get()),
                    use_zta=bool(self.var_zta.get()),
                )
                self.root.after(0, lambda: self._apply_results(ctx, None))
            except Exception as e:
                self.root.after(0, lambda: self._apply_results(None, e))

        threading.Thread(target=work, daemon=True).start()

    def _apply_results(self, ctx: dict | None, err: Exception | None) -> None:
        self.btn_run.configure(state=tk.NORMAL)
        if err is not None:
            self._status.set("Error.")
            messagebox.showerror("Pipeline error", str(err))
            return

        assert ctx is not None
        self._ctx = ctx
        self._status.set("Pipeline complete.")

        m = ctx["metrics"]
        ss = ctx["safety_summary"]
        ms = ctx["mit_summary"]
        results = ctx["results"]

        # Overview text
        ae = m.get("avg_edge_processing_latency_us") or 0
        ass = m.get("avg_simulated_stack_latency_us") or 0
        overview = f"""BEFORE (no protection)
  Brake ECU protection: NONE
  Intrusion detection: NONE
  Attack success rate: 100%

AFTER (CAN-Guard AI)
  Detection rate:        {m['detection_rate']:.2%}
  False positive rate:   {m['false_positive_rate']:.2%}
  Accuracy:              {m['accuracy']:.2%}
  Precision / Recall / F1: {m['precision']:.2%} / {m['recall']:.2%} / {m['f1_score']:.2%}
  Avg total path latency: {m['avg_detection_latency_us']:.1f} μs
  Avg edge (IF) latency:  {ae:.1f} μs
  Avg sim. stack latency: {ass:.1f} μs
  Batch wall time:       {m.get('wall_clock_batch_detection_s', 0):.4f} s

Safety layer
  Allowed: {ss['allowed']}  Alerts: {ss['alerts']}  Blocked: {ss['blocked']}
  Safe mode activations: {ss['safe_mode_activations']}

Mitigation
  Incidents: {ms['total_incidents']}  Alerts: {ms['total_alerts']}
  Signing: {ms.get('signing_algorithm', 'n/a')}
"""
        overview += format_prevented_threats_summary(ctx["mitigation"])
        self._overview_text.configure(state="normal")
        self._overview_text.delete("1.0", "end")
        self._overview_text.insert("1.0", overview)
        self._overview_text.configure(state="disabled")

        # Charts (3×2) — requires matplotlib
        if HAS_MPL and self._fig is not None and self._canvas is not None:
            self._fig.clf()
            ax1 = self._fig.add_subplot(3, 2, 1)
            ax2 = self._fig.add_subplot(3, 2, 2)
            ax3 = self._fig.add_subplot(3, 2, 3)
            ax4 = self._fig.add_subplot(3, 2, 4)
            ax5 = self._fig.add_subplot(3, 2, 5)
            ax6 = self._fig.add_subplot(3, 2, 6)

            # Time-sorted rows → index 0..N-1 matches message order in the test batch
            rp = results.reset_index(drop=True)
            normal_scores = rp.loc[rp["is_malicious"] == 0, "anomaly_score"]
            attack_scores = rp.loc[rp["is_malicious"] == 1, "anomaly_score"]

            raw = rp["anomaly_score"].values.astype(float)
            s_min = float(np.nanmin(raw))
            s_max = float(np.nanmax(raw))
            if not np.isfinite(s_min) or not np.isfinite(s_max):
                s_min, s_max = 0.0, 1.0
            if s_max - s_min < 1e-12:
                s_min -= 0.5
                s_max += 0.5
            # Identical bin edges for both histograms (comparable counts per bin)
            bin_edges = np.linspace(s_min, s_max, 31)

            # sklearn IsolationForest: lower decision_function → more anomalous
            score_xlabel = "IF decision_function (↓ more anomalous, ↑ more inlier-like)"

            if len(normal_scores) > 0:
                ax1.hist(normal_scores, bins=bin_edges, color="#00cc96", alpha=0.75, label="Normal")
            else:
                ax1.text(0.5, 0.5, "No normal samples", ha="center", va="center", transform=ax1.transAxes, color="#888")
            ax1.set_title("Ground-truth normal — score distribution")
            ax1.set_xlabel(score_xlabel, fontsize=8)
            ax1.set_xlim(s_min, s_max)

            if len(attack_scores) > 0:
                ax2.hist(attack_scores, bins=bin_edges, color="#ff4b4b", alpha=0.75, label="Malicious")
            else:
                ax2.text(0.5, 0.5, "No malicious samples", ha="center", va="center", transform=ax2.transAxes, color="#888")
            ax2.set_title("Ground-truth malicious — score distribution")
            ax2.set_xlabel(score_xlabel, fontsize=8)
            ax2.set_xlim(s_min, s_max)

            nm = rp["is_malicious"] == 0
            mk = rp["is_malicious"] == 1
            ax3.scatter(rp.index[nm], rp.loc[nm, "anomaly_score"], s=4, c="#00cc96", alpha=0.5, label="Normal")
            ax3.scatter(rp.index[mk], rp.loc[mk, "anomaly_score"], s=16, c="#ff4b4b", marker="x", label="Malicious")
            ax3.set_title("Score vs message order")
            ax3.set_xlabel("Message index (time-sorted batch)")
            ax3.set_ylabel("IF score")
            ax3.legend(loc="upper right", fontsize=8)

            id_n = rp.loc[rp["is_malicious"] == 0, "can_id_hex"].value_counts()
            id_m = rp.loc[rp["is_malicious"] == 1, "can_id_hex"].value_counts()
            keys = sorted(set(id_n.index) | set(id_m.index), key=str)
            x = np.arange(len(keys), dtype=float)
            bw = 0.35
            ax4.bar(x - bw / 2, [id_n.get(k, 0) for k in keys], width=bw, label="Normal", color="#00cc96")
            ax4.bar(x + bw / 2, [id_m.get(k, 0) for k in keys], width=bw, label="Malicious", color="#ff4b4b")
            ax4.set_xticks(x)
            ax4.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
            ax4.set_title("CAN ID message counts (normal vs malicious subset)")
            ax4.set_ylabel("Count")
            ax4.legend(fontsize=8)

            action_counts: dict[str, int] = {}
            for d in ctx["decisions"]:
                action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1

            color_map = {"allow": "#00cc96", "alert": "#ffd700", "block": "#ff4b4b", "safe_mode": "#ff6b00"}
            pie_labels = sorted(action_counts.keys(), key=str)
            pie_values = [action_counts[k] for k in pie_labels]
            pie_colors = [color_map.get(str(lbl).lower(), "#888888") for lbl in pie_labels]

            ax5.pie(
                pie_values,
                labels=pie_labels,
                autopct="%1.0f%%",
                colors=pie_colors,
                textprops={"fontsize": 8},
            )
            ax5.set_title("Safety decisions (all messages)")

            cm = np.array(
                [
                    [m["true_negatives"], m["false_positives"]],
                    [m["false_negatives"], m["true_positives"]],
                ],
                dtype=float,
            )
            vmax_cm = max(1.0, float(np.max(cm)))
            ax6.imshow(cm, cmap="RdBu_r", vmin=0.0, vmax=vmax_cm)
            ax6.set_xticks([0, 1])
            ax6.set_yticks([0, 1])
            ax6.set_xticklabels(["Pred. normal", "Pred. anomaly"])
            ax6.set_yticklabels(["Actual normal", "Actual malicious"])
            for (i, j), val in [((0, 0), cm[0][0]), ((0, 1), cm[0][1]), ((1, 0), cm[1][0]), ((1, 1), cm[1][1])]:
                ax6.text(j, i, str(int(val)), ha="center", va="center", color="white", fontsize=11)
            ax6.set_title("Confusion matrix")

            for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                ax.set_facecolor("#161b22")
                ax.tick_params(colors="#c9d1d9", labelsize=8)
                ax.title.set_color("#e6edf3")
            self._fig.suptitle("Traffic & safety analysis", color="#e6edf3", fontsize=11)
            self._fig.tight_layout(pad=1.5)
            self._canvas.draw()

        # Incidents tree
        for i in self._tree.get_children():
            self._tree.delete(i)
        mitigation = ctx["mitigation"]
        for inc in mitigation.incidents[: config.INCIDENT_TABLE_MAX_ROWS]:
            self._tree.insert(
                "",
                tk.END,
                values=(
                    inc.incident_id,
                    inc.can_id,
                    inc.ecu_name,
                    inc.attack_type,
                    inc.action_taken,
                    f"{inc.confidence:.0%}",
                    getattr(inc, "signature_algorithm", "") or ms.get("signing_algorithm", ""),
                ),
            )

        # Threat Path LLM Generator
        threat_json_str = json.dumps(ctx["threat_logger"].to_json_ready(), indent=2)
        self._threat_text.delete("1.0", tk.END)
        self._threat_text.insert("1.0", "Analyzing lateral movement using Local LLM Engine...\n")
        
        def fetch_threat_path():
            threat_txt = self.llm_engine.generate_threat_path(threat_json_str)
            self.root.after(0, lambda: [self._threat_text.delete("1.0", tk.END), self._threat_text.insert("1.0", threat_txt)])
            
        threading.Thread(target=fetch_threat_path, daemon=True).start()

        report = build_explanation_report(
            metrics=m,
            results_df=results,
            safety_summary=ss,
            mit_summary=ms,
            mitigation=mitigation,
            decisions=ctx["decisions"],
        )
        threat_json_str = json.dumps(ctx["threat_logger"].to_json_ready(), indent=2)
        self._last_report = report + "\n### Threat Path Sequence Logs\n" + threat_json_str
        
        self._insights_text.configure(state=tk.NORMAL)
        self._insights_text.delete("1.0", tk.END)
        self._insights_text.insert(tk.END, "Analyzing threat telemetry…", "meta")
        self._insights_text.insert(tk.END, "\n", "meta")
        self._insights_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()
        
        def fetch_insight():
            insight = self.llm_engine.generate_insight(report)
            self.root.after(0, lambda: self._update_insight_ui(insight))
            
        threading.Thread(target=fetch_insight, daemon=True).start()

        self.nb.set(self._TAB_SUMMARY)

    def _update_insight_ui(self, insight_text: str):
        self._insights_text.configure(state=tk.NORMAL)
        self._insights_text.delete("1.0", tk.END)
        _insert_formatted_llm(self._insights_text, insight_text)
        self._insights_text.configure(state=tk.DISABLED)

    def _apply_suggested_question(self, question: str) -> None:
        """Insert a curated prompt and send — same flow as typing in the box + Send."""
        self._chat_input.delete(0, tk.END)
        self._chat_input.insert(0, question)
        self._on_chat_send()

    def _on_chat_send(self):
        user_msg = self._chat_input.get().strip()
        if not user_msg:
            return
            
        self._chat_input.delete(0, tk.END)
        self._chat_history.configure(state=tk.NORMAL)
        self._chat_history.insert(tk.END, "\n", "body")
        self._chat_history.insert(tk.END, "You\n", "chat_you")
        self._chat_history.insert(tk.END, user_msg + "\n\n", "chat_user_msg")
        self._chat_history.mark_set("pending_reply", tk.INSERT)
        self._chat_history.insert(tk.END, "Assistant\n", "chat_assistant_label")
        self._chat_history.insert(tk.END, "Thinking…\n", "meta")
        self._chat_history.see(tk.END)
        self._chat_history.configure(state=tk.DISABLED)
        
        def process_chat():
            if not getattr(self, '_ctx', None):
                reply = "Please run the pipeline first to generate telemetry data."
            else:
                m = self._ctx["metrics"]
                ss = self._ctx["safety_summary"]
                ms = self._ctx["mit_summary"]
                distilled_context = build_distilled_dashboard_context(m, ss, ms, self._ctx["mitigation"])
                reply = self.llm_engine.chat(distilled_context, user_msg)
                    
            self.root.after(0, lambda: self._update_chat(reply))
            
        threading.Thread(target=process_chat, daemon=True).start()
        
    def _update_chat(self, reply: str):
        self._chat_history.configure(state=tk.NORMAL)
        try:
            self._chat_history.delete("pending_reply", tk.END)
        except tk.TclError:
            pass
        self._chat_history.insert(tk.END, "Assistant\n", "chat_assistant_label")
        _insert_formatted_llm(self._chat_history, reply)
        self._chat_history.insert(tk.END, "\n", "body")
        try:
            self._chat_history.mark_unset("pending_reply")
        except tk.TclError:
            pass
        self._chat_history.see(tk.END)
        self._chat_history.configure(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    CanGuardTkApp().run()
