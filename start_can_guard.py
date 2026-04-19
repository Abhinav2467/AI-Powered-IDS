#!/usr/bin/env python3
"""
CAN-Guard — start screen.
Run:  python3 start_can_guard.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

try:
    import customtkinter as ctk
except ImportError as e:
    print("Install CustomTkinter:  python3 -m pip install customtkinter")
    raise SystemExit(1) from e

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _spawn(cmd: list[str]) -> None:
    kwargs: dict = {"cwd": _ROOT, "start_new_session": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen(cmd, **kwargs)


def _open_desktop() -> None:
    script = os.path.join(_ROOT, "04_main_integration.py")
    _spawn([sys.executable, script])


def _open_web() -> None:
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.headless", "true"],
        cwd=_ROOT,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _browse() -> None:
        time.sleep(2.5)
        webbrowser.open("http://localhost:8501")

    threading.Thread(target=_browse, daemon=True).start()


def _run_quick_demo() -> None:
    _spawn([sys.executable, "main.py"])


def _show_help() -> None:
    import tkinter as tk
    from tkinter import messagebox

    body = (
        "Desktop — full dashboard with charts and chat.\n"
        "Web — same demo in your browser (Streamlit).\n"
        "Quick demo — runs the full pipeline in a new window.\n\n"
        "After the desktop app opens, use the Run button there."
    )
    messagebox.showinfo("CAN-Guard — quick guide", body)


def main() -> None:
    ctk.set_appearance_mode("dark")
    try:
        ctk.set_default_color_theme("dark-blue")
    except Exception:
        ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("CAN-Guard")
    app.geometry("520x400")
    app.minsize(480, 360)

    shell = ctk.CTkFrame(app, corner_radius=16, border_width=0)
    shell.pack(fill="both", expand=True, padx=20, pady=20)

    header = ctk.CTkFrame(shell, fg_color="transparent")
    header.pack(fill="x", pady=(0, 16))
    ctk.CTkLabel(header, text="CAN-Guard", font=("Helvetica", 22, "bold")).pack(anchor="w")
    ctk.CTkLabel(
        header,
        text="Choose how to open the demo.",
        font=("Helvetica", 13),
        text_color="gray70",
    ).pack(anchor="w")

    grid = ctk.CTkFrame(shell, fg_color="transparent")
    grid.pack(fill="both", expand=True)
    for c in (0, 1):
        grid.grid_columnconfigure(c, weight=1)
    for r in (0, 1):
        grid.grid_rowconfigure(r, weight=1)

    def tile(row: int, col: int, title: str, subtitle: str, cmd) -> None:
        box = ctk.CTkFrame(grid, corner_radius=14, border_width=0)
        box.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        btn = ctk.CTkButton(
            box,
            text=title,
            font=("Helvetica", 15, "bold"),
            height=44,
            corner_radius=12,
            command=cmd,
        )
        btn.pack(fill="x", padx=12, pady=(16, 8))
        ctk.CTkLabel(box, text=subtitle, font=("Helvetica", 11), text_color="gray60", wraplength=200).pack(
            padx=12, pady=(0, 16)
        )

    tile(0, 0, "Desktop app", "Charts, safety view, and assistant panel.", _open_desktop)
    tile(0, 1, "Web browser", "Same flow in the browser (Streamlit).", _open_web)
    tile(1, 0, "Quick demo", "Runs the full pipeline in a new window.", _run_quick_demo)
    tile(1, 1, "Help", "Short guide in a popup.", _show_help)

    ctk.CTkLabel(
        shell,
        text="You can close this window — other apps keep running.",
        font=("Helvetica", 11),
        text_color="gray50",
    ).pack(pady=(8, 0))

    app.mainloop()


if __name__ == "__main__":
    main()
