#!/usr/bin/env python3
"""
Run from repo root: delegates to can-guard-project/04_main_integration.py
with cwd set to that folder (required for relative assets and imports).
"""
from __future__ import annotations

import os
import runpy


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    sub = os.path.join(here, "can-guard-project")
    target = os.path.join(sub, "04_main_integration.py")
    if not os.path.isfile(target):
        raise SystemExit(f"Missing: {target}")
    os.chdir(sub)
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
