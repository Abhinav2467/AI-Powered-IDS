"""
Gateway ECU simulator — adds modeled stack/queue delay before edge AI sees each frame.

Does not modify CAN semantics; augments DataFrame columns for timing analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_gateway_path_delay(
    df: pd.DataFrame,
    mean_us: float,
    jitter_us: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Per-row simulated gateway + CAN stack latency (microseconds), i.i.d. Gaussian-ish.

    Columns added:
      simulated_stack_latency_us
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    jitter = rng.normal(0, jitter_us / 2, size=n) if jitter_us > 0 else 0
    out["simulated_stack_latency_us"] = np.clip(mean_us + jitter, 1.0, None)
    return out
