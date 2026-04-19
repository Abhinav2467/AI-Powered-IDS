"""
CAN-Guard AI — central configuration (paths, simulated delays, modes).
"""

from pathlib import Path

# Repository root (directory containing this file)
ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "models"
PRETRAINED_MODEL_PATH = MODELS_DIR / "pretrained.joblib"
MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
ONNX_MODEL_PATH = MODELS_DIR / "edge_model.onnx"

# Gateway / CAN stack delay modeled as extra latency before edge inference (microseconds).
# Typical order-of-magnitude: CAN frame + driver + gateway queue (tunable for demo).
DEFAULT_SIMULATED_STACK_DELAY_US_MEAN = 48.0
DEFAULT_SIMULATED_STACK_DELAY_US_JITTER = 12.0

# SocketCAN capture defaults
DEFAULT_SOCKETCAN_DURATION_S = 3.0
DEFAULT_SOCKETCAN_MAX_FRAMES = 5000

# UI / demo presets (normal training messages; attack injection counts)
NORMAL_MESSAGE_PRESETS: tuple[int, ...] = (1000, 2000, 5000)
ATTACK_COUNT_PRESETS: tuple[int, ...] = (50, 100, 200, 500, 1000)

# Max rows shown in UI incident tables (Treeview / dataframe)
INCIDENT_TABLE_MAX_ROWS = 1000
