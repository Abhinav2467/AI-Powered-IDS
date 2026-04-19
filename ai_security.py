"""
AI Security Layer for CAN-Guard
================================
Encrypts all AI agent data at rest using AES-128-CBC via Fernet (from the
`cryptography` library). Three distinct encrypted stores are maintained:

  1. ai_secure_store.enc     — AI interaction log (received prompts + generated responses)
  2. ai_metadata_store.enc   — AI agent metadata (model config, endpoint URLs, runtime stats)
  3. ai_session_store.enc    — Per-session audit record (init time, call counts, model identity)

Key is auto-generated on first run (or loaded from ai_crypto.key if it exists).
In a production/HSM environment, rotate this key and store it in a secure vault.
"""

from __future__ import annotations

import os
import json
import sys
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from cryptography.fernet import Fernet


# ─── File paths ──────────────────────────────────────────────────────────────

_KEY_FILE          = "ai_crypto.key"
_LOG_FILE          = "ai_secure_store.enc"        # prompts + responses
_META_FILE         = "ai_metadata_store.enc"       # agent config metadata
_SESSION_FILE      = "ai_session_store.enc"        # per-session audit


# ─── Core cipher helper ──────────────────────────────────────────────────────

def _load_or_generate_key() -> bytes:
    """
    Load the Fernet key from disk, or generate + persist a new one.
    KEY ROTATION: delete ai_crypto.key and restart to rotate. Existing
    encrypted files become unreadable without the original key.
    """
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE, "rb") as f:
            return f.read().strip()
    key = Fernet.generate_key()
    with open(_KEY_FILE, "wb") as f:
        f.write(key)
    print(f"[AI-SEC] New encryption key generated → {_KEY_FILE}  (keep this safe!)")
    return key


def _get_fernet() -> Fernet:
    """Returns a ready-to-use Fernet cipher using the project key."""
    return Fernet(_load_or_generate_key())


def _append_encrypted(store_path: str, record: Dict[str, Any], fernet: Fernet) -> None:
    """Serialize ``record`` to JSON, encrypt it, and append one line to ``store_path``."""
    raw = json.dumps(record, default=str).encode("utf-8")
    cipher_line = fernet.encrypt(raw) + b"\n"
    with open(store_path, "a+b") as f:
        f.write(cipher_line)


def _read_encrypted(store_path: str, fernet: Fernet) -> List[Dict[str, Any]]:
    """Read and decrypt every record from ``store_path``."""
    if not os.path.exists(store_path):
        return []
    records: List[Dict[str, Any]] = []
    with open(store_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = fernet.decrypt(line)
                records.append(json.loads(payload.decode("utf-8")))
            except Exception as exc:
                print(f"[AI-SEC] Could not decrypt record in {store_path}: {exc}")
    return records


# ─── Public logger class ─────────────────────────────────────────────────────

class SecureAILogger:
    """
    Drop-in secure audit wrapper for LLMInsightEngine.

    Usage (inside insight_engine._call_ollama):
        SecureAILogger().encrypt_and_log(
            user_prompt=user_safe,
            ai_response=reply,
            metadata={...}           # any agent metadata dict you want to protect
        )
    """

    def __init__(self) -> None:
        self.fernet = _get_fernet()

    # ── Interaction log ──────────────────────────────────────────────────────

    def encrypt_and_log(
        self,
        user_prompt: str,
        ai_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Encrypt and persist:
          • ``user_prompt``  — the sanitised CAN telemetry / user question
          • ``ai_response``  — the LLM's generated reply
          • ``metadata``     — AI agent metadata (model name, endpoint, call index, etc.)

        All three fields are written to their dedicated encrypted stores.
        """
        now = datetime.now(timezone.utc).isoformat()

        # 1. Interaction record
        if user_prompt or ai_response:
            interaction: Dict[str, Any] = {
                "event":          "llm_call",
                "utc_timestamp":  now,
                "received_data":  user_prompt,
                "generated_data": ai_response,
            }
            _append_encrypted(_LOG_FILE, interaction, self.fernet)

        # 2. Metadata record (model config, endpoint, runtime identifiers, etc.)
        if metadata:
            meta_record: Dict[str, Any] = {
                "event":         "agent_metadata",
                "utc_timestamp": now,
                **metadata,
            }
            _append_encrypted(_META_FILE, meta_record, self.fernet)

    # ── Session snapshot ─────────────────────────────────────────────────────

    def log_session_init(
        self,
        model_name: str,
        base_url: str,
        http_timeout: float,
        installed_models: List[str],
    ) -> None:
        """
        Encrypt and store the full AI agent session initialisation details.
        Call this once when LLMInsightEngine.__init__ completes.
        """
        record: Dict[str, Any] = {
            "event":              "session_init",
            "utc_timestamp":      datetime.now(timezone.utc).isoformat(),
            "model_name":         model_name,
            "base_url":           base_url,
            "http_timeout_s":     http_timeout,
            "installed_models":   installed_models,
            "python_version":     sys.version,
            "platform":           platform.platform(),
        }
        _append_encrypted(_SESSION_FILE, record, self.fernet)

    def log_session_stat(
        self,
        model_name: str,
        base_url: str,
        total_calls: int,
        online: bool,
    ) -> None:
        """
        Encrypt and store a runtime telemetry snapshot of the AI agent's state.
        Call this periodically (e.g. when _ensure_online runs) to capture drift.
        """
        record: Dict[str, Any] = {
            "event":        "session_stat",
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name":   model_name,
            "base_url":     base_url,
            "total_calls":  total_calls,
            "is_online":    online,
        }
        _append_encrypted(_SESSION_FILE, record, self.fernet)

    # ── Audit reader (admin/debug only) ──────────────────────────────────────

    def read_interaction_logs(self) -> List[Dict[str, Any]]:
        """Decrypt and return all interaction records (prompts + responses)."""
        return _read_encrypted(_LOG_FILE, self.fernet)

    def read_metadata_logs(self) -> List[Dict[str, Any]]:
        """Decrypt and return all agent metadata records."""
        return _read_encrypted(_META_FILE, self.fernet)

    def read_session_logs(self) -> List[Dict[str, Any]]:
        """Decrypt and return all session init/stat records."""
        return _read_encrypted(_SESSION_FILE, self.fernet)

    def print_audit_summary(self) -> None:
        """Pretty-print a summary of all encrypted stores to stdout (dev/debug)."""
        interactions = self.read_interaction_logs()
        metadata     = self.read_metadata_logs()
        sessions     = self.read_session_logs()
        print("\n" + "=" * 60)
        print("  AI SECURITY AUDIT SUMMARY")
        print("=" * 60)
        print(f"  Interaction records  : {len(interactions)}")
        print(f"  Metadata  records    : {len(metadata)}")
        print(f"  Session   records    : {len(sessions)}")
        if sessions:
            latest = sessions[-1]
            print(f"  Last model           : {latest.get('model_name', 'n/a')}")
            print(f"  Last endpoint        : {latest.get('base_url',   'n/a')}")
        print("=" * 60 + "\n")


# ─── Standalone smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = SecureAILogger()

    # Simulate a session init
    logger.log_session_init(
        model_name="llama3.2:1b",
        base_url="http://127.0.0.1:11434",
        http_timeout=2.5,
        installed_models=["llama3.2:1b", "llama3:8b"],
    )

    # Simulate an LLM call
    logger.encrypt_and_log(
        user_prompt="Detection Rate: 95%. Attack: Injection on CAN ID 0x200.",
        ai_response="The system detected a high-confidence brake ECU injection attack.",
        metadata={
            "model_name":     "llama3.2:1b",
            "base_url":       "http://127.0.0.1:11434",
            "call_index":     1,
            "timeout_s":      120.0,
            "prompt_chars":   53,
            "response_chars": 73,
        },
    )

    # Simulate a session stat update
    logger.log_session_stat(
        model_name="llama3.2:1b",
        base_url="http://127.0.0.1:11434",
        total_calls=1,
        online=True,
    )

    logger.print_audit_summary()

    # Verify ciphertext is unreadable
    print("--- Ciphertext sample from ai_secure_store.enc ---")
    with open(_LOG_FILE, "rb") as fh:
        raw_line = fh.readline().strip()
    print(raw_line[:80], "...")
    assert b"CAN" not in raw_line, "Plain text leaked into ciphertext!"
    assert b"0x200" not in raw_line, "CAN ID leaked into ciphertext!"
    print("[OK] No plain text visible in ciphertext.\n")
