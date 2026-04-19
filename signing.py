"""
Pluggable incident signing: HMAC-SHA3-256 (default) or optional liboqs PQC (ML-DSA / Dilithium).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from abc import ABC, abstractmethod
from typing import Any


class SignatureProvider(ABC):
    @abstractmethod
    def sign(self, canonical_payload: str) -> str:
        ...

    @abstractmethod
    def verify(self, canonical_payload: str, signature: str) -> bool:
        ...

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        ...


class HMAC_SHA3_256_Provider(SignatureProvider):
    """Default: symmetric authentication (development / integration tests)."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key if key is not None else os.urandom(32)

    def sign(self, canonical_payload: str) -> str:
        return hmac.new(
            self._key,
            canonical_payload.encode("utf-8"),
            hashlib.sha3_256,
        ).hexdigest()

    def verify(self, canonical_payload: str, signature: str) -> bool:
        expected = self.sign(canonical_payload)
        return hmac.compare_digest(expected, signature)

    @property
    def algorithm_name(self) -> str:
        return "HMAC-SHA3-256"


class LibOQSProvider(SignatureProvider):
    """
    Post-quantum signatures via liboqs-python (ML-DSA / Dilithium) when installed.
    Picks first enabled mechanism from a preferred list.
    """

    def __init__(self, mechanism: str | None = None) -> None:
        import oqs  # type: ignore

        self._oqs = oqs
        enabled = list(oqs.get_enabled_sig_mechanisms())
        if mechanism is None:
            for candidate in ("ML-DSA-65", "ML-DSA-44", "Dilithium3", "Dilithium2"):
                if candidate in enabled:
                    mechanism = candidate
                    break
            if mechanism is None and enabled:
                mechanism = enabled[0]
        if mechanism is None:
            raise RuntimeError("No liboqs signature mechanisms enabled")
        self._mechanism = mechanism
        self._signer = oqs.Signature(mechanism)
        self._public_key = self._signer.generate_keypair()

    def sign(self, canonical_payload: str) -> str:
        msg = canonical_payload.encode("utf-8")
        raw = self._signer.sign(msg)
        return raw.hex()

    def verify(self, canonical_payload: str, signature: str) -> bool:
        msg = canonical_payload.encode("utf-8")
        try:
            sig_bytes = bytes.fromhex(signature)
        except ValueError:
            return False
        verifier = self._oqs.Signature(self._mechanism)
        return verifier.verify(msg, sig_bytes, self._public_key)

    @property
    def algorithm_name(self) -> str:
        return self._mechanism


def build_signature_provider(prefer_pqc: bool = False) -> SignatureProvider:
    """
    prefer_pqc: try liboqs first; on failure use HMAC.
    """
    if prefer_pqc:
        try:
            return LibOQSProvider()
        except Exception:
            pass
    return HMAC_SHA3_256_Provider()


def canonical_incident_json(incident_dict: dict[str, Any]) -> str:
    """Stable serialization for signing (signature field excluded by caller)."""
    return json.dumps(incident_dict, sort_keys=True, default=str)
