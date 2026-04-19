"""
Local LLM integration via Ollama HTTP API.
Used by the desktop dashboard and Streamlit demo.

Connection (default http://127.0.0.1:11434):
  OLLAMA_BASE_URL   — full base URL, e.g. http://192.168.1.10:11434
  OLLAMA_HOST       — host:port or http(s)://host:port (same idea as the Ollama CLI)
  OLLAMA_HTTP_TIMEOUT — seconds for health checks (default 2.5)
  OLLAMA_MODEL      — exact model tag for /api/chat. If unset, the first match from a built-in preference
                      list is used against `ollama list`; if none match, the first installed model is used.
                      If Ollama is empty, defaults to llama3.2:1b (pull when ready).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from assistant_prompts import sanitize_can_data_for_prompt
from ai_security import SecureAILogger

_DEFAULT_LOCAL = "http://127.0.0.1:11434"


def resolve_ollama_base_url() -> str:
    """
    Base URL without trailing slash. Prefer OLLAMA_BASE_URL, then OLLAMA_HOST, then local default.
    """
    explicit = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        if host.startswith(("http://", "https://")):
            return host.rstrip("/")
        return f"http://{host}".rstrip("/")
    return _DEFAULT_LOCAL


class LLMInsightEngine:
    """
    Local LLM via Ollama HTTP API.
    Connectivity is re-checked before each call; on macOS we try to open the Ollama app once if offline
    (only when the configured URL is on this machine).
    """

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        self.base_url = (base_url or resolve_ollama_base_url()).rstrip("/")
        self.endpoint = f"{self.base_url}/api/chat"
        self._http_timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "2.5"))
        self._launch_attempted = False
        self._call_count = 0  # tracks total LLM calls for metadata encryption
        self.model_name = self._resolve_model_name(model_name)
        self.is_online = self._check_ollama()

        # Encrypt and persist agent session metadata securely at startup
        try:
            _installed = self._list_installed_model_names()
            SecureAILogger().log_session_init(
                model_name=self.model_name,
                base_url=self.base_url,
                http_timeout=self._http_timeout,
                installed_models=_installed,
            )
        except Exception as _sec_e:
            print(f"[AI-SEC] Could not encrypt session metadata: {_sec_e}")

    def is_server_reachable(self) -> bool:
        """Whether the Ollama HTTP API responds (no launch attempt)."""
        return self._check_ollama()

    def _tags_urls(self) -> list[str]:
        """GET /api/tags endpoints to try (localhost / 127.0.0.1 / IPv6 aliases on default port)."""
        out: list[str] = []
        seen: set[str] = set()

        def add(u: str) -> None:
            if u not in seen:
                seen.add(u)
                out.append(u)

        add(f"{self.base_url}/api/tags")
        if self.base_url in (_DEFAULT_LOCAL, "http://localhost:11434", "http://[::1]:11434"):
            add("http://127.0.0.1:11434/api/tags")
            add("http://localhost:11434/api/tags")
            add("http://[::1]:11434/api/tags")
        return out

    def _is_probably_local(self) -> bool:
        u = self.base_url.lower()
        return u.startswith("http://127.0.0.1:") or u.startswith("http://localhost:")

    def _check_ollama(self) -> bool:
        """True if the Ollama HTTP API responds."""
        for url in self._tags_urls():
            # Dedupe while preserving order
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self._http_timeout) as response:
                    if response.status == 200:
                        return True
            except Exception:
                continue
        return False

    def _try_launch_ollama(self) -> None:
        """Best-effort: start Ollama GUI on macOS, or `ollama serve` if available."""
        if not self._is_probably_local():
            return
        if self._launch_attempted:
            return
        self._launch_attempted = True
        try:
            if sys.platform == "darwin":
                subprocess.Popen(
                    ["open", "-a", "Ollama"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(4.0)
            elif shutil.which("ollama"):
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(2.0)
        except Exception:
            pass

    def _ensure_online(self) -> bool:
        """Refresh status; optionally try to start Ollama once, then re-check."""
        self.is_online = self._check_ollama()
        if self.is_online:
            return True
        self._try_launch_ollama()
        self.is_online = self._check_ollama()
        return self.is_online

    _MODEL_PREFERENCE: tuple[str, ...] = (
        "llama3.2:1b",
        "llama3.2:latest",
        "llama3.2",
        "llama3:latest",
        "llama3",
        "mistral:latest",
        "mistral",
        "phi3:latest",
        "phi3",
        "tinyllama:latest",
        "tinyllama",
        "gemma2:2b",
        "gemma2:latest",
    )

    def _resolve_model_name(self, explicit: str | None) -> str:
        """
        Prefer `explicit`, then OLLAMA_MODEL, only if that tag exists locally (when we can list models).
        Otherwise pick from _MODEL_PREFERENCE, else the first `ollama list` name, else llama3.2:1b.
        """
        installed = self._list_installed_model_names()
        inst_set = set(installed)

        def pick_auto() -> str:
            if not installed:
                return "llama3.2:1b"
            for candidate in self._MODEL_PREFERENCE:
                if candidate in inst_set:
                    return candidate
            return installed[0]

        if explicit and str(explicit).strip():
            want = str(explicit).strip()
            if not installed:
                return want
            if want in inst_set:
                return want
            return pick_auto()

        env = os.environ.get("OLLAMA_MODEL", "").strip()
        if env:
            if not installed:
                return env
            if env in inst_set:
                return env
            return pick_auto()

        return pick_auto()

    @staticmethod
    def _parse_models_from_tags_payload(data: Any) -> list[str]:
        """Model tags from /api/tags JSON (`name` and/or `model` keys)."""
        if not isinstance(data, dict):
            return []
        models = data.get("models") or []
        names: list[str] = []
        for m in models:
            if not isinstance(m, dict):
                continue
            raw = m.get("name") or m.get("model") or ""
            tag = str(raw).strip()
            if tag:
                names.append(tag)
        return names

    def _list_installed_model_names_http(self) -> list[str]:
        """First successful GET /api/tags; updates self.base_url when a URL works."""
        for url in self._tags_urls():
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self._http_timeout) as response:
                    if response.status != 200:
                        continue
                    data = json.loads(response.read().decode("utf-8"))
                names = self._parse_models_from_tags_payload(data)
                if url.endswith("/api/tags"):
                    base = url[: -len("/api/tags")].rstrip("/")
                    self.base_url = base
                    self.endpoint = f"{self.base_url}/api/chat"
                return names
            except Exception:
                continue
        return []

    def _list_installed_model_names_cli(self) -> list[str]:
        """Parse `ollama list` when HTTP tags are empty or unreliable."""
        exe = shutil.which("ollama")
        if not exe:
            return []
        try:
            kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": 20,
            }
            if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            out = subprocess.run([exe, "list"], **kwargs)
        except Exception:
            return []
        if out.returncode != 0 or not out.stdout:
            return []
        lines = out.stdout.strip().splitlines()
        names: list[str] = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names

    def _list_installed_model_names(self) -> list[str]:
        """
        Merge GET /api/tags with `ollama list` so we still see models if HTTP parsing fails
        (proxies, IPv6, or API quirks).
        """
        http_names = self._list_installed_model_names_http()
        cli_names = self._list_installed_model_names_cli()
        merged: list[str] = []
        seen: set[str] = set()
        for n in http_names + cli_names:
            if n and n not in seen:
                seen.add(n)
                merged.append(n)
        return merged

    def _repick_model_if_missing(self) -> None:
        """If self.model_name is not in the local library, switch to a valid tag."""
        installed = self._list_installed_model_names()
        if not installed:
            return
        if self.model_name in installed:
            return
        inst_set = set(installed)
        for candidate in self._MODEL_PREFERENCE:
            if candidate in inst_set:
                self.model_name = candidate
                return
        self.model_name = installed[0]

    def _format_model_missing_message(self, http_body: str) -> str | None:
        """If Ollama says the model is missing, return a helpful message; else None."""
        raw = (http_body or "").lower()
        if "not found" not in raw:
            return None
        installed = self._list_installed_model_names()
        lines = [
            f"⚠️ Model `{self.model_name}` is not installed in Ollama.",
            "",
            f"Pull it: **`ollama pull {self.model_name}`**",
            "",
            "Or set **`OLLAMA_MODEL`** to a model you already have (see list below).",
        ]
        if installed:
            show = installed[:16]
            lines.extend(["", "**Installed models:** " + ", ".join(show) + (" …" if len(installed) > 16 else "")])
        else:
            lines.extend(["", "Run `ollama list` in a terminal to see local models."])
        return "\n".join(lines)

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float = 120.0,
        *,
        _retry_on_404: bool = True,
    ) -> str | None:
        if not self._ensure_online():
            return None
        self._repick_model_if_missing()

        # User/telemetry content only — system prompt stays fixed instructions from this codebase.
        user_safe = sanitize_can_data_for_prompt(user_prompt)

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_safe},
            ],
            "stream": False,
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as f:
                response = json.loads(f.read().decode("utf-8"))
                reply = response.get("message", {}).get("content", "")
                self._call_count += 1

                # Secure Logging: Encrypt prompt, response AND agent metadata at rest
                try:
                    SecureAILogger().encrypt_and_log(
                        user_prompt=user_safe,
                        ai_response=reply,
                        metadata={
                            "model_name":     self.model_name,
                            "base_url":       self.base_url,
                            "endpoint":       self.endpoint,
                            "call_index":     self._call_count,
                            "timeout_s":      timeout_s,
                            "http_timeout_s": self._http_timeout,
                            "prompt_chars":   len(user_safe),
                            "response_chars": len(reply),
                            "is_online":      self.is_online,
                        },
                    )
                except Exception as log_e:
                    print(f"[AI-SEC] Warning: Failed to securely encrypt AI logs: {log_e}")

                return reply
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code == 404 and _retry_on_404:
                prev = self.model_name
                self._repick_model_if_missing()
                if self.model_name != prev:
                    return self._call_ollama(
                        system_prompt,
                        user_prompt,
                        timeout_s,
                        _retry_on_404=False,
                    )
            if e.code == 404:
                friendly = self._format_model_missing_message(body)
                if friendly:
                    return friendly
            return f"⚠️ Ollama HTTP {e.code}: {body[:500]}"
        except urllib.error.URLError as e:
            return (
                f"⚠️ Cannot reach Ollama at {self.base_url}: {e}\n\n"
                f"Check that the server is running (`ollama serve` or the Ollama app). "
                f"If it uses another host/port, set **OLLAMA_BASE_URL** (e.g. `http://127.0.0.1:11434`)."
            )
        except Exception as e:
            return f"⚠️ Unexpected AI error: {e}"

    def _stream_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float = 120.0,
    ):
        """
        Generator that yields text tokens as they arrive from the Ollama /api/chat
        streaming endpoint. Designed for use with st.write_stream().
        Falls back to a single error string if the model is offline.
        """
        if not self._ensure_online():
            yield (
                f"⚠️ Ollama offline at `{self.base_url}`. "
                f"Run `ollama serve` then `ollama pull {self.model_name}`."
            )
            return

        self._repick_model_if_missing()
        user_safe = sanitize_can_data_for_prompt(user_prompt)

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_safe},
            ],
            "stream": True,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        full_reply: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_reply.append(token)
                        yield token
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            yield f"\n\n⚠️ Connection error: {exc}"
            return
        except Exception as exc:
            yield f"\n\n⚠️ Unexpected error: {exc}"
            return

        # Encrypt full assembled reply at rest
        assembled = "".join(full_reply)
        self._call_count += 1
        try:
            SecureAILogger().encrypt_and_log(
                user_prompt=user_safe,
                ai_response=assembled,
                metadata={
                    "model_name": self.model_name,
                    "base_url":   self.base_url,
                    "call_index": self._call_count,
                    "stream":     True,
                },
            )
        except Exception as log_e:
            print(f"[AI-SEC] Warning: {log_e}")

    def generate_insight(self, report_context: str) -> str:
        if not self._ensure_online():
            return (
                f"⚠️ Local AI offline (Ollama not reachable at {self.base_url}).\n\n"
                "Start the **Ollama** app or `ollama serve`, confirm `curl " + self.base_url + "/api/tags` works, "
                "then run: `ollama pull " + self.model_name + "`\n"
                "If Ollama runs on another machine/port, set **OLLAMA_BASE_URL** before starting the dashboard.\n\n"
                "--- SIMULATED EDGE AI INSIGHT ---\n"
                "Based on the provided metrics, the Edge AI Isolation Forest has detected anomalous patterns "
                "typical of injection and fuzzing attacks. The detection rate indicates separation between "
                "normal CAN bus operations and malicious actuation commands mapped to the Braking ECU (0x200). "
                "By tracking stack delay jitter and IF scores, we intercept suspicious lateral movement from the "
                "gateway to the CAN bus.\n\n"
                "Safety-first design: the system aims for fail-operational behavior by filtering anomalous packets "
                "rather than a full shutdown. Charts show distinct clusters of anomalous behavior, allowing "
                "mitigation with signed incident reports when risk thresholds are exceeded."
            )

        system_prompt = (
            "You are an expert automotive cybersecurity analyst running locally on an ECU gateway. "
            "Explain the Isolation Forest results for CAN injection/fuzzing. "
            "Cover: detection quality, false positives, safety posture (fail-safe vs fail-operational), and how "
            "the charts (score histograms, timeline, CAN ID bars, decision pie, confusion matrix) support the story. "
            "Be concise but complete. "
            "Format for on-screen reading: short paragraphs; use optional ## section headings; use bullet lines "
            "starting with '- ' for lists; use **bold** only for key terms (detection rate, ECU names, CAN IDs)."
        )
        user_prompt = (
            f"Edge AI report:\n{report_context}\n\n"
            "Analyze for the security team: risk posture, anomalies, and how to read the dashboard graphs."
        )
        reply = self._call_ollama(system_prompt, user_prompt)
        return reply if reply else "⚠️ Error generating insight."

    def stream_insight(self, metrics: dict, safety_summary: dict, mit_summary: dict):
        """
        Streaming generator for the AI narrative brief.
        Sends a SHORT, token-efficient prompt so the 1B model responds quickly.
        Use with st.write_stream().
        """
        system_prompt = (
            "You are a concise automotive cybersecurity analyst. "
            "Summarise the CAN-Guard IDS run below in 4-6 bullet points. "
            "Mention: detection rate, false positives, blocked ECUs, safety mode triggers, and signing. "
            "Use **bold** for key numbers. Be brief — 120 words max."
        )
        user_prompt = (
            f"Detection rate: {metrics.get('detection_rate',0):.1%}, "
            f"FPR: {metrics.get('false_positive_rate',0):.1%}, "
            f"Accuracy: {metrics.get('accuracy',0):.1%}, "
            f"F1: {metrics.get('f1_score',0):.1%}. "
            f"Blocked: {safety_summary.get('blocked',0)}, "
            f"Safe mode: {safety_summary.get('safe_mode_activations',0)}, "
            f"Incidents signed: {mit_summary.get('total_incidents',0)} "
            f"({mit_summary.get('signing_algorithm','HMAC-SHA3-256')}). "
            "Write the brief now."
        )
        yield from self._stream_ollama(system_prompt, user_prompt, timeout_s=90.0)

    def chat(self, context_report: str, user_msg: str) -> str:
        if not self._ensure_online():
            if "count" in user_msg.lower() or "how many" in user_msg.lower():
                import re

                m = re.search(r"Total incidents reported / attacks blocked:\s*(\d+)", context_report)
                if m:
                    return (
                        f"[Offline AI] About {m.group(1)} incidents appear in the current run. "
                        f"Start Ollama at {self.base_url} for full answers."
                    )
                return "[Offline AI] See the Mitigation and Safety summaries for counts."
            if "graph" in user_msg.lower():
                return "[Offline AI] Charts summarize anomaly scores, timeline, CAN IDs, decisions, and confusion matrix. Start Ollama for detailed explanations."
            if "insight" in user_msg.lower() or "explain" in user_msg.lower():
                return "[Offline AI] Start the Ollama app and pull a model, then ask again for natural-language answers."
            return "[Offline AI] Ollama not reachable. Open the Ollama app or run `ollama serve`, then `ollama pull " + self.model_name + "`."

        system_prompt = (
            "You are an AI analyst helping the user understand the CAN-bus security dashboard. "
            "Use ONLY the provided context (metrics, incident logs, graph summaries). "
            "For incident questions, use AGGREGATED INCIDENT LOGS and explain specific attacks when asked. "
            "For graph questions, use GRAPHS SUMMARY and relate to the metrics. "
            "Do NOT write code. Do NOT invent IP/TCP/DNS — this is in-vehicle CAN. "
            "Answer clearly and completely. "
            "Format like a clear chat answer: optional ## headings, '- ' bullets for lists, short paragraphs, "
            "and **bold** for important numbers or ECU/CAN identifiers only."
        )
        user_prompt = f"Dashboard Context:\n{context_report}\n\nUser question: {user_msg}"
        reply = self._call_ollama(system_prompt, user_prompt, timeout_s=180.0)
        return reply if reply else "Error generating chat."

    def stream_chat(self, metrics: dict, safety_summary: dict, mit_summary: dict, user_msg: str):
        """
        Streaming generator for the AI chat assistant.
        Sends compact context so the 1B model stays fast.
        Use with st.write_stream().
        """
        system_prompt = (
            "You are a brief automotive cybersecurity assistant. "
            "Answer using only the metrics provided. No code. No IP/TCP. "
            "Use **bold** for numbers. Max 80 words."
        )
        context_snippet = (
            f"Detection rate {metrics.get('detection_rate',0):.1%}, "
            f"FPR {metrics.get('false_positive_rate',0):.1%}, "
            f"Accuracy {metrics.get('accuracy',0):.1%}. "
            f"Blocked {safety_summary.get('blocked',0)}, "
            f"Safe mode {safety_summary.get('safe_mode_activations',0)}, "
            f"Incidents {mit_summary.get('total_incidents',0)}."
        )
        user_prompt = f"Context: {context_snippet}\n\nQuestion: {user_msg}"
        yield from self._stream_ollama(system_prompt, user_prompt, timeout_s=90.0)

    def generate_threat_path(self, threat_json: str) -> str:
        if not self._ensure_online():
            return f"⚠️ Offline — raw threat JSON:\n{threat_json}"

        system_prompt = (
            "You are an automotive cybersecurity analyst. Read the JSON log of CAN bus events. "
            "Write 3–5 sentences explaining the attack path: infotainment → gateway → target CAN IDs/ECUs. "
            "Do NOT invent IP addresses, TCP/UDP, or DNS. No code. Stick to CAN IDs and ECUs from the JSON."
        )
        user_prompt = f"Threat JSON:\n{threat_json}"
        reply = self._call_ollama(system_prompt, user_prompt, timeout_s=90.0)
        return reply if reply else "Error analyzing threat path."


def build_distilled_dashboard_context(
    metrics: dict[str, Any],
    safety_summary: dict[str, Any],
    mit_summary: dict[str, Any],
    mitigation: Any,
) -> str:
    """
    Compact telemetry + incident text for local LLM chat (same structure as the desktop dashboard).
    """
    m = metrics
    ss = safety_summary
    ms = mit_summary

    incidents_log: list[str] = []
    current_group: str | None = None
    start_i = 1

    for i, inc in enumerate(mitigation.incidents):
        action_str = getattr(inc.action_taken, "value", str(inc.action_taken))
        desc = f"{inc.attack_type} on {inc.ecu_name} (CAN {inc.can_id}). Action taken: {action_str}"

        if current_group is None:
            current_group = desc
            start_i = i + 1
        elif desc != current_group:
            if start_i == i:
                incidents_log.append(f"Attack {start_i}: {current_group}")
            else:
                incidents_log.append(f"Attacks {start_i} to {i}: {current_group}")
            current_group = desc
            start_i = i + 1

    if current_group is not None:
        i = len(mitigation.incidents)
        if start_i == i:
            incidents_log.append(f"Attack {start_i}: {current_group}")
        else:
            incidents_log.append(f"Attacks {start_i} to {i}: {current_group}")

    inc_str = "\n".join(incidents_log)

    inc_index_lines: list[str] = []
    for i, inc in enumerate(mitigation.incidents[:200]):
        action_str = getattr(inc.action_taken, "value", str(inc.action_taken))
        inc_index_lines.append(
            f"{i + 1}. id={inc.incident_id} CAN={inc.can_id} ECU={inc.ecu_name} "
            f"attack={inc.attack_type} action={action_str} conf={inc.confidence:.0%}"
        )
    inc_index = "\n".join(inc_index_lines)

    graphs_str = (
        "Graphs 1–2: Histograms of sklearn Isolation Forest decision_function on the SAME bin edges; "
        "lower scores = more anomalous, higher = more inlier-like (not necessarily ‘negative’ on the axis).\n"
        "Graph 3: Each point is one CAN message in time-sorted order (x = index); y = IF score.\n"
        "Graph 4: Grouped counts of messages per CAN ID in normal-only vs malicious-only subsets.\n"
        "Graph 5: Share of safety actions on all messages (allow / alert / block / safe_mode).\n"
        "Graph 6: Confusion matrix for ML detection vs ground-truth labels (TN, FP, FN, TP)."
    )

    return (
        "[CAN BUS TELEMETRY DATA]\n"
        f"- Total network messages processed: {ss.get('total_messages', 0)}\n"
        f"- Total attacks attempted (injected): {int(m.get('true_positives', 0) + m.get('false_negatives', 0))}\n"
        f"- Total incidents reported / attacks blocked: {ms.get('total_incidents', 0)}\n"
        f"- Detection Accuracy: {m.get('accuracy', 0):.1%}\n"
        f"- Average Inference Latency: {m.get('avg_edge_processing_latency_us', 0):.1f} us\n\n"
        "[GRAPHS SUMMARY — explain these when the user asks about charts]\n"
        f"{graphs_str}\n\n"
        "[INCIDENT INDEX — numbered list; use for questions about specific incidents or ‘all incidents’]\n"
        f"{inc_index}\n\n"
        "[AGGREGATED INCIDENT LOGS — grouped consecutive duplicates]\n"
        f"{inc_str}\n\n"
        "CRITICAL RULES:\n"
        "1. This is an internal vehicle CAN bus. There are NO IP addresses, TCP, or DNS logs.\n"
        "2. When asked to explain all incidents, summarize using INCIDENT INDEX and highlight patterns; mention safety actions (allow/block/safe_mode).\n"
        "3. For graph questions, tie each chart (1–6) to what the user sees in the dashboard."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CAN-GUARD AI: Insight Engine (Ollama)")
    print(f"Resolved API base: {resolve_ollama_base_url()}")
    print("=" * 60)
    engine = LLMInsightEngine()
    test_context = "Detection Rate: 95.0%. False Positive Rate: 2.1%. Simulated attack: Injection on CAN ID 0x200."
    print("Insight:")
    print(engine.generate_insight(test_context))
    print("=" * 60)
