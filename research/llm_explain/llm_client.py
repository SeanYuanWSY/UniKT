"""LLM client for the explanation pipeline (OpenAI-compatible, archived).

Providers: glm (glm-5.3-flash) and kimi (kimi-for-coding). All calls are
archived to JSONL (request, response, model version string, tokens, retries).
Retries cover schema/parse failures only -- faithfulness violations are
recorded from the first attempt (design v2).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request

DEFAULT_KEYS_FILE = "/root/.llm-keys.env"

PROVIDERS: dict[str, dict[str, str]] = {
    "glm": {"model": "glm-5.3-flash", "env_base": "GLM_BASE_URL", "env_key": "GLM_API_KEY"},
    "kimi": {"model": "kimi-for-coding", "env_base": "KIMI_BASE_URL", "env_key": "KIMI_API_KEY"},
}

ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"


def load_keys(keys_file: str = DEFAULT_KEYS_FILE) -> dict[str, str]:
    """Load API endpoints/keys from the env file (never committed)."""
    env: dict[str, str] = {}
    with open(keys_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


@dataclass
class LLMCall:
    """One archived LLM call (including every retry attempt)."""

    provider: str
    model: str
    purpose: str
    user_id: int
    condition: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    parsed: dict[str, Any] | None = None
    parse_ok: bool = False


def _chat_once(
    base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if "glm" in model:
        payload["temperature"] = temperature
        # glm-5.x are reasoning models; disable thinking so max_tokens covers
        # the JSON answer itself.
        payload["thinking"] = {"type": "disabled"}
    else:
        # kimi-for-coding only accepts temperature=1 (reasoning-only model);
        # omit it entirely rather than send 0 (400 Bad Request otherwise).
        pass
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from a reply (fences stripped)."""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(text[start : i + 1])
                    return out if isinstance(out, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def call_llm(
    provider: str,
    messages: list[dict],
    *,
    purpose: str,
    user_id: int,
    condition: str,
    keys_file: str = DEFAULT_KEYS_FILE,
    max_tokens: int = 4096,
    max_retries: int = 2,
    temperature: float = 0.0,
    archive: bool = True,
) -> LLMCall:
    """Call a provider, parse JSON, retry parse failures, archive everything."""
    env = load_keys(keys_file)
    spec = PROVIDERS[provider]
    base, key = env[spec["env_base"]], env[spec["env_key"]]
    model = spec["model"]

    call = LLMCall(provider=provider, model=model, purpose=purpose, user_id=user_id, condition=condition)

    for attempt in range(1 + max_retries):
        try:
            raw = _chat_once(base, key, model, messages, max_tokens, temperature)
        except Exception as e:  # noqa: BLE001 - archive and retry any transport error
            call.attempts.append({"attempt": attempt, "error": repr(e)})
            time.sleep(2)
            continue
        finish = raw.get("choices", [{}])[0].get("finish_reason", "?")
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        call.attempts.append(
            {
                "attempt": attempt,
                "finish_reason": finish,
                "content": text,
                "reasoning_head": (raw["choices"][0]["message"].get("reasoning_content") or "")[:200],
                "usage": raw.get("usage", {}),
                "model_returned": raw.get("model", ""),
                "created": raw.get("created", 0),
                "id": raw.get("id", ""),
            }
        )
        if finish != "stop":
            time.sleep(1)
            continue  # truncated (kimi reasoning burns budget): retry with same budget
        parsed = extract_json(text)
        if parsed is not None:
            call.final_text, call.parsed, call.parse_ok = text, parsed, True
            break
        time.sleep(1)

    if archive:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = ARCHIVE_DIR / f"{provider}_{time.strftime('%Y%m%d')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(call.__dict__, ensure_ascii=False) + "\n")
    return call
