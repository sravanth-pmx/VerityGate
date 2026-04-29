"""Thin LLM client — mock mode for tests, API modes for real runs.

Supports:
  LLM_MODE=mock   — canned responses for tests (no API needed)
  LLM_MODE=api    — OpenAI-compatible endpoint (also works for Ollama)
  LLM_MODE=groq   — Groq API (fast, free tier, OpenAI-compatible)
  LLM_MODE=hf_api — HuggingFace Inference API
  LLM_MODE=nvidia
"""

from __future__ import annotations

import os
import re
import time
from collections import deque

from . import config

# ── Per-minute rate limiter ───────────────────────────────────────────
_CALL_TIMES: deque[float] = deque()


def _enforce_rate_limit() -> None:
    """Sleep if necessary to stay under LLM_MAX_CALLS_PER_MINUTE."""
    max_per_min = config.LLM_MAX_CALLS_PER_MINUTE
    if max_per_min <= 0:
        return

    now = time.time()
    # Remove timestamps older than 60s
    while _CALL_TIMES and _CALL_TIMES[0] < now - 60:
        _CALL_TIMES.popleft()

    if len(_CALL_TIMES) >= max_per_min:
        # Wait until the oldest call falls out of the 60-second window
        wait = _CALL_TIMES[0] + 60 - now
        if wait > 0:
            print(f"  [rate limit] waiting {wait:.1f}s to stay under {max_per_min}/min")
            time.sleep(wait)

    _CALL_TIMES.append(time.time())


# ── Mock responses ─────────────────────────────────────────────────────

_MOCK_DRAFT = (
    "Based on the provided evidence, the meeting is at 3pm in Conference Room B. "
    "The duration has not been confirmed. Some details could not be verified."
)

_MOCK_VERIFIER_TABLE = (
    "1. SUPPORTED | The meeting is at 3pm | span_0\n"
    "2. SUPPORTED | The meeting is in Conference Room B | span_0\n"
    "3. NOT_IN_EVIDENCE | The meeting duration is not specified | none\n"
    "4. UNSUPPORTED | Some details could not be verified | none"
)

_MOCK_BASELINE = "This is a mock baseline answer."


def _mock_response(system: str, user: str) -> str:
    lower = (system + user).lower()
    if "verify" in lower or "verifier" in lower or ("extract" in lower and "label" in lower):
        return _MOCK_VERIFIER_TABLE
    if "draft" in lower or "answer the" in lower or "answer only" in lower:
        return _MOCK_DRAFT
    return _MOCK_BASELINE

def _clean_model_output(content: str) -> str:
    """Normalize common model/provider artifacts."""
    if not content:
        return ""

    # Strip common reasoning tags.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"思考.*?结束思考", "", content, flags=re.DOTALL)

    # Normalize badly rendered dash characters only lightly.
    # Do not remove replacement chars globally because they may signal encoding issues.
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    return content.strip()

def _should_use_streaming(mode: str) -> bool:
    """NVIDIA large models can be more reliable with streaming."""
    if os.getenv("LLM_STREAM", "0") == "1":
        return True
    if mode == "nvidia" and os.getenv("NVIDIA_STREAM", "1") == "1":
        return True
    return False

# ── OpenAI/Groq API mode ──────────────────────────────────────────────

def _api_call(system: str, user: str, mode: str = "api") -> str:
    _enforce_rate_limit()
    if config.LLM_CALL_DELAY > 0:
        time.sleep(config.LLM_CALL_DELAY)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package not installed.") from exc

    if mode == "groq":
        api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        base_url = config.GROQ_BASE_URL
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
    elif mode == "nvidia":
        api_key = config.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")
        base_url = config.NVIDIA_BASE_URL
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")
    else:
        api_key = config.OPENAI_API_KEY
        base_url = config.OPENAI_BASE_URL

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            kwargs = dict(
                model=config.MODEL_NAME,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

            if _should_use_streaming(mode):
                stream = client.chat.completions.create(**kwargs, stream=True)
                chunks: list[str] = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        chunks.append(chunk.choices[0].delta.content)
                return _clean_model_output("".join(chunks))

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return _clean_model_output(content)
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            print(
                f"[llm error] mode={mode} model={config.MODEL_NAME} "
                f"base_url={base_url} attempt={attempt + 1}/3 error={type(exc).__name__}: {exc}"
            )

            if any(x in msg for x in ("429", "rate limit", "timeout", "temporarily", "503", "502", "connection error")):
                time.sleep((attempt + 1) * 5)
                continue

            raise

    raise RuntimeError(f"API call failed after retries: {last_error}")

# ── HuggingFace Inference API mode ────────────────────────────────────

_hf_client = None

def _hf_api_call(system: str, user: str) -> str:
    global _hf_client
    _enforce_rate_limit()
    if config.LLM_CALL_DELAY > 0:
        time.sleep(config.LLM_CALL_DELAY)

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError("huggingface_hub not installed.") from exc

    if _hf_client is None:
        api_key = os.getenv("HF_API_KEY", config.OPENAI_API_KEY)
        _hf_client = InferenceClient(api_key=api_key)

    for attempt in range(3):
        try:
            resp = _hf_client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
            content = resp.choices[0].message.content or ""
            return _clean_model_output(content)
        except Exception as e:
            if "402" in str(e) or "429" in str(e):
                time.sleep((attempt + 1) * 10)
            else:
                raise
    raise RuntimeError("HF API call failed after 3 retries")


# ── Public interface ───────────────────────────────────────────────────

def llm_call(system: str, user: str) -> str:
    mode = os.getenv("LLM_MODE", config.LLM_MODE)
    if mode == "mock":
        return _mock_response(system, user)
    if mode == "hf_api":
        return _hf_api_call(system, user)
    if mode == "groq":
        return _api_call(system, user, mode="groq")
    if mode == "nvidia":
        return _api_call(system, user, mode="nvidia")
    return _api_call(system, user, mode="api")
    
