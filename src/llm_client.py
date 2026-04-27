"""Thin LLM client — mock mode for tests, API modes for real runs.

Supports:
  LLM_MODE=mock   — canned responses for tests (no API needed)
  LLM_MODE=api    — OpenAI-compatible endpoint
  LLM_MODE=hf_api — HuggingFace Inference API
"""

from __future__ import annotations

import os
import re
import time
from collections import deque

from . import config

# ── Per-minute rate limiter ───────────────────────────────────────────
# Global to track call timestamps across all API calls in the process.
# Thread-safe only in single-threaded use (async/threaded runs need lock).
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

# Realistic multi-claim table — exercises filter, relabel, and gate paths
_MOCK_VERIFIER_TABLE = (
    "1. SUPPORTED | The meeting is at 3pm | span_0\n"
    "2. SUPPORTED | The meeting is in Conference Room B | span_0\n"
    "3. NOT_IN_EVIDENCE | The meeting duration is not specified | none\n"
    "4. UNSUPPORTED | Some details could not be verified | none"
)

_MOCK_BASELINE = "This is a mock baseline answer."


def _mock_response(system: str, user: str) -> str:
    """Return a canned response based on prompt keywords.

    Returns realistic multi-claim output so mock-mode pipeline runs
    exercise the full filter → relabel → inference → gate chain.
    """
    lower = (system + user).lower()
    if "verify" in lower or "verifier" in lower or ("extract" in lower and "label" in lower):
        return _MOCK_VERIFIER_TABLE
    if "draft" in lower or "answer the" in lower or "answer only" in lower:
        return _MOCK_DRAFT
    return _MOCK_BASELINE


# ── OpenAI-compatible API mode ─────────────────────────────────────────

def _api_call(system: str, user: str) -> str:
    _enforce_rate_limit()
    time.sleep(config.LLM_CALL_DELAY)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Install it or set LLM_MODE=mock."
        ) from exc

    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=config.MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ── HuggingFace Inference API mode ────────────────────────────────────

_hf_client = None


def _hf_api_call(system: str, user: str) -> str:
    global _hf_client
    _enforce_rate_limit()
    time.sleep(config.LLM_CALL_DELAY)  # per-call delay

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. Install it or set LLM_MODE=mock."
        ) from exc

    if _hf_client is None:
        api_key = os.getenv("HF_API_KEY", config.OPENAI_API_KEY)
        _hf_client = InferenceClient(api_key=api_key)

    model = os.getenv("MODEL_NAME", config.MODEL_NAME)

    for attempt in range(3):
        try:
            resp = _hf_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
            content = resp.choices[0].message.content or ""
            # Strip 思考... tags (Qwen3)
            content = re.sub(r" 思考.*?结束思考", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            err = str(e)
            if "402" in err or "429" in err:
                wait = (attempt + 1) * 10
                print(f"  [retry] Rate limited, waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("HF API call failed after 3 retries")


# ── Public interface ───────────────────────────────────────────────────

def llm_call(system: str, user: str) -> str:
    """Send a system+user prompt pair to the configured LLM backend."""
    mode = os.getenv("LLM_MODE", config.LLM_MODE)
    if mode == "mock":
        return _mock_response(system, user)
    if mode == "hf_api":
        return _hf_api_call(system, user)
    return _api_call(system, user)
