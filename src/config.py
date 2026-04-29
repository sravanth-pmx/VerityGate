"""Configuration for Project Verity-H v0.1."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
GOLD_CASES_PATH = DATA_DIR / "gold_cases.jsonl"

# ── Dataset version ─────────────────────────────────────────────────────
DATASET_VERSION: str = "dev_v0.4"  # current gold_cases.jsonl is a DEV set

# ── LLM settings ──────────────────────────────────────────────────────
LLM_MODE: str = os.getenv("LLM_MODE", "mock")  # "mock", "api", "hf_api", "groq", or "nvidia"

# OpenAI-compatible (LLM_MODE=api, or Ollama local)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Groq (LLM_MODE=groq)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

# NVIDIA NIM / NVIDIA API Catalog (OpenAI-compatible)
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Shared model settings (all API modes)
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_CALL_DELAY: float = float(os.getenv("LLM_CALL_DELAY", "2.0"))
LLM_MAX_CALLS_PER_MINUTE: int = int(os.getenv("LLM_MAX_CALLS_PER_MINUTE", "30"))  # 0 = unlimited

# ── Debug / tracing ───────────────────────────────────────────────────
VERITY_TRACE_LLM: bool = os.getenv("VERITY_TRACE_LLM", "0") == "1"
VERITY_TRACE_FULL_PROMPTS: bool = os.getenv("VERITY_TRACE_FULL_PROMPTS", "0") == "1"

# Approx token accounting. Exact token usage depends on provider/model.
ENABLE_TOKEN_ESTIMATES: bool = os.getenv("ENABLE_TOKEN_ESTIMATES", "1") == "1"

# ── Output paths ──────────────────────────────────────────────────────
BASELINE_NORMAL_PATH = RESULTS_DIR / "baseline_normal.jsonl"
BASELINE_HONESTY_PATH = RESULTS_DIR / "baseline_honesty.jsonl"
PIPELINE_RESULTS_PATH = RESULTS_DIR / "verity_pipeline.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"
