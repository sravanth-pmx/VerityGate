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
LLM_MODE: str = os.getenv("LLM_MODE", "mock")  # "mock", "api", or "hf_api"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_CALL_DELAY: float = float(os.getenv("LLM_CALL_DELAY", "2.0"))  # Seconds between HF API calls
LLM_MAX_CALLS_PER_MINUTE: int = int(os.getenv("LLM_MAX_CALLS_PER_MINUTE", "30"))  # Per-minute rate limit (0=unlimited)

# ── Output paths ──────────────────────────────────────────────────────
BASELINE_NORMAL_PATH = RESULTS_DIR / "baseline_normal.jsonl"
BASELINE_HONESTY_PATH = RESULTS_DIR / "baseline_honesty.jsonl"
PIPELINE_RESULTS_PATH = RESULTS_DIR / "verity_pipeline.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"
