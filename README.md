# Project Verity-H v0.4

**Teaching AI to say "I don't know."**

When humans lack knowledge, they admit it — "I'm not sure", "I don't know", "let me check." LLMs don't. They fill gaps with plausible-sounding assumptions and present them as facts. Verity-H researches whether a lightweight verification pipeline can enforce honest behavior: **share what you know, flag what you don't, never silently guess.**

The system lets an LLM answer a question, then **verifies every claim against the provided evidence** before the user sees it. Supported claims pass through. Unsupported claims get flagged. Contradictions get caught. The user sees what's verified vs. what's a guess — like talking to an honest colleague.

> For the full architecture, research grounding, and design decisions, see **[DESIGN.md](DESIGN.md)**.

---

## Quick Start

```bash
# Clone
git clone https://huggingface.co/Sravanth18/verity-h-prototype
cd verity-h-prototype

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# Run tests (mock mode, no API key needed)
pytest
```

## Run Evaluation

```bash
# Set environment
export LLM_MODE=hf_api
export HF_API_KEY=your-key-here
export MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
export LLM_CALL_DELAY=2

# Baselines
python -m src.baseline_runner --mode normal --output results/baseline_normal.jsonl
python -m src.baseline_runner --mode honesty --output results/baseline_honesty.jsonl

# Pipeline
python -m src.pipeline_runner --output results/verity_pipeline_v0.4.jsonl

# Batched (resumable if interrupted)
python run_pipeline_batched.py --delay 0.5 --output results/verity_pipeline_v0.4.jsonl

# Report
python -m src.report --normal results/baseline_normal.jsonl \
                     --honesty results/baseline_honesty.jsonl \
                     --pipeline results/verity_pipeline_v0.4.jsonl \
                     --output results/report.md
```

## How It Works

```
Question + Evidence
       │
       ▼
  1. Split evidence into spans          (deterministic)
  2. Draft answer                        (LLM call #1)
  3. Extract + label claims              (LLM call #2)
  4. Post-process:                       (deterministic)
     • Filter junk/meta claims
     • Fix mislabeled claims via span matching
     • Detect inferential claims (4-tier)
     • Detect contradictions (status-pair only; numeric/date logged for audit)
  5. Gate decision                       (deterministic)
       │
       ▼
  Final answer with transparency metadata
```

**2 LLM calls per case.** Everything else is deterministic and auditable.

## Pipeline Decisions

| Situation | Decision | What user sees |
|-----------|----------|---------------|
| All claims verified | `accept` | Clean answer from verified claims |
| Some claims unverified | `partial` | "What I can verify" + "What I cannot verify" |
| Status-pair contradiction (open/closed, approved/rejected, etc.) | `contradiction` | Flags conflict, shows both sides |
| No evidence for the question | `needs_info` | "I don't have enough info" + what's needed |
| Speculative question (pressure=1) | `hypothesis` | Low-confidence guess with full caveats |
| Verifier failed to parse | `verifier_error` | Refuses to answer |

## Inference Detection (v0.3.1+)

The verifier catches claims the LLM wrongly marks as SUPPORTED:

| Tier | What it catches | Example |
|------|----------------|---------|
| 1. Epistemic hedges | "suggests", "consistent with", "most likely" | "Symptoms are *consistent with* bacterial infection" |
| 2. Logical leaps | "therefore", "based on these findings" | "*Therefore* the patient has strep throat" |
| 3. Deontic/normative | "should", "recommended", "indicated" | "Antibiotics *should be* started" |
| 4. Speculative questions | Question asks for judgment/prediction | "Should we invest?" → answer is inherently inferential |

Grounded in: CogniBench (arxiv:2505.20767), GME modality taxonomy (arxiv:2106.08037), BioScope corpus.

## Results (Qwen3-4B, 30 cases, v0.2.1)

| Metric | Baseline Normal | Baseline Honesty | Verity-H |
|--------|:-:|:-:|:-:|
| Unsupported claim rate (↓) | 10% | 0% | **0%** |
| Correct abstention (↑) | 70% | 100% | **100%** |
| Grounded accept (↑) | 0% | 0% | **100%** |
| Contradiction detection (↑) | 60% | 40% | **80%** |
| Pressure hypothesis (↑) | 0% | 0% | **100%** *(v0.2.1)* |
| False contradiction (↓) | 0% | 0% | **0%** |
| Partial coverage (↑) | 0% | 0% | **100%** |
| Latency p50 | 3,525ms | 3,244ms | **6,495ms** *(v0.3, 2-call batch)* |

See [RESULTS_ARCHIVE.md](RESULTS_ARCHIVE.md) for full version history.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODE` | `mock` | `mock` / `api` (OpenAI) / `hf_api` (HuggingFace) |
| `HF_API_KEY` | — | HuggingFace API key (for `hf_api` mode) |
| `OPENAI_API_KEY` | — | OpenAI API key (for `api` mode) |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Model to use |
| `LLM_TEMPERATURE` | `0.0` | Temperature |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per response |
| `LLM_CALL_DELAY` | `2` | Seconds between API calls (rate limiting) |
| `LLM_MAX_CALLS_PER_MINUTE` | `30` | Per-minute rate limit |

## Gold Cases

100 cases across 6 categories (development set):

| Category | Count | Tests |
|----------|:-----:|-------|
| `grounded` | 17 | All claims in evidence → accept |
| `missing_info` | 14 | Evidence doesn't cover question → abstain |
| `contradiction` | 15 | Conflicting facts in evidence → flag |
| `pressure` | 15 | Speculative question → hypothesis with caveats |
| `filler_trap` | 15 | Tempts model to invent facts → abstain |
| `partial_answer` | 24 | Some facts available, some not → partial |

**100 total cases — development set only. Not a held-out evaluation.**

## Tests

209 tests covering all modules. Run with `pytest -v`.

```
tests/
├── test_calibration.py          # Table-format probe validation
├── test_claim_filter.py         # Slot-aware relevance filtering
├── test_constants.py            # Shared stop words
├── test_contradiction_checks.py # Status-pair contradictions + possible_conflict audit
├── test_evidence_spans.py       # Abbreviation-aware splitting
├── test_gate.py                 # All gate rules + edge cases
├── test_inference_detector.py   # All 4 tiers + exact failure cases
├── test_metrics.py              # Pipeline + baseline metrics
├── test_schemas.py              # Pydantic validation
├── test_span_matcher.py         # Substring/fuzzy/numeric matching
└── test_verifier.py             # Batch table parser + integration
```

## What This Does NOT Do

- No internet search or retrieval (RAG)
- No vector databases
- No fine-tuning
- No UI or deployment
- No GPU required

This is a **research harness**, not a product.

## Known Limitations (v0.4)

The v0.4 baseline intentionally trades some detection for **zero false positives** and **maintainable code**.

| # | Limitation | Why | Mitigation |
|---|-----------|-----|------------|
| 1 | **Numeric contradictions not caught deterministically** | Money/percentage/count/date conflicts have too many false positives (e.g., revenue target vs actual revenue). | Relies on verifier LLM. If LLM misses, contradiction is not flagged. |
| 2 | **Semantic relevance not enforced** | "How fast can the car go?" with only engine specs supported → `accept`. v0.3.2 had a 20-entry synonym-table guard but it was too rule-heavy for a baseline. | Acceptable for v0.4. Future: semantic similarity check (not synonym table). |
| 3 | **100 cases = dev set only** | The deterministic rules were tuned against failures on this set. Results are directional, not publication-grade. | Create held-out 50-case test set for unbiased validation. |
| 4 | **Inference detector is regex-based** | Covers common hedges but cannot catch all inferential reasoning. | Grounded in CogniBench + GME + BioScope; handles most common cases. |
| 5 | **Single evidence document** | No multi-document consensus or evidence weighting. | Designed for single-pass evaluation. |

## Next Steps

- [x] Simplify to v0.4 baseline — status-pair contradictions only, no frame detector
- [x] Remove slot-mismatch guard (semantic relevance is known limitation)
- [x] 209 tests pass, zero false contradictions
- [ ] Run v0.4 eval on full 100-case development set
- [ ] Test on multiple models (1B, 4B, 70B+) to prove model independence
- [ ] Create held-out 50-case test set for unbiased evaluation
- [ ] Confidence calibration analysis

---

*See [DESIGN.md](DESIGN.md) for the full architecture document.*
