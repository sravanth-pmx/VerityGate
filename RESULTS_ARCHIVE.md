# Results Archive — Project Verity-H

This file summarizes all evaluation results across versions. The actual result files
are stored in the repo history; this document provides a human-readable summary.

---

## v0.2.1 (Qwen3-4B-Instruct-2507, 30 cases)

Best pre-batch-verifier results. 7 LLM calls per case.

| Metric | Baseline Normal | Baseline Honesty | Verity-H |
|--------|----------------|-----------------|-------------------|
| unsupported_claim_rate | 10% | 0% | **0%** |
| correct_abstention_rate | 70% | 100% | **100%** |
| grounded_accept_rate | 0% | 0% | **100%** |
| contradiction_detection_rate | 60% | 40% | **80%** |
| pressure_hypothesis_correctness | 0% | 0% | **100%** |
| false_contradiction_rate | 0% | 0% | **0%** |
| partial_answer_coverage | 0% | 0% | **100%** |
| parse_error_rate | — | — | **0%** |
| latency_p50 | 3,525ms | 3,244ms | 11,339ms |

Stored in: `results_qwen3_4b_v6/`

---

## v0.3 (Qwen3-4B, 30 cases)

First batch-verifier version. 2 LLM calls per case. 43% faster.

| Metric | v0.2.1 | v0.3 | Change |
|--------|--------|------|--------|
| unsupported_claim_rate | 0% | 0% | ✅ |
| correct_abstention_rate | 100% | 100% | ✅ |
| grounded_accept_rate | 100% | 100% | ✅ |
| contradiction_detection_rate | 80% | 80% | ✅ |
| pressure_hypothesis_correctness | **100%** | **60%** | 🔻 (fixed in v0.3.1) |
| latency_p50 | 11,339ms | **6,495ms** | ✅ 43% faster |

Stored in: `results_qwen3_4b_v7/`

---

## v0.3.1 (Qwen3-4B, 100 cases)

Added inference detector. Fixed pressure regression.

Stored in: `results_real_llm/`

---

## v0.3.2 (Qwen3-4B, 100 cases, sandbox run)

Conservative frame detector, slot-mismatch guard, safer pressure routing.

| Metric | v0.3.1 | v0.3.2 | Target | Status |
|--------|--------|--------|--------|--------|
| false_contradiction_rate | 9.4% | **0.0%** | <3% | ✅ Fixed |
| pressure_hypothesis_correctness | 53.3% | **66.7%** | >75% | ⚠️ Missed |
| not_in_evidence_label_rate | 69% | **79.3%** | >80% | ⚠️ Just missed |
| unsupported_claim_rate | 15% | **0.0%** | 0% | ✅ Fixed |
| contradiction_detection_rate | 60% | **26.7%** | high | ❌ Regressed |

Stored in: `results/verity_pipeline_v0.3.2.jsonl`

---

## v0.4 — Simplified Baseline

**Status:** Current version. Simplified for maintainability.

Key simplifications:
- Frame-based contradiction detector removed (status-pair only)
- Slot-mismatch guard removed (semantic relevance is known limitation)
- Numeric/date/money conflicts logged as `possible_conflict` but NOT forced
- 209 tests pass, zero false contradictions
- 2 LLM calls per case

Stored in: `results/` (future runs)
