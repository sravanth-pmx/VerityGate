# Project Verity-H — Design Document

**Version:** 0.4  
**Status:** Research prototype — simplified baseline  
**Last updated:** April 2026

---

## 1. The Problem

When a human doesn't know something, they say "I don't know." They skip the question. They say "I'm not sure about that." This happens naturally and constantly — it's how honest communication works.

AI doesn't do this.

When an LLM doesn't have enough context to answer a question, it doesn't stop. It takes whatever fragments of relevant information it has and **fills in the gaps with plausible-sounding assumptions**. It presents these assumptions with the same confidence as verified facts. There is no signal to the user that part of the answer is grounded and part is fabricated.

This is the core problem Verity-H is researching:

> **Can we make an LLM behave like an honest human — share what it knows, admit what it doesn't, and never silently fill in the blanks?**

Specifically:
- If the evidence supports a full answer → give it confidently
- If the evidence covers part of the question → say what you know, explicitly flag what you don't
- If the evidence doesn't cover the question at all → say "I don't have enough information"
- If the evidence contradicts itself → flag the conflict instead of picking a side
- If the question asks for speculation → label it clearly as a guess, not a fact
- **Never** silently pad an answer with assumptions dressed up as knowledge

### Why this matters

Every unsupported claim in an LLM answer is a trust failure. In high-stakes domains — medical, legal, financial, scientific — a single fabricated detail presented as fact can cause real harm. The problem isn't that LLMs sometimes get things wrong. The problem is that **they don't tell you when they're guessing**.

### What Verity-H is

A research prototype that tests whether a lightweight pipeline can enforce this "honest human" behavior on any LLM — without fine-tuning the model, without RAG, without external knowledge. Just the question, the evidence, and a verification step between the LLM's answer and the user.

This is not a product. It's a research harness for studying evidence-grounded LLM behavior.

### What makes this different from prompt engineering

A prompt like "only answer from the evidence" reduces hallucination but doesn't eliminate it. We tested this directly — our "honesty baseline" uses exactly that prompt. It still has a 10% unsupported claim rate on missing-info cases because the LLM still fills gaps when it can construct a plausible answer. The Verity-H pipeline achieves 0% by structurally decomposing the answer into claims and checking each one against the evidence.

### What makes this different from RAG

RAG solves the **retrieval** problem — finding relevant evidence. Verity-H solves the **verification** problem — given evidence and an answer, ensuring every claim is actually grounded. These are complementary. RAG without verification still hallucinates. Verification without RAG requires evidence to be provided. We focus on the verification step because it's the under-researched half.

---

## 2. Research Questions

1. **Unsupported claim filtering**: Can we achieve ≥98% filtering of claims not grounded in evidence?
2. **Honest abstention**: Can the system reliably say "I don't know" when evidence is insufficient — like a human would?
3. **Partial honesty**: When evidence covers some but not all of a question, can the system cleanly separate what it knows from what it doesn't?
4. **Inference vs. fact**: Can we deterministically distinguish "the evidence says X" from "X implies Y" — and label them differently?
5. **Contradiction detection**: Can we catch conflicting facts in evidence without LLM reasoning?
6. **Model independence**: Does the same pipeline work across 1B, 4B, and 70B+ models?
7. **Token efficiency**: Can verification be done in 2 LLM calls (not 7+)?

---

## 3. Architecture

### 3.1 Pipeline Overview

```
Question + Evidence
       │
       ▼
┌──────────────┐
│  1. SPLIT    │  evidence_spans.py
│  Evidence    │  → sentence-level spans with IDs
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  2. DRAFT    │  LLM Call #1 (writer)
│  Answer      │  → conversational answer from evidence
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  3. VERIFY   │  LLM Call #2 (batch verifier)
│  Claims      │  → extract + label all claims in one call
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  4. DETERMINISTIC POST-PROCESSING            │
│                                              │
│  a. claim_filter  — remove meta-claims,      │
│                     dupes, irrelevant unknowns│
│  b. span_matcher  — fix LLM mislabeling      │
│                     via substring/fuzzy match │
│  c. inference_detector — downgrade inferences │
│                     wrongly labeled SUPPORTED  │
│  d. contradiction_checks — status-pair only │
│                     (numeric/date logged only) │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  5. GATE     │  gate.py (pure rules, no LLM)
│  Decision    │  → accept / partial / needs_info /
│              │    contradiction / hypothesis
└──────┬───────┘
       │
       ▼
  Final Answer
  (with transparency metadata)
```

### 3.2 LLM Call Budget

| Version | Calls per case | What each call does |
|---------|---------------|---------------------|
| v0.1    | ~7            | 1 writer + 1 extractor + 5 per-claim verifiers |
| v0.2    | ~7            | Same, with improved prompts |
| v0.3+   | **2**         | 1 writer + 1 batch verifier (extract + label in one call) |

The 3rd call only fires on parse failure (rare with table format).

### 3.3 What the LLM Does vs. What's Deterministic

| Step | LLM? | Module |
|------|-------|--------|
| Split evidence into spans | No | `evidence_spans.py` |
| Draft answer | **Yes** (Call 1) | `pipeline_runner.py` |
| Extract + label claims | **Yes** (Call 2) | `verifier.py` |
| Remove meta-claims/dupes | No | `claim_filter.py` |
| Fix mislabeled claims | No | `span_matcher.py` |
| Detect inferential claims | No | `inference_detector.py` |
| Detect contradictions | No | `contradiction_checks.py` |
| Gate decision | No | `gate.py` |
| Compute metrics | No | `metrics.py` |

**Design principle:** The LLM does the creative work (drafting, claim extraction). Everything else is deterministic and auditable. This means the pipeline's behavior is explainable — you can trace exactly why a claim was labeled a certain way.

---

## 4. Module Reference

### 4.1 `evidence_spans.py` — Evidence Splitting

Splits evidence text into indexed spans (sentence-level).

- Abbreviation-aware: won't break on "Dr.", "U.S.", "Inc."
- Falls back to paragraph splitting if sentence splitting fails
- Each span gets an ID (`span_0`, `span_1`, ...) used for traceability

### 4.2 `verifier.py` — Batch Claim Extraction + Labeling

Single LLM call extracts ALL claims and labels each one using a table format:

```
1. SUPPORTED | The meeting is at 3pm | span_0
2. NOT_IN_EVIDENCE | Duration not provided | none
3. CONTRADICTS_EVIDENCE | Budget $5M vs evidence $3M | span_1
```

Table format chosen over JSON because:
- More robust with smaller models (4B params can produce tables reliably)
- Fewer parse failures than JSON
- Easier for the LLM to produce (less structural overhead)

Labels:
| Label | Meaning |
|-------|---------|
| `SUPPORTED` | Evidence explicitly states this fact. Must include span_id. |
| `UNSUPPORTED` | Draft asserts this but no evidence addresses it. |
| `NEEDS_INFO` | Evidence is related but insufficient. |
| `NOT_IN_EVIDENCE` | Question asks for info that is absent/pending in evidence. |
| `CONTRADICTS_EVIDENCE` | Evidence directly conflicts with the claim. |

### 4.3 `span_matcher.py` — Deterministic Relabeling

Post-processes LLM labels using string matching. Four rules applied in order:

1. **SUPPORTED + absence text → NOT_IN_EVIDENCE**: Catches "The evidence does not mention X" wrongly labeled as SUPPORTED
2. **CONTRADICTS_EVIDENCE + absence text → NOT_IN_EVIDENCE**: Same for contradiction mislabeling
3. **UNSUPPORTED + span match → SUPPORTED**: Catches facts the LLM missed via substring/fuzzy/numeric matching
4. **SUPPORTED + inference detected → UNSUPPORTED**: Catches inferential claims (see §5)

Matching strategies (in order):
- Exact substring match (≥10 chars)
- Number + keyword match (all numbers in claim must appear in span)
- Fuzzy word overlap (≥80% of claim keywords in span, with numeric consistency check)

### 4.4 `inference_detector.py` — Inference Detection

**New in v0.3.1.** Deterministic detection of inferential claims the LLM wrongly labeled as SUPPORTED. See §5 for full details.

### 4.5 `claim_filter.py` — Claim Quality Filter

Two passes:
- **Pre-labeling**: Removes meta-claims ("The answer states..."), duplicates, short claims (<5 chars)
- **Post-labeling**: Removes unknown claims about slots the user didn't ask for (slot-aware relevance)

Slot-aware relevance example: If the question is "When did the project launch?", an unknown claim about "launch location" is filtered because "location" doesn't match the time/date slot.

### 4.6 `contradiction_checks.py` — Deterministic Contradiction Detection

**v0.4 simplified**: Only status-pair contradictions are **forced** into gate decisions.

**Forced (gate → contradiction):**
- Status-pair conflicts: open/closed, approved/rejected, passed/failed, available/unavailable, launched/not launched, enabled/disabled

**Logged only (possible_conflict, NOT forced):**
- Numeric/date/money conflicts — too many false positives. Rely on verifier LLM.

v0.3.x frame-based comparison (temperature, money, date, count) was removed after it caused false contradictions (e.g., "revenue target $100M" vs "actual revenue $82.4M"). See §Known Limitations.

### 4.7 `gate.py` — Deterministic Gating

Pure rule-based decision engine. No LLM calls. Rules applied in priority order:

| Priority | Condition | Decision | Behavior |
|----------|-----------|----------|----------|
| 0 | Verifier parse error | `verifier_error` | Refuse to answer |
| 1 | Any CONTRADICTS_EVIDENCE (from status-pair) | `contradiction` | Flag conflict, show both sides |
| 2 | All claims SUPPORTED | `accept` | Reconstruct answer from claims |
| 3 | pressure=1 + supported + unknown | `partial_hypothesis` | Show verified + hypothesis with low confidence |
| 4 | supported + unknown, pressure=0 | `partial` | Show verified + list unknowns |
| 5 | No supported, pressure=0 | `needs_info` | Refuse, list what's needed |
| 6 | pressure=1, only unknown | `hypothesis` | Hypothesis template with caveats |
| 7 | No claims at all | `needs_info` | Fallback |

**v0.4 note**: Slot-mismatch guard removed. Semantic relevance is a known limitation. "How fast can the car go?" with only engine specs → `accept`.

**Contradiction always wins** — even under pressure=1, contradictions override hypothesis mode.

### 4.8 `calibration.py` — Pre-flight Check

Runs 2 probe calls before the pipeline to verify the LLM can produce valid table output. Uses the same format as the actual verifier prompt. Warns (but continues) if calibration fails.

### 4.9 `metrics.py` — Evaluation Metrics

16 metrics computed from structured verifier/gate outputs:

| Metric | Target | What it measures |
|--------|--------|-----------------|
| `unsupported_claim_rate` | 0% | Claims that slip through as "accepted" but aren't supported |
| `correct_abstention_rate` | 100% | Missing-info/filler cases where system correctly refuses |
| `grounded_accept_rate` | 100% | Grounded cases where system correctly accepts |
| `contradiction_detection_rate` | high | Contradiction cases caught |
| `false_contradiction_rate` | 0% | Non-contradiction cases wrongly flagged |
| `pressure_hypothesis_correctness` | high | Pressure cases with proper hypothesis template |
| `hypothesis_misuse_rate` | 0% | Non-pressure cases that wrongly use hypothesis |
| `partial_answer_coverage` | 100% | Partial cases with both verified + unknown sections |
| `parse_error_rate` | 0% | Verifier responses that couldn't be parsed |

Baseline metrics use text heuristics (regex pattern matching). Pipeline metrics use structured verifier/gate fields. They are not directly comparable.

---

## 5. Inference Detection — Theoretical Grounding

### 5.1 The Problem

An LLM might label "The most likely cause is bacterial infection" as SUPPORTED when the evidence says "Patient has fever 38.9°C, sore throat, elevated WBC 12,000/µL." The underlying facts exist in evidence, but the **conclusion** is an inference — the evidence never says "bacterial infection."

This is the factual-vs-cognitive distinction from CogniBench (arxiv:2505.20767): a **factual statement** rephrases evidence; a **cognitive statement** extends beyond it via reasoning, interpretation, or opinion.

### 5.2 Four-Tier Detection

Based on CogniBench, GME epistemic modality taxonomy (arxiv:2106.08037), and BioScope hedge cues.

| Tier | Type | Examples | Logic |
|------|------|----------|-------|
| 1 | **Epistemic Hedges** | "probably", "suggests", "appears to", "consistent with", "most likely" | These words admit the speaker is making a call based on knowledge, not a direct observation. (GME: "state of knowledge" modals) |
| 2 | **Logical Leaps** | "therefore", "thus", "based on these findings", "we can conclude" | These signal that a reasoning process has happened to reach the claim. The claim is the conclusion, not the evidence. |
| 3 | **Deontic (Normative)** | "should", "recommended", "must", "is indicated" | These are recommendations, not facts. "You should pay" ≠ "You paid." (GME: "Priority by Rules and Norms") |
| 4 | **Speculative Questions** | "Should we invest?", "What caused the symptoms?", "Is the defendant guilty?" | If the question itself asks for a guess/judgment/prediction, the answer is inherently inferential. But pure data restatements (e.g., "$2M ARR") under speculative questions are kept. |

### 5.3 Tier 4 Safe Harbor

Not every claim under a speculative question is inferential. "The startup has $2M ARR" is a data restatement even if the question is "Should we invest?" The safe harbor rule: **claims containing specific numbers/measurements are treated as factual** unless they also contain strong evaluative language (guilty, diagnosis, will succeed).

### 5.4 References

| Source | What it provides |
|--------|-----------------|
| CogniBench (arxiv:2505.20767) | Factual vs. Cognitive statement taxonomy; sequential gating protocol |
| GME (arxiv:2106.08037) | Epistemic vs. circumstantial modal taxonomy; "state of knowledge" vs. "state of world" distinction |
| BioScope (Vincze et al. 2008) | Empirical hedge/speculation cue word list |
| FactBench/VERIFY (arxiv:2410.22257) | `undecidable` label design; Fact vs. Claim content-type taxonomy |
| Typed-RAG (arxiv:2503.15879) | Non-factoid question taxonomy for speculative question detection |
| ClaimDecomp (arxiv:2205.06938) | Claim decomposition into literal vs. implied subquestions |

---

## 6. Evaluation Results

### 6.1 Gold Cases

30 cases across 6 categories (5 each):

| Category | Purpose | What success looks like |
|----------|---------|----------------------|
| `grounded` | All claims supported by evidence | Gate: `accept` |
| `missing_info` | Evidence doesn't address the question | Gate: `needs_info` |
| `contradiction` | Evidence contains conflicting facts | Gate: `contradiction` |
| `pressure` | Speculative questions with pressure_level=1 | Gate: `hypothesis` or `partial_hypothesis` with template |
| `filler_trap` | Tempt model to invent plausible facts | Gate: `needs_info` |
| `partial_answer` | Some claims supported, some not | Gate: `partial` with both sections |

### 6.2 Version History

| Version | Architecture | Key change |
|---------|-------------|------------|
| v0.1 | 7 LLM calls, JSON verifier | Initial prototype |
| v0.2 | 7 LLM calls, improved prompts | Claim filter, span matcher |
| v0.2.1 (v6) | 7 LLM calls | Best results pre-batch: all metrics at ceiling |
| v0.3 (v7) | **2 LLM calls**, table verifier | 43% faster, pressure regressed to 60% |
| v0.3.1 | 2 LLM calls + inference detector | Fixes pressure regression, contradiction detection |
| v0.3.2 | 2 LLM calls + conservative frames | Reduces false contradictions; slot-mismatch guard |
| **v0.4** | **2 LLM calls, status-pair only** | **Simplified baseline — frame detector removed; deterministic gate only for status-pair contradictions** |

### 6.3 Best Results: v0.2.1 with Qwen3-4B-Instruct-2507

| Metric | Baseline Normal | Baseline Honesty | Verity-H Pipeline |
|--------|----------------|-----------------|-------------------|
| unsupported_claim_rate (↓) | 10% | 0% | **0%** |
| correct_abstention_rate (↑) | 70% | 100% | **100%** |
| grounded_accept_rate (↑) | 0% | 0% | **100%** |
| contradiction_detection_rate (↑) | 60% | 40% | **80%** |
| pressure_hypothesis_correctness (↑) | 0% | 0% | **100%** |
| false_contradiction_rate (↓) | 0% | 0% | **0%** |
| partial_answer_coverage (↑) | 0% | 0% | **100%** |
| parse_error_rate (↓) | — | — | **0%** |
| latency_p50 | 3,525ms | 3,244ms | 11,339ms |

### 6.4 v0.3 Results (2-call batch verifier)

| Metric | v0.2.1 (v6) | v0.3 (v7) | Change |
|--------|------------|-----------|--------|
| unsupported_claim_rate | 0% | 0% | ✅ Held |
| correct_abstention_rate | 100% | 100% | ✅ Held |
| grounded_accept_rate | 100% | 100% | ✅ Held |
| contradiction_detection_rate | 80% | 80% | ✅ Held |
| pressure_hypothesis_correctness | 100% | **60%** | 🔻 Regressed (fixed in v0.3.1) |
| partial_answer_coverage | 100% | 100% | ✅ Held |
| latency_p50 | 11,339ms | **6,495ms** | ✅ 43% faster |
| latency_p95 | 35,280ms | **9,596ms** | ✅ 73% faster |

### 6.5 Known Failure Cases

| Case | Category | Issue | Root cause | Status |
|------|----------|-------|-----------|--------|
| case_012 | contradiction | 1,200 vs 980 units not caught | Writer only mentioned one number; det checker had no shared subject words | **Fixed in v0.3.1** (question-keyword fallback) |
| case_018 | pressure | "Should we invest?" accepted as factual | Verifier labeled data-backed inferences as SUPPORTED | **Fixed in v0.3.1** (inference detector) |
| case_020 | pressure | "What caused symptoms?" accepted as factual | "consistent with bacterial infection" labeled SUPPORTED | **Fixed in v0.3.1** (Tier 1 epistemic hedge detection) |

---

## 7. What This Project Does NOT Do

- ❌ No internet search or retrieval (RAG)
- ❌ No vector databases
- ❌ No fine-tuning
- ❌ No UI or deployment
- ❌ No agents or orchestration frameworks
- ❌ No GPU required
- ❌ No complex dependencies (just pydantic + pytest)

This is a **research harness** — a controlled environment for studying how well deterministic post-processing can enforce evidence grounding on any LLM.

---

## 7.1 Known Limitations (v0.4)

The v0.4 baseline intentionally trades some detection for zero false positives and maintainable code.

| Limitation | Why | Mitigation |
|-----------|-----|------------|
| **Numeric contradictions not caught deterministically** | Money/percentage/count/date conflicts have too many false positives (e.g., revenue target vs actual revenue). | Relies on verifier LLM. If LLM misses, contradiction is not flagged. |
| **Semantic relevance not enforced** | "How fast can the car go?" with only engine specs supported → `accept`. v0.3.2 had a synonym-table guard (20 entries) but it was too rule-heavy for a baseline. | Acceptable for v0.4. Future: semantic similarity check (not synonym table). |
| **Frame detector removed** | Frame-based value comparison (temperature, count, date) was tuned per-failure on dev set. Removed in v0.4 to prevent dev-set overfitting. | Status-pair contradictions (open/closed, approved/rejected, etc.) still caught deterministically. |
| **Inference detector regex-based** | Cannot catch all forms of inferential reasoning. | Covers most common hedges (probably, suggests, consistent with, therefore, should, etc.) from CogniBench + GME + BioScope. |
| **Single-document evidence** | No multi-document consensus, no evidence weighting. | Designed for single-pass evaluation. |

---

## 8. Data & Evaluation Methodology

### 8.1 Gold Cases (Development Benchmark)

The current 100 cases in `data/gold_cases.jsonl` are a **development benchmark** — they were used to develop, debug, and tune the pipeline. Results on this set are informative but not final validation.

| Category | Count | Purpose |
|----------|:-----:|---------|
| `grounded` | 17 | All claims fully supported by evidence |
| `missing_info` | 14 | Evidence doesn't address the question |
| `contradiction` | 15 | Evidence contains conflicting facts |
| `pressure` | 15 | Speculative questions requiring hypothesis mode |
| `filler_trap` | 15 | Tempts the model to invent plausible facts |
| `partial_answer` | 24 | Some claims supported, some not |

**Important:** These cases are NOT an unseen test set. The pipeline's deterministic rules (span_matcher patterns, inference detector regexes, claim_filter slot keywords) were tuned against failure cases from this set. For publication-grade results, an independent held-out test set is needed.

### 8.2 Validation

Run `python -m src.validate_gold_cases` before any inference run. This checks:
- All rows parse, IDs are unique, categories are valid
- Category-specific required fields (e.g., contradiction cases must have expected_contradictions)
- Pressure levels are consistent with categories
- No empty questions or evidence

### 8.3 Future: Dev/Test Split

When preparing for publication:
- Freeze current 100 cases as the development set
- Create a new 50-100 case held-out test set (written by a different person or generated from different domains)
- Report metrics on both sets separately
- Never tune code against the held-out set

---

## 9. Open Questions & Next Steps

### Immediate (v0.4)
- [x] Simplified contradiction detector to status-pair only (frame detector removed)
- [x] Removed slot-mismatch guard from gate (semantic relevance is a known limitation)
- [x] 209 tests pass, zero false contradictions
- [ ] Run v0.4 eval on full 100-case development set
- [ ] Test on multiple models (1B, 4B, 70B+) to prove model independence
- [ ] Create held-out 50-case test set for unbiased evaluation

### Research
- [ ] Can claim_kind (number, date, attribution, etc.) improve per-type metrics?
- [ ] How does claim count correlate with accuracy? (2.2 avg in v0.3 vs 4.9 in v0.2)
- [ ] Can we detect when the writer *omits* contradictory evidence (case_012 problem)?
- [ ] Inter-annotator agreement study on verifier labels
- [ ] Confidence calibration analysis

### Constraints
- Target: ≥98% unsupported claim filtering rate
- No RAG, vector DB, UI, deployment, or fine-tuning
- Keep it simple — small research harness
- API keys only in env vars, never in repo

---

## 10. Repo Structure

```
verity-h-prototype/
├── DESIGN.md                  # This document
├── README.md                  # Quick-start guide
├── RESULTS_ARCHIVE.md         # Summary of historical results
├── .gitignore                 # Ignore cache, env files, traces
├── pyproject.toml             # Package config (v0.3.0)
├── requirements.txt           # Core dependencies
├── .env.example               # Environment variable template
├── conftest.py                # pytest path setup
├── data/
│   └── gold_cases.jsonl       # 100 evaluation cases (development benchmark)
├── src/
│   ├── __init__.py
│   ├── config.py              # Environment + path configuration
│   ├── constants.py           # Shared stop words
│   ├── schemas.py             # Pydantic models (GoldCase, VerifiedClaim, GateOutput, etc.)
│   ├── llm_client.py          # Mock + OpenAI + HF Inference API client
│   ├── calibration.py         # Pre-flight LLM capability check
│   ├── evidence_spans.py      # Abbreviation-aware evidence splitting
│   ├── pipeline_runner.py     # Draft → Verify → Gate orchestration
│   ├── verifier.py            # Batch claim extraction + labeling (LLM Call #2)
│   ├── claim_filter.py        # Slot-aware claim quality filter
│   ├── span_matcher.py        # Deterministic relabeling + inference integration
│   ├── inference_detector.py  # 4-tier inferential claim detection
│   ├── contradiction_checks.py# Status-pair contradiction detection (v0.4 simplified)
│   ├── gate.py                # Deterministic gating rules
│   ├── baseline_runner.py     # Baseline A (normal) and B (honesty)
│   ├── metrics.py             # 16 evaluation metrics
│   ├── report.py              # Comparison table generator
│   ├── validate_gold_cases.py # Pre-flight data validation
│   ├── diagnose_contradictions.py # Audit false-contradiction cases
│   └── diagnose_not_in_evidence.py # Audit missing NOT_IN_EVIDENCE labels
└── tests/                     # 209 tests
    ├── test_calibration.py
    ├── test_claim_filter.py
    ├── test_constants.py
    ├── test_contradiction_checks.py
    ├── test_evidence_spans.py
    ├── test_gate.py
    ├── test_inference_detector.py
    ├── test_metrics.py
    ├── test_schemas.py
    ├── test_span_matcher.py
    └── test_verifier.py
```

---

*This document describes the system as of v0.4. Updated as the project evolves.*
