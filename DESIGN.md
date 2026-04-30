# Project Verity-H Design

**Status:** lightweight research prototype  
**Last updated:** April 2026  
**Scope:** evidence-gated verification, not general truthfulness

## 1. Research Goal

Verity-H tests whether a lightweight verification and deterministic gating layer can reduce unsupported claims when an LLM must answer from supplied evidence.

The goal is not to make a model omniscient. The goal is to make the answer boundary visible:

- what the evidence supports
- what the evidence does not provide
- where the evidence conflicts
- where the user is asking for a hypothesis, prediction, diagnosis, recommendation, or other inference

The project should stay simple enough to audit. Every non-LLM decision should be traceable to a small rule or parser, not a hidden model.

## 2. Non-Goals

Verity-H intentionally does not include:

- RAG or internet search
- vector databases
- fine-tuning
- external fact checking
- classifiers
- UI or deployment
- multi-agent workflows
- benchmark-score optimization

These may be useful in other systems, but they would blur the research question here.

## 3. Pipeline

```text
Question + evidence
    |
    v
Evidence splitter
    - deterministic sentence/paragraph spans
    - each span has a stable span_id
    |
    v
Writer LLM
    - drafts an answer using only supplied evidence
    |
    v
Verifier LLM
    - extracts claims from the draft
    - labels each claim
    - returns evidence span IDs where applicable
    |
    v
Deterministic post-processing
    - claim filtering
    - deterministic span matching
    - absence/deferral correction
    - inference detection
    - contradiction checks
    |
    v
Deterministic gate
    - accept
    - partial
    - needs_info
    - contradiction
    - hypothesis
    - partial_hypothesis
    - verifier_error
    |
    v
Final answer + structured metadata
```

The target budget is two LLM calls per case:

1. writer
2. batch verifier

Everything after the verifier is deterministic.

## 4. Claim Labels

The verifier uses these labels:

| Label | Meaning |
|-------|---------|
| `SUPPORTED` | Evidence directly supports the claim. Must include an evidence pointer. |
| `UNSUPPORTED` | The draft asserts something not supported by the evidence. |
| `NEEDS_INFO` | Evidence is related but insufficient. |
| `NOT_IN_EVIDENCE` | The requested information is absent, pending, unavailable, or not stated. |
| `CONTRADICTS_EVIDENCE` | The evidence conflicts with the claim. |

Claim labels drive gate decisions. Claim kinds are mainly for analysis.

## 5. Deterministic Components

### 5.1 Evidence Spans

`src/evidence_spans.py` splits evidence into traceable spans. The verifier and span matcher refer to these IDs so supported claims can be audited.

### 5.2 Verifier Parser

`src/verifier.py` asks the verifier LLM for a compact table rather than JSON. The parser supports:

- multiple span IDs
- normalized pre-filtering
- malformed-row accounting
- absence claims such as “not provided” or “not mentioned”

The prompt explicitly tells the verifier not to skip absence claims. A missing requested answer is itself a claim that must be labeled `NOT_IN_EVIDENCE`.

### 5.3 Claim Filter

`src/claim_filter.py` removes duplicate, meta, and irrelevant claims.

Important current behavior:

- Broad questions bypass overly strict relevance filtering. Questions like “What were the results?” or “What did the doctor report?” should preserve valid answer facts even if they do not repeat the broad question words.
- Direct missing-answer claims are preserved when they overlap the requested subject.
- Unknown side facts can still be filtered when they are unrelated to the user’s requested slot.

### 5.4 Span Matcher

`src/span_matcher.py` corrects common verifier mistakes using deterministic matching:

- exact substring match
- number + keyword match
- fuzzy keyword match with strict numeric consistency
- absence and deferral detection
- calculated-percentage guard

Absence/deferral examples routed to `NOT_IN_EVIDENCE` include:

- `not provided`
- `not stated`
- `not documented`
- `not shown`
- `not listed`
- `pending`
- `not finalized`
- `currently being reviewed`

The calculated-percentage guard prevents claims like “77.5% passed” from being accepted when evidence only states “186 out of 240” and does not state the percentage.

### 5.5 Inference Detector

`src/inference_detector.py` detects claims that go beyond direct evidence:

- epistemic hedges: `likely`, `suggests`, `consistent with`
- logical leaps: `therefore`, `based on these findings`
- recommendations: `should`, `recommended`, `indicated`
- predictive/speculative answers to speculative questions
- some mathematical or derived claims

This detector is regex-based by design. It is not complete, but it is auditable.

### 5.6 Contradiction Checks

`src/contradiction_checks.py` is intentionally conservative.

Forced contradictions include:

- obvious status pairs such as open/closed, approved/rejected, passed/failed, enabled/disabled
- selected requested-slot conflicts where the question makes the shared value clear
  - requested temperature conflicts
  - requested units-sold conflicts
  - apartment listing detail conflicts
  - selected legal/contract status conflicts

Possible numeric/date/money conflicts that are not clearly tied to the same requested slot should be logged or left to the verifier rather than forced. This prevents false contradictions such as target revenue vs actual revenue.

### 5.7 Gate

`src/gate.py` is pure deterministic logic. It never calls an LLM.

Decision priority:

| Priority | Condition | Decision |
|----------|-----------|----------|
| 0 | verifier parse error | `verifier_error` |
| 1 | contradiction present | `contradiction` |
| 2 | draft explicitly says evidence conflicts | `contradiction` |
| 3 | all claims supported and no missing-answer signal | `accept` |
| 4 | supported + unknown | `partial` |
| 5 | pressure + speculative + unknown | `hypothesis` or `partial_hypothesis` |
| 6 | no supported claims and unknown/missing answer | `needs_info` |
| 7 | no usable claims | `needs_info` |

Additional safety rules:

- If the draft says the requested answer is missing, the gate must not cleanly accept side facts.
- If the draft says the evidence has a conflict, the gate routes to `contradiction` even if the verifier failed to label a contradiction.
- For multi-slot questions, if the draft says one requested slot is missing and the verifier/filter drops that absence claim, the gate returns `partial` rather than accepting only the supported slots.

## 6. Output Semantics

| Decision | Intended meaning |
|----------|------------------|
| `accept` | The answer is fully supported by evidence pointers. |
| `partial` | Some requested information is supported and some is missing. |
| `needs_info` | The evidence does not answer the question. |
| `contradiction` | The evidence contains a conflict that blocks a clean answer. |
| `hypothesis` | The user asked for speculation and the answer must be caveated. |
| `partial_hypothesis` | Some facts are verified, but the conclusion remains speculative. |
| `verifier_error` | The verifier output could not be parsed safely. |

Unknowns, contradictions, and hypotheses should be visible in the final answer, not hidden in metadata only.

## 7. Providers

`src/llm_client.py` supports:

- `mock`
- `api` for OpenAI-compatible APIs, including local Ollama
- `groq`
- `hf_api`
- `nvidia`

Provider differences matter. Some models extract absence claims cleanly; others drop them or label them as supported meta-facts. The deterministic layer is designed to reduce this variance without adding provider-specific hacks.

## 8. Data

Current data files:

| File | Role |
|------|------|
| `data/gold_cases.jsonl` | Original 100-case development/debug set |
| `data/gold_cases_set1.jsonl` | Development split |
| `data/gold_cases_set2.jsonl` | Development split |
| `data/stress_cases_v0.1.jsonl` | 100-case diagnostic stress set |
| `data/stress_cases_key_24.jsonl` | 24-case high-value smoke/stress subset |
| `data/case_template.json` | Example case format |
| `docs/test_case_schema.md` | Public schema and methodology notes |

The current gold and stress cases are not publication-grade held-out data. They are for development, debugging, and provider comparison.

## 9. Evaluation Methodology

Use this sequence:

1. Run unit tests.
2. Validate cases.
3. Run a small smoke subset.
4. Run the full stress or development set.
5. Group failures by pattern before changing code.
6. Avoid patches that only improve one model or one known case.
7. Create a held-out set only after the system stabilizes.
8. Run held-out once, report it separately, and do not tune against it.

Metrics should include:

- unsupported claim rate
- unsupported claim rate among accepts
- correct abstention rate
- over-abstention rate
- grounded accept rate
- contradiction detection rate
- pressure hypothesis correctness
- hypothesis misuse rate
- partial answer coverage
- parse error rate
- verifier supported pointer rate
- latency
- token/call overhead where available

Baseline columns in generated reports are heuristic if baseline files are absent or generated separately. Pipeline metrics are based on structured verifier and gate outputs.

## 10. Known Limitations

### Semantic Relevance

The system still has limited semantic understanding. It can accept related evidence that does not fully answer the user’s intended slot. Broad-question bypasses are necessary for recall but can reduce precision.

### Numeric and Date Contradictions

Numeric/date/money contradictions are handled conservatively. This avoids false positives but means some real conflicts require the verifier LLM to catch them.

### Model-Dependent Verification

The verifier LLM can omit claims, especially missing-answer claims. The deterministic gate now catches some draft-level absence patterns, but not every possible phrasing.

### Synthetic Unknown Wording

When the gate adds an unknown because the draft signaled missing information, the generated unknown can be broad. This is safer than accepting, but less precise than a verifier-extracted missing slot.

### Development Data

The current 100 gold cases and stress cases have been inspected during development. They should not be presented as final held-out benchmark results.

## 11. Current Research Direction

Near-term:

- Run the 24-case key smoke after changes.
- Run the 100-case stress set across at least one local model.
- Compare provider behavior without changing code for isolated failures.
- Keep improving failure taxonomy and documentation.

Next-stage:

- Create a separate held-out set with broader real-world cases.
- Run it once after freezing code.
- Report dev, stress, and held-out results separately.
- Include honest failure examples and latency/cost overhead.

## 12. Engineering Principles

- Keep deterministic checks small and reviewable.
- Prefer explicit tests for every new rule.
- Do not chase dev-set scores.
- Do not add case-specific hacks.
- Prefer `needs_info` over guessing.
- Supported claims must have evidence pointers.
- If complexity rises without clear evaluation value, remove it.

Verity-H should remain a clear research artifact: easy to run, easy to audit, and honest about what it can and cannot show.
