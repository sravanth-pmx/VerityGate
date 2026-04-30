# VerityGate Test Case Schema

VerityGate runnable evaluation cases are stored as JSON Lines (`.jsonl`): one JSON object per line. The pipeline currently expects the `GoldCase` schema below.

## Required Fields

- `id`: Stable unique case ID, e.g. `case_001` or `stress_042`.
- `category`: One of:
  - `grounded`
  - `missing_info`
  - `contradiction`
  - `pressure`
  - `filler_trap`
  - `partial_answer`
- `question`: User question to answer using only `evidence_text`.
- `evidence_text`: Source evidence available to the writer/verifier.
- `pressure_level`: `0` for ordinary factual use, `1` for pressure/speculative cases.
- `expected_supported_claims`: List of facts expected to be supportable.
- `expected_unknowns`: List of requested facts/conclusions absent from evidence.
- `expected_contradictions`: List of conflicts expected to be detected.
- `notes`: Short author note explaining the case intent.

## Category Rules

- `grounded`: Evidence fully answers the question. Must include `expected_supported_claims`.
- `missing_info`: Evidence does not answer the requested slot. Must include `expected_unknowns`.
- `contradiction`: Evidence contains mutually incompatible facts. Must include `expected_contradictions`.
- `pressure`: Question asks for a prediction, recommendation, diagnosis, cause, or judgment beyond evidence. Must use `pressure_level: 1` and include `expected_unknowns`.
- `filler_trap`: Evidence contains related distractors or computable-but-unstated values. Must include `expected_unknowns`.
- `partial_answer`: Evidence answers part of a multi-part question. Must include both `expected_supported_claims` and `expected_unknowns`.

## Methodology Notes

- Current `gold` and `stress` cases are diagnostic, not publication-grade held-out results.
- Do not tune code after inspecting held-out failures unless that case is moved to dev/stress and replaced.
- Prefer broad, real-world failure patterns over case-specific examples.
- Supported claims should be directly grounded in evidence. Derived arithmetic should be marked unknown unless explicitly stated by evidence.

## Validation

Run:

```powershell
python -m src.validate_gold_cases --cases data/stress_cases_v0.1.jsonl --dataset-version stress_v0.1
```
