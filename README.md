# Project Verity-H

Verity-H is a lightweight evidence-gated LLM verification prototype.

The project tests a narrow research question:

> Can a small verification and gating layer reduce unsupported LLM claims when the model must answer from provided evidence only?

It is not trying to solve truthfulness in general. It does not use retrieval, fine-tuning, classifiers, vector databases, or external knowledge. The goal is a simple, auditable research harness that can be tested across multiple LLM providers with controlled token and latency overhead.

For architecture details, see [DESIGN.md](DESIGN.md). For public test-case format, see [docs/test_case_schema.md](docs/test_case_schema.md).

## Current Status

- Two LLM calls per case: writer, then batch verifier.
- Deterministic post-processing fixes common verifier mistakes.
- Deterministic gate returns one of:
  - `accept`
  - `partial`
  - `needs_info`
  - `contradiction`
  - `hypothesis`
  - `partial_hypothesis`
  - `verifier_error`
- Supported claims must have evidence pointers.
- Unknowns, contradictions, and hypotheses are shown in the final answer.
- Current 100 gold cases are a development/debug set, not publication-grade held-out results.
- A separate 100-case stress set and 24-case key smoke subset are included for diagnostic evaluation.

## Current Diagnostic Runs

These are development/stress diagnostics, not held-out benchmark results. Baseline columns are omitted here because the saved reports were pipeline-only runs.

### 100-Case Development Set Splits

The original 100 development cases were run as two 50-case splits for provider quota and resumability.

| Provider / model | Cases | Unsupported among accepts | Correct abstention | Grounded accept | Contradiction detection | Pressure correctness | Partial coverage | Parse errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Groq Llama 70B, set 1 | 50 | 0.0% | 100.0% | 100.0% | 80.0% | 80.0% | 100.0% | 0.0% |
| Groq Llama 70B, set 2 | 50 | 0.0% | 100.0% | n/a | 100.0% | 90.0% | 100.0% | 0.0% |
| NVIDIA Devstral, set 1 | 50 | 0.0% | 100.0% | 100.0% | 80.0% | 100.0% | 80.0% | 0.0% |
| NVIDIA Devstral, set 2 | 50 | 0.0% | 90.9% | n/a | 90.0% | 90.0% | 100.0% | 0.0% |

`n/a` means that split did not contain grounded cases, so the metric is not meaningful for that split.

### 24-Case Key Smoke

The key smoke set is balanced across six categories and is intended for quick regression checks before full runs.

| Provider / model | Cases | Unsupported among accepts | Correct abstention | Grounded accept | Contradiction detection | Pressure correctness | Partial coverage | Parse errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ollama Gemma 31B | 24 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% | 0.0% |
| NVIDIA Devstral | 24 | 0.0% | 87.5% | 100.0% | 100.0% | 100.0% | 50.0% | 0.0% |

After the latest partial-answer guard, the targeted NVIDIA mini rerun corrected the two main key-smoke accept/partial failures (`stress_077`, `stress_090`). A fresh full 24-case or 100-case run should be used for any updated headline table.

## Pipeline

```text
Question + evidence
    |
    v
Split evidence into spans              deterministic
    |
    v
Writer drafts answer                   LLM call 1
    |
    v
Verifier extracts and labels claims    LLM call 2
    |
    v
Deterministic post-processing
  - claim filtering
  - span matching
  - absence/deferral correction
  - inference detection
  - conservative contradiction checks
    |
    v
Deterministic gate
    |
    v
Final answer with verification metadata
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[test]"
```

Run tests in mock mode:

```powershell
pytest
```

Validate the main development cases:

```powershell
python -m src.validate_gold_cases
```

Validate the stress cases:

```powershell
python -m src.validate_gold_cases --cases data\stress_cases_v0.1.jsonl --dataset-version stress_v0.1
python -m src.validate_gold_cases --cases data\stress_cases_key_24.jsonl --dataset-version stress_key_24
```

## LLM Providers

Set `LLM_MODE` and `MODEL_NAME` in `.env` or the shell.

| Mode | Use case |
|------|----------|
| `mock` | Tests and local development without API calls |
| `api` | OpenAI-compatible APIs, including local Ollama |
| `groq` | Groq API |
| `hf_api` | Hugging Face Inference API |
| `nvidia` | NVIDIA NIM / NVIDIA API Catalog |

Examples:

```powershell
# Local Ollama through OpenAI-compatible API
$env:LLM_MODE="api"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:MODEL_NAME="gemma3:27b"
```

```powershell
# NVIDIA API Catalog
$env:LLM_MODE="nvidia"
$env:NVIDIA_API_KEY="your-key"
$env:MODEL_NAME="nvidia/devstral-small-2507"
```

```powershell
# Groq
$env:LLM_MODE="groq"
$env:GROQ_API_KEY="your-key"
$env:MODEL_NAME="llama-3.3-70b-versatile"
```

Do not commit `.env` or API keys.

## Running Evaluations

Small key smoke set, recommended before larger runs:

```powershell
python run_pipeline_batched.py --cases data\stress_cases_key_24.jsonl --output results\ollama_stress_key_24.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\ollama_stress_key_24.jsonl --output results\ollama_stress_key_24_report.md
```

Full stress set:

```powershell
python run_pipeline_batched.py --cases data\stress_cases_v0.1.jsonl --output results\ollama_stress_v0.1.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\ollama_stress_v0.1.jsonl --output results\ollama_stress_v0.1_report.md
```

Full stress set with local Ollama/Gemma, assuming Ollama is running and the model name matches `ollama list`:

```powershell
$env:LLM_MODE="api"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:MODEL_NAME="gemma3:27b"

python run_pipeline_batched.py --cases data\stress_cases_v0.1.jsonl --output results\ollama_gemma31_stress_v0.1.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\ollama_gemma31_stress_v0.1.jsonl --output results\ollama_gemma31_stress_v0.1_report.md
```

Original 100-case development set:

```powershell
python run_pipeline_batched.py --cases data\gold_cases.jsonl --output results\dev_gold_pipeline.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\dev_gold_pipeline.jsonl --output results\dev_gold_report.md
```

The batched runner is resumable at the output-file level: if an output JSONL already contains completed cases, it skips them.

## Datasets

| File | Purpose |
|------|---------|
| `data/gold_cases.jsonl` | 100-case development/debug set |
| `data/gold_cases_set1.jsonl` | Split of the development set |
| `data/gold_cases_set2.jsonl` | Split of the development set |
| `data/stress_cases_v0.1.jsonl` | 100-case diagnostic stress set |
| `data/stress_cases_key_24.jsonl` | 24-case high-value smoke/stress subset |
| `data/case_template.json` | Example case object for contributors |
| `docs/test_case_schema.md` | Public case schema and methodology notes |

Development and stress results should be reported separately. Do not tune code after looking at a held-out set unless the case is moved to development/stress and replaced.

## Current Deterministic Checks

The deterministic layer intentionally stays small:

- Evidence span matching by substring, number+keyword match, and fuzzy keyword overlap with strict numeric consistency.
- Absence and deferral detection, such as `not provided`, `not documented`, `not shown`, `pending`, or `not finalized`.
- Calculated percentage guard: unstated computed percentages should not be accepted as supported facts.
- Inference detection for hedges, logical leaps, recommendations, predictions, diagnoses, and other speculative claims.
- Conservative contradiction checks:
  - obvious status-pair conflicts
  - selected requested-slot value conflicts where the question and evidence make the shared slot clear
- Draft-level safety checks:
  - if the draft says the answer is missing, do not accept side facts as a clean answer
  - if the draft says the evidence conflicts, route to contradiction
  - for multi-slot questions, if the draft says one requested slot is missing, do not accept only the supported slots

These checks are intentionally auditable regex/string rules, not a hidden classifier.

## What This Does Not Do

- No internet search
- No RAG
- No vector database
- No fine-tuning
- No UI or deployment
- No publication-grade benchmark claim yet

## Known Limitations

- The current gold and stress cases are diagnostic. They are useful for engineering, not final scientific claims.
- Semantic relevance is still limited. The system can miss cases where evidence is related but not actually sufficient.
- Numeric/date/money contradiction detection is conservative to avoid false positives.
- The verifier is still model-dependent: stronger LLMs extract cleaner absence and partial-answer claims.
- Final-answer wording for synthetic unknowns can be broad when the verifier drops a specific missing slot.
- Single-document evidence only; no source ranking or evidence precedence.

## Recommended Evaluation Sequence

1. Run `pytest`.
2. Run `python -m src.validate_gold_cases`.
3. Run the 24-case key smoke set.
4. If the smoke result is sane, run the 100-case stress set.
5. Only after the design stabilizes, create a held-out set and run it once.

## Repo Structure

```text
data/                       evaluation cases and templates
docs/                       public schema and methodology docs
results/                    local run outputs
src/
  baseline_runner.py        baseline prompts
  claim_filter.py           claim cleanup and relevance filtering
  contradiction_checks.py   deterministic contradiction checks
  evidence_spans.py         evidence splitting
  gate.py                   deterministic gate
  inference_detector.py     inference/speculation detection
  llm_client.py             provider clients
  pipeline_runner.py        pipeline orchestration
  span_matcher.py           deterministic relabeling
  validate_gold_cases.py    dataset validation
  verifier.py               batch verifier prompt and parser
tests/                      unit tests
run_pipeline_batched.py     resumable evaluation runner
```

Verity-H is a research harness. Treat results as evidence for design decisions, not as leaderboard claims.
