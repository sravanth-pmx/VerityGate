# VerityGate

**Teaching AI to say "I don't know."**

When humans lack knowledge, they admit it: "I'm not sure", "I don't know", "let me check." LLMs often do the opposite. They fill gaps with plausible-sounding assumptions and present them as facts.

VerityGate researches whether a lightweight verification pipeline can enforce more honest behavior: **share what is supported, flag what is missing, and never silently guess.**

The project tests a narrow research question:

> Can a small verification and gating layer reduce unsupported LLM claims when the model must answer from provided evidence only?

It is not trying to solve truthfulness in general. It does not use retrieval, fine-tuning, classifiers, vector databases, or external knowledge. The goal is a simple, auditable research harness that can be tested across multiple LLM providers with controlled token and latency overhead.

For architecture details, see [DESIGN.md](DESIGN.md). For public test-case format, see [docs/test_case_schema.md](docs/test_case_schema.md).

## Current Status

- VerityGate is a lightweight research prototype for evidence-gated answering.
- It uses two LLM calls per case: writer, then batch verifier.
- Deterministic post-processing fixes common verifier mistakes before the gate runs.
- The deterministic gate returns one of:
  - `accept`
  - `partial`
  - `needs_info`
  - `contradiction`
  - `hypothesis`
  - `partial_hypothesis`
  - `verifier_error`
- Supported claims must have evidence pointers.
- Unknowns, contradictions, and hypotheses are shown in the final answer.
- The current 100-case stress set is diagnostic, not a publication-grade held-out benchmark.
- The current implementation is the strongest diagnostic baseline so far: it is conservative, model-agnostic, and has shown zero unsupported accepted claims across the latest reviewed full stress runs.

## Current Diagnostic Runs

These are diagnostic full stress runs on `data/stress_cases_v0.1.jsonl` using the current v5 implementation. They are useful for engineering comparison and failure analysis. They should not be read as publication-grade benchmark claims.

| Model | Cases | Unsupported among accepts | Correct abstention | Over-abstention | Grounded accept | Contradiction detection | Pressure correctness | Partial coverage | Parse errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ministral 3 14B | 100 | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| Gemma 4 | 100 | 0.0% | 100.0% | 0.0% | 100.0% | 94.1% | 100.0% | 93.8% | 0.0% |
| Devstral 2 123B Instruct | 100 | 0.0% | 100.0% | 5.9% | 94.1% | 100.0% | 100.0% | 100.0% | 0.0% |
| Nemotron 3 Super | 100 | 0.0% | 100.0% | 5.9% | 94.1% | 100.0% | 94.1% | 100.0% | 1.0% |
| Llama 3.1 8B | 100 | 0.0% | 100.0% | 58.8% | 41.2% | 70.6% | 100.0% | 100.0% | 0.0% |

Interpretation:

- Across the latest reviewed full stress runs, VerityGate produced no unsupported accepted answers.
- Stronger models tend to extract cleaner claims, producing more `accept` decisions on grounded cases.
- Weaker models can still be useful, but they often over-extract irrelevant missing claims. VerityGate then returns `partial` or `needs_info` rather than silently accepting.
- The system is intentionally conservative. A lower `accept` rate can be acceptable when the alternative is accepting unsupported claims.
- Some remaining failures are verifier extraction/filtering issues rather than deterministic gate failures.

What to expect when using this repo:

- If the evidence directly supports the answer, stronger models should often produce `accept`.
- If the answer is partly supported, VerityGate should preserve supported claims and expose missing claims as `partial`.
- If evidence is missing, it should return `needs_info` rather than inventing an answer.
- If evidence conflicts, it should return `contradiction` when the conflict is extracted or deterministically detected.
- For prediction, recommendation, diagnosis, or causal-pressure questions, it should return `hypothesis` or `partial_hypothesis` instead of presenting speculation as fact.
- Runtime cost is higher than a single LLM call because each case uses a writer call and verifier call, followed by deterministic processing.

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
python src\report\report.py --pipeline results\ollama_stress_key_24.jsonl --dataset-version stress_key_24 --output results\ollama_stress_key_24_report.md
```

Full stress set:

```powershell
python run_pipeline_batched.py --cases data\stress_cases_v0.1.jsonl --output results\ollama_stress_v0.1.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\ollama_stress_v0.1.jsonl --dataset-version stress_v0.1 --output results\ollama_stress_v0.1_report.md
```

Full stress set with local Ollama/Gemma, assuming Ollama is running and the model name matches `ollama list`:

```powershell
$env:LLM_MODE="api"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:MODEL_NAME="gemma3:27b"

python run_pipeline_batched.py --cases data\stress_cases_v0.1.jsonl --output results\ollama_gemma31_stress_v0.1.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\ollama_gemma31_stress_v0.1.jsonl --dataset-version stress_v0.1 --output results\ollama_gemma31_stress_v0.1_report.md
```

Original 100-case development set:

```powershell
python run_pipeline_batched.py --cases data\gold_cases.jsonl --output results\dev_gold_pipeline.jsonl --delay 0 --skip-calibration
python src\report\report.py --pipeline results\dev_gold_pipeline.jsonl --dataset-version dev_v0.4 --output results\dev_gold_report.md
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

## License

VerityGate is released under the Apache License 2.0.

You can use, modify, and distribute it, including for commercial use, subject to the Apache-2.0 license terms.

This project is a lightweight research prototype. It does not provide legal, medical, financial, or compliance advice, and it should be evaluated carefully before production use.

VerityGate is a research harness. Treat results as evidence for design decisions, not as leaderboard claims.
