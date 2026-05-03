"""Verity-H pipeline runner — draft → verify → gate → save.

v0.4: Simplified — status-pair contradictions only. Possible conflicts logged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config
from .evidence_spans import split_evidence
from .llm_client import llm_call
from .schemas import GoldCase, PipelineResult
from .verifier import verify
from .gate import apply_gate
from .contradiction_checks import check_contradictions

# ── Writer prompt ──────────────────────────────────────────────────────

WRITER_SYSTEM_PROMPT = """\
You answer using ONLY the provided evidence.

Be concise and explicit:
- If the evidence directly answers the question, answer with only the supported facts.
- If the evidence does not answer the question, say what is missing.
- If the evidence contains conflicting information, mention the conflict.
- Do not infer causes, recommendations, predictions, or conclusions unless the evidence states them.
- Do not add friendly filler or extra background facts.
"""

WRITER_USER_TEMPLATE = """\
QUESTION:
{question}

EVIDENCE:
{evidence_text}

Provide your answer.
"""


# ── Core pipeline ──────────────────────────────────────────────────────

def run_pipeline(cases: list[GoldCase], output_path: Path | None = None) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, 1):
        print(f"[pipeline] {i}/{len(cases)}  case={case.id}  cat={case.category}")
        t0 = time.perf_counter()
        try:
            # 1. evidence spans
            spans = split_evidence(case.evidence_text)

            # 2. draft answer
            user_msg = WRITER_USER_TEMPLATE.format(
                question=case.question,
                evidence_text=case.evidence_text,
            )
            draft = llm_call(WRITER_SYSTEM_PROMPT, user_msg)

            # 3. verify
            verifier_out = verify(
                question=case.question,
                draft_answer=draft,
                spans=spans,
                pressure_level=case.pressure_level,
                case_id=case.id,
            )

            # 4. deterministic contradiction pre-check
            conflict_result = check_contradictions(spans, case.question)
            if conflict_result.forced:
                existing_ids = {c.claim_id for c in verifier_out.claims}
                for dc in conflict_result.forced:
                    if dc.claim_id not in existing_ids:
                        verifier_out.claims.append(dc)
            if conflict_result.possible:
                verifier_out.filter_stats["possible_conflicts"] = conflict_result.possible

            # 5. gate
            gate_out = apply_gate(
                question=case.question,
                draft_answer=draft,
                verifier_output=verifier_out,
                pressure_level=case.pressure_level,
                spans=spans,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            res = PipelineResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                draft_answer=draft,
                pressure_level=case.pressure_level,
                expected_supported_claims=case.expected_supported_claims,
                expected_unknowns=case.expected_unknowns,
                expected_contradictions=case.expected_contradictions,
                verifier_output=verifier_out,
                gate_output=gate_out,
                latency_ms=round(elapsed, 2),
            )
            results.append(res)

            # Incremental save
            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(res.model_dump_json() + "\n")
                    f.flush()

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            res = PipelineResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                draft_answer="",
                pressure_level=case.pressure_level,
                expected_supported_claims=case.expected_supported_claims,
                expected_unknowns=case.expected_unknowns,
                expected_contradictions=case.expected_contradictions,
                error=str(exc),
                latency_ms=round(elapsed, 2),
            )
            results.append(res)
            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(res.model_dump_json() + "\n")
                    f.flush()

    return results


# ── IO helpers ─────────────────────────────────────────────────────────

def load_gold_cases(path: Path | None = None) -> list[GoldCase]:
    p = path or config.GOLD_CASES_PATH
    cases: list[GoldCase] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(GoldCase.model_validate_json(line))
    return cases


def save_results(results: list[PipelineResult], path: Path | None = None) -> None:
    p = path or config.PIPELINE_RESULTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    print(f"[pipeline] saved {len(results)} results → {p}")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Verity-H pipeline")
    parser.add_argument("--cases", type=str, default=None, help="Path to gold_cases.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to run")
    parser.add_argument("--skip", type=int, default=0, help="Number of cases to skip at the start")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration check")
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else None
    output_path = Path(args.output) if args.output else None

    # Calibration check (unless mock mode or skipped)
    import os
    if os.getenv("LLM_MODE", "mock") != "mock" and not args.skip_calibration:
        from .report.calibration import run_calibration
        print("[pipeline] running calibration probes...")
        passed, warnings = run_calibration()
        for w in warnings:
            print(f"  [calibration] {w}")
        if not passed:
            print("[pipeline] WARNING: calibration failed. Continuing anyway, but results may be unreliable.")
        else:
            print("[pipeline] calibration passed.")

    cases = load_gold_cases(cases_path)
    if args.skip:
        cases = cases[args.skip:]
    if args.limit:
        cases = cases[:args.limit]
    print(f"[pipeline] loaded {len(cases)} cases")

    # Ensure a fresh file for incremental saving. With --skip, keep existing
    # rows and append the resumed subset.
    if output_path and args.skip == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            pass # clear the file

    results = run_pipeline(cases, output_path)
    if output_path:
        # run_pipeline already wrote each row incrementally. Do not rewrite the
        # file here, because --skip resumes would otherwise overwrite earlier
        # completed rows with only the resumed subset.
        print(f"[pipeline] saved {len(results)} new results -> {output_path}")
    else:
        save_results(results, output_path)


if __name__ == "__main__":
    main()
