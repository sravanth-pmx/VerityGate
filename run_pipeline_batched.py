"""Batched pipeline runner with progress persistence.

Run this instead of pipeline_runner.py for large jobs.
Benefits:
  - Saves after EVERY case (resumable if interrupted)
  - Runs in configurable batch sizes
  - Tracks elapsed time and ETA
  - Handles rate-limit retries gracefully

Usage:
  python run_pipeline_batched.py --batch-size 25 --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src import config
from src.pipeline_runner import run_pipeline, load_gold_cases, WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
from src.llm_client import llm_call
from src.evidence_spans import split_evidence
from src.verifier import verify
from src.gate import apply_gate
from src.contradiction_checks import check_contradictions
from src.schemas import PipelineResult, GoldCase


def run_batched(
    cases: list[GoldCase],
    output_path: Path,
    batch_size: int = 25,
    delay: float = 0.5,
) -> list[PipelineResult]:
    """Run pipeline in batches, saving progress after each case."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed results
    completed: dict[str, PipelineResult] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = PipelineResult.model_validate_json(line)
                    completed[r.case_id] = r
        print(f"[batched] loaded {len(completed)} already-completed cases from {output_path}")

    results: list[PipelineResult] = list(completed.values())
    remaining = [c for c in cases if c.id not in completed]

    if not remaining:
        print(f"[batched] all {len(cases)} cases already complete!")
        return results

    print(f"[batched] running {len(remaining)}/{len(cases)} remaining cases")
    print(f"[batched] batch_size={batch_size} delay={delay}s")

    t0_global = time.perf_counter()
    for i, case in enumerate(remaining, 1):
        t0 = time.perf_counter()
        print(f"[batched] {i}/{len(remaining)}  case={case.id}  cat={case.category}  ", end="", flush=True)

        try:
            spans = split_evidence(case.evidence_text)
            user_msg = WRITER_USER_TEMPLATE.format(
                question=case.question,
                evidence_text=case.evidence_text,
            )
            draft = llm_call(WRITER_SYSTEM_PROMPT, user_msg)

            verifier_out = verify(
                question=case.question,
                draft_answer=draft,
                spans=spans,
                pressure_level=case.pressure_level,
                case_id=case.id,
            )

            conflict_result = check_contradictions(spans, case.question)
            # v0.4: Only FORCED contradictions (status-pair) gate to contradiction.
            if conflict_result.forced:
                existing_ids = {c.claim_id for c in verifier_out.claims}
                for dc in conflict_result.forced:
                    if dc.claim_id not in existing_ids:
                        verifier_out.claims.append(dc)
            if conflict_result.possible:
                pc_texts = [pc.claim_text for pc in conflict_result.possible]
                verifier_out.filter_stats["possible_conflicts"] = pc_texts

            gate_out = apply_gate(
                question=case.question,
                draft_answer=draft,
                verifier_output=verifier_out,
                pressure_level=case.pressure_level,
                spans=spans,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            r = PipelineResult(
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
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            r = PipelineResult(
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

        # Append immediately (resumable)
        with open(output_path, "a") as f:
            f.write(r.model_dump_json() + "\n")
        results.append(r)

        elapsed_s = time.perf_counter() - t0_global
        eta_s = (elapsed_s / i) * (len(remaining) - i)
        print(f"decision={r.gate_output.decision if r.gate_output else 'ERROR'}  "
              f"time={r.latency_ms:.0f}ms  "
              f"elapsed={elapsed_s/60:.1f}min  eta={eta_s/60:.1f}min")

        # Small delay to avoid hammering API
        if delay > 0:
            time.sleep(delay)

    print(f"[batched] done! {len(results)} total cases saved → {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/verity_pipeline_v0.4.jsonl")
    parser.add_argument("--batch-size", type=int, default=25, help="Print progress every N cases")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between cases")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else config.GOLD_CASES_PATH
    output_path = Path(args.output)

    # Set delay from arg (override env)
    os.environ["LLM_CALL_DELAY"] = str(args.delay)

    cases = load_gold_cases(cases_path)
    print(f"[batched] loaded {len(cases)} cases from {cases_path}")

    run_batched(cases, output_path, batch_size=args.batch_size, delay=args.delay)


if __name__ == "__main__":
    main()
