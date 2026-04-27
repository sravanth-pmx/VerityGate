"""Diagnostic: list all false-contradiction cases from pipeline results.

Usage:
  python -m src.diagnose_contradictions --input results_latest/verity_pipeline.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .schemas import PipelineResult


def load_pipeline_results(path: Path) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(PipelineResult.model_validate_json(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to verity_pipeline.jsonl")
    args = parser.parse_args()

    results = load_pipeline_results(Path(args.input))

    print("=" * 80)
    print("FALSE CONTRADICTION DIAGNOSTIC")
    print("Cases where category != 'contradiction' but gate decision == 'contradiction'")
    print("=" * 80)

    false_contras: list[PipelineResult] = []
    for r in results:
        if r.category != "contradiction" and r.gate_output and r.gate_output.decision == "contradiction":
            false_contras.append(r)

    if not false_contras:
        print("\n✅ No false contradictions found!\n")
        return

    print(f"\nFound {len(false_contras)} false contradiction(s):\n")

    for r in false_contras:
        print(f"{'─' * 80}")
        print(f"case_id:      {r.case_id}")
        print(f"category:     {r.category}")
        print(f"pressure:     {r.pressure_level}")
        print(f"question:     {r.question}")
        print(f"\nDraft answer (first 200 chars):")
        print(f"  {r.draft_answer[:200]}...")

        if r.verifier_output:
            contra_claims = [c for c in r.verifier_output.claims
                            if c.label == "CONTRADICTS_EVIDENCE"]
            print(f"\nCONTRADICTS_EVIDENCE claims ({len(contra_claims)}):")
            for c in contra_claims:
                print(f"  • claim_text: {c.claim_text}")
                for ptr in c.evidence_pointers:
                    print(f"    pointer: {ptr.span_id} | preview: {ptr.text_preview}")

        if r.gate_output:
            print(f"\nFinal answer (first 300 chars):")
            print(f"  {r.gate_output.final_answer[:300]}...")

        print()

    # Summary by category
    from collections import Counter
    cats = Counter(r.category for r in false_contras)
    print(f"{'─' * 80}")
    print("Summary by category:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    print("=" * 80)


if __name__ == "__main__":
    main()
