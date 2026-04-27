"""Diagnostic: list missing_info/filler_trap cases with no NOT_IN_EVIDENCE label.

Usage:
  python -m src.diagnose_not_in_evidence --input results_latest/verity_pipeline.jsonl
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
    print("NOT_IN_EVIDENCE LABEL AUDIT")
    print("Cases where category in (missing_info, filler_trap) but no NOT_IN_EVIDENCE label")
    print("=" * 80)

    audit_cases: list[PipelineResult] = []
    for r in results:
        if r.category in ("missing_info", "filler_trap"):
            if r.verifier_output:
                nie = any(c.label == "NOT_IN_EVIDENCE" for c in r.verifier_output.claims)
                if not nie:
                    audit_cases.append(r)

    if not audit_cases:
        print("\n✅ All missing_info/filler_trap cases have NOT_IN_EVIDENCE label!\n")
        return

    print(f"\nFound {len(audit_cases)} case(s) without NOT_IN_EVIDENCE:\n")

    for r in audit_cases:
        print(f"{'─' * 80}")
        print(f"case_id:      {r.case_id}")
        print(f"category:     {r.category}")
        print(f"question:     {r.question}")
        print(f"pressure:     {r.pressure_level}")
        print(f"\nDraft answer (first 200 chars):")
        print(f"  {r.draft_answer[:200]}...")

        if r.verifier_output:
            print(f"\nAll claims ({len(r.verifier_output.claims)}):")
            for c in r.verifier_output.claims:
                print(f"  [{c.label:<20}] {c.claim_text[:80]}...")

        if r.gate_output:
            print(f"\nGate decision: {r.gate_output.decision}")
            print(f"Final answer (first 300 chars):")
            print(f"  {r.gate_output.final_answer[:300]}...")

        print()

    from collections import Counter
    cats = Counter(r.category for r in audit_cases)
    print(f"{'─' * 80}")
    print("Summary by category:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    print("=" * 80)


if __name__ == "__main__":
    main()
