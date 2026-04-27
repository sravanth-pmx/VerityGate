"""Validate gold_cases.jsonl — structural checks before inference runs.

Catches data issues early so we don't waste API calls on malformed cases.

Run: python -m src.validate_gold_cases
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .schemas import GoldCase
from . import config

_VALID_CATEGORIES = frozenset({
    "grounded", "missing_info", "contradiction",
    "pressure", "filler_trap", "partial_answer",
})


def validate_gold_cases(path: Path | None = None) -> tuple[bool, list[str], list[GoldCase]]:
    """Validate gold_cases.jsonl.

    Returns: (all_passed, list_of_errors, list_of_parsed_cases)
    """
    p = path or config.GOLD_CASES_PATH
    errors: list[str] = []
    cases: list[GoldCase] = []

    if not p.exists():
        errors.append(f"File not found: {p}")
        return False, errors, cases

    # ── Parse all rows ────────────────────────────────────────────────
    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = GoldCase.model_validate_json(line)
                cases.append(c)
            except Exception as exc:
                errors.append(f"Line {i}: parse error — {exc}")

    if not cases:
        errors.append("No valid cases found.")
        return False, errors, cases

    # ── Unique IDs ────────────────────────────────────────────────────
    ids = [c.id for c in cases]
    dupes = [x for x in set(ids) if ids.count(x) > 1]
    if dupes:
        errors.append(f"Duplicate case IDs: {dupes}")

    # ── Per-case validation ───────────────────────────────────────────
    for c in cases:
        prefix = f"{c.id}"

        # Category validity (already enforced by Pydantic Literal, but be explicit)
        if c.category not in _VALID_CATEGORIES:
            errors.append(f"{prefix}: invalid category '{c.category}'")

        # Non-empty fields
        if not c.question.strip():
            errors.append(f"{prefix}: empty question")
        if not c.evidence_text.strip():
            errors.append(f"{prefix}: empty evidence_text")

        # Pressure level
        if c.pressure_level not in (0, 1):
            errors.append(f"{prefix}: pressure_level must be 0 or 1, got {c.pressure_level}")

        # Category-specific expected fields
        if c.category == "grounded" and not c.expected_supported_claims:
            errors.append(f"{prefix}: grounded case has no expected_supported_claims")

        if c.category in ("missing_info", "filler_trap") and not c.expected_unknowns:
            errors.append(f"{prefix}: {c.category} case has no expected_unknowns")

        if c.category == "contradiction" and not c.expected_contradictions:
            errors.append(f"{prefix}: contradiction case has no expected_contradictions")

        if c.category == "pressure" and not c.expected_unknowns:
            errors.append(f"{prefix}: pressure case has no expected_unknowns")

        if c.category == "pressure" and c.pressure_level != 1:
            errors.append(f"{prefix}: pressure case should have pressure_level=1, got {c.pressure_level}")

        if c.category != "pressure" and c.pressure_level == 1:
            errors.append(f"{prefix}: non-pressure case has pressure_level=1")

        # Partial answer should have BOTH supported and unknowns
        if c.category == "partial_answer":
            if not c.expected_supported_claims:
                errors.append(f"{prefix}: partial_answer has no expected_supported_claims")
            if not c.expected_unknowns:
                errors.append(f"{prefix}: partial_answer has no expected_unknowns")

    all_passed = len(errors) == 0
    return all_passed, errors, cases


def print_summary(cases: list[GoldCase], errors: list[str], dataset_version: str = "") -> None:
    """Print a human-readable summary."""
    cats = Counter(c.category for c in cases)
    ev_lens = [len(c.evidence_text) for c in cases]
    pressure_count = sum(1 for c in cases if c.pressure_level == 1)

    print(f"{'─' * 50}")
    print(f"Gold Cases Validation Summary")
    print(f"{'─' * 50}")
    print(f"Dataset version: {dataset_version or config.DATASET_VERSION}")
    print(f"Status: {'DEV SET' if (dataset_version or config.DATASET_VERSION).startswith('dev') else 'UNKNOWN'}")
    print(f"Total cases: {len(cases)}")
    print()
    print("Category distribution:")
    for cat in sorted(_VALID_CATEGORIES):
        print(f"  {cat:20s}: {cats.get(cat, 0)}")
    print()
    print(f"Evidence length: min={min(ev_lens)}, max={max(ev_lens)}, avg={sum(ev_lens)/len(ev_lens):.0f} chars")
    print(f"Pressure cases: {pressure_count}")
    print()

    if errors:
        print(f"❌ {len(errors)} error(s) found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ All validation checks passed.")
    print(f"{'─' * 50}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate gold cases")
    parser.add_argument("--cases", type=str, default=None, help="Path to gold_cases.jsonl")
    parser.add_argument("--dataset-version", type=str, default=config.DATASET_VERSION, help="Dataset version string")
    args = parser.parse_args()

    path = Path(args.cases) if args.cases else None
    passed, errors, cases = validate_gold_cases(path)
    print_summary(cases, errors, dataset_version=args.dataset_version)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
