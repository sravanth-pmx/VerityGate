"""Tests for validate_gold_cases.py — structural validation of gold cases."""

import json
import tempfile
import pytest
from pathlib import Path

from src.validate_gold_cases import validate_gold_cases


class TestValidateCurrentData:
    """Validate the actual data/gold_cases.jsonl we ship."""

    def test_all_cases_parse(self):
        passed, errors, cases = validate_gold_cases()
        assert passed, f"Validation errors: {errors}"
        assert len(cases) == 100

    def test_no_duplicate_ids(self):
        _, _, cases = validate_gold_cases()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_category_distribution(self):
        _, _, cases = validate_gold_cases()
        from collections import Counter
        cats = Counter(c.category for c in cases)
        # Every category has at least 10 cases
        for cat in ["grounded", "missing_info", "contradiction", "pressure", "filler_trap", "partial_answer"]:
            assert cats[cat] >= 10, f"{cat} only has {cats[cat]} cases"

    def test_pressure_cases_have_pressure_level(self):
        _, _, cases = validate_gold_cases()
        for c in cases:
            if c.category == "pressure":
                assert c.pressure_level == 1, f"{c.id}: pressure case has pressure_level={c.pressure_level}"
            else:
                assert c.pressure_level == 0, f"{c.id}: non-pressure has pressure_level={c.pressure_level}"

    def test_no_empty_evidence(self):
        _, _, cases = validate_gold_cases()
        for c in cases:
            assert len(c.evidence_text.strip()) > 10, f"{c.id}: evidence too short"

    def test_no_empty_questions(self):
        _, _, cases = validate_gold_cases()
        for c in cases:
            assert len(c.question.strip()) > 5, f"{c.id}: question too short"


class TestValidatorCatchesErrors:
    """Verify the validator catches specific problems."""

    def _write_cases(self, cases: list[dict]) -> Path:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for c in cases:
            f.write(json.dumps(c) + '\n')
        f.close()
        return Path(f.name)

    def test_duplicate_ids_caught(self):
        path = self._write_cases([
            {"id": "x1", "category": "grounded", "question": "Q?", "evidence_text": "Evidence here.",
             "expected_supported_claims": ["claim"]},
            {"id": "x1", "category": "grounded", "question": "Q?", "evidence_text": "More evidence.",
             "expected_supported_claims": ["claim"]},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("Duplicate" in e for e in errors)

    def test_empty_evidence_caught(self):
        path = self._write_cases([
            {"id": "x1", "category": "grounded", "question": "Q?", "evidence_text": "",
             "expected_supported_claims": ["claim"]},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("empty evidence" in e for e in errors)

    def test_missing_expected_contradictions_caught(self):
        path = self._write_cases([
            {"id": "x1", "category": "contradiction", "question": "Q?",
             "evidence_text": "Source A says yes. Source B says no.",
             "expected_contradictions": []},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("expected_contradictions" in e for e in errors)

    def test_missing_expected_unknowns_for_missing_info(self):
        path = self._write_cases([
            {"id": "x1", "category": "missing_info", "question": "Q?",
             "evidence_text": "Some unrelated evidence.",
             "expected_unknowns": []},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("expected_unknowns" in e for e in errors)

    def test_missing_supported_claims_for_grounded(self):
        path = self._write_cases([
            {"id": "x1", "category": "grounded", "question": "Q?",
             "evidence_text": "Evidence here.",
             "expected_supported_claims": []},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("expected_supported_claims" in e for e in errors)

    def test_wrong_pressure_level_caught(self):
        path = self._write_cases([
            {"id": "x1", "category": "pressure", "question": "Should we?",
             "evidence_text": "Evidence here.", "pressure_level": 0,
             "expected_unknowns": ["answer"]},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("pressure_level" in e for e in errors)

    def test_partial_answer_missing_both_fields(self):
        path = self._write_cases([
            {"id": "x1", "category": "partial_answer", "question": "Q?",
             "evidence_text": "Evidence here.",
             "expected_supported_claims": [], "expected_unknowns": []},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert not passed
        assert any("partial_answer" in e for e in errors)

    def test_valid_case_passes(self):
        path = self._write_cases([
            {"id": "ok1", "category": "grounded", "question": "What time?",
             "evidence_text": "The meeting is at 3pm.",
             "expected_supported_claims": ["meeting at 3pm"]},
        ])
        passed, errors, _ = validate_gold_cases(path)
        assert passed, f"Unexpected errors: {errors}"

    def test_file_not_found(self):
        passed, errors, _ = validate_gold_cases(Path("/nonexistent/path.jsonl"))
        assert not passed
        assert any("not found" in e for e in errors)
