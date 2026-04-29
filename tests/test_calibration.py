"""Tests for calibration.py — v0.3 table-format probes."""

import pytest
from src.report.calibration import run_calibration, _LINE_RE, _VALID_LABELS


class TestCalibrationProbeFormat:
    """Verify that calibration probes use table format, not JSON."""

    def test_line_regex_matches_standard_table(self):
        line = "1. SUPPORTED | The car is red | span_0"
        m = _LINE_RE.match(line)
        assert m is not None
        assert m.group(1).upper() == "SUPPORTED"

    def test_line_regex_matches_not_in_evidence(self):
        line = "2. NOT_IN_EVIDENCE | Price is not mentioned | none"
        m = _LINE_RE.match(line)
        assert m is not None
        assert m.group(1).upper() == "NOT_IN_EVIDENCE"

    def test_line_regex_matches_dash_format(self):
        line = "- CONTRADICTS_EVIDENCE | Two values conflict | span_1"
        m = _LINE_RE.match(line)
        assert m is not None
        assert m.group(1).upper() == "CONTRADICTS_EVIDENCE"

    def test_line_regex_no_match_plain_text(self):
        line = "This is just plain text without any table format."
        m = _LINE_RE.match(line)
        assert m is None

    def test_valid_labels_match_verifier(self):
        """Calibration should check the same labels as the verifier."""
        from src.verifier import _VALID_LABELS as verifier_labels
        assert _VALID_LABELS == verifier_labels

    def test_mock_mode_passes(self):
        """Calibration in mock mode should pass (mock returns table format)."""
        import os
        os.environ["LLM_MODE"] = "mock"
        passed, warnings = run_calibration()
        assert passed
        os.environ["LLM_MODE"] = "mock"  # ensure cleanup


class TestCalibrationProbeContent:
    """Verify probe structure matches the actual verifier prompt format."""

    def test_probes_use_span_bracket_format(self):
        """Probes should use [span_0] format matching BATCH_USER in verifier."""
        from report.calibration import _PROBES
        for probe in _PROBES:
            assert "[span_0]" in probe["user"]

    def test_probe_system_mentions_table(self):
        """System prompt should ask for table format, not JSON."""
        from report.calibration import _PROBE_SYSTEM
        assert "table" in _PROBE_SYSTEM.lower()
        assert "json" not in _PROBE_SYSTEM.lower()
