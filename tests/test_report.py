"""Tests for report.py — v0.3.2 with per-category breakdown and strict metrics."""

import json
import tempfile
import pytest
from pathlib import Path

from src.report import generate_report
from src import config
from src.schemas import (
    BaselineResult, PipelineResult, VerifierOutput, GateOutput,
    VerifiedClaim, EvidencePointer,
)


def _pointer():
    return EvidencePointer(span_id="span_0", start_char=0, end_char=10, text_preview="mock")


def _write_jsonl(records: list, path: Path) -> None:
    with open(path, 'w') as f:
        for r in records:
            f.write(r.model_dump_json() + '\n')


def _make_baseline(case_id="t1", category="grounded", answer="test answer",
                   latency=100.0, pressure_level=0):
    return BaselineResult(
        case_id=case_id, category=category, question="q?",
        answer=answer, latency_ms=latency, pressure_level=pressure_level,
    )


def _make_pipeline(case_id="t1", category="grounded", decision="accept",
                   claims=None, latency=200.0, pressure_level=0,
                   parse_error=False, final_answer="answer"):
    if claims is None:
        claims = [VerifiedClaim(
            claim_id="c1", claim_text="test claim", claim_kind="fact",
            label="SUPPORTED", evidence_pointers=[_pointer()],
        )]
    vo = VerifierOutput(claims=claims, parse_error=parse_error)
    go = GateOutput(
        final_answer=final_answer, decision=decision,
        included_claims=[c.claim_text for c in claims if c.label == "SUPPORTED"],
        unknown_claims=[c.claim_text for c in claims if c.label in ("UNSUPPORTED", "NEEDS_INFO", "NOT_IN_EVIDENCE")],
        contradicted_claims=[c.claim_text for c in claims if c.label == "CONTRADICTS_EVIDENCE"],
        hypothesis_claims=[],
    )
    return PipelineResult(
        case_id=case_id, category=category, question="q?",
        draft_answer="draft", pressure_level=pressure_level,
        verifier_output=vo, gate_output=go, latency_ms=latency,
    )


class TestReportGeneration:
    """Report should generate without crashing and contain key metrics."""

    def test_report_does_not_crash(self, tmp_path):
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        _write_jsonl([_make_baseline()], normal_path)
        _write_jsonl([_make_baseline()], honesty_path)
        _write_jsonl([_make_pipeline()], pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_key_metrics(self, tmp_path):
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        _write_jsonl([_make_baseline()], normal_path)
        _write_jsonl([_make_baseline()], honesty_path)
        _write_jsonl([_make_pipeline()], pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "parse_error_rate" in report
        assert "false_contradiction_rate" in report
        assert "grounded_accept_rate" in report
        assert "latency_p50_ms" in report
        assert "latency_p95_ms" in report

    def test_report_with_empty_files(self, tmp_path):
        """Report should handle empty result files gracefully."""
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        normal_path.write_text("")
        honesty_path.write_text("")
        pipeline_path.write_text("")

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "Cases: normal=0" in report

    def test_report_with_missing_files(self, tmp_path):
        """Report should handle missing files gracefully."""
        report = generate_report(
            tmp_path / "missing_normal.jsonl",
            tmp_path / "missing_honesty.jsonl",
            tmp_path / "missing_pipeline.jsonl",
        )
        assert "Cases: normal=0" in report


class TestReportPerCategory:
    """v0.3.2: per-category breakdown must appear in report."""

    def test_per_category_section_exists(self, tmp_path):
        pipe = _make_pipeline(category="grounded")
        pipe_path = tmp_path / "pipeline.jsonl"
        _write_jsonl([pipe], pipe_path)
        report = generate_report(
            pipeline_path=pipe_path,
            dataset_version="test_v1",
        )
        assert "## Per-Category Breakdown (Pipeline)" in report
        assert "grounded" in report

    def test_dataset_version_in_report(self, tmp_path):
        pipe = _make_pipeline()
        pipe_path = tmp_path / "pipeline.jsonl"
        _write_jsonl([pipe], pipe_path)
        report = generate_report(pipeline_path=pipe_path, dataset_version="dev_v0.99")
        assert "dev_v0.99" in report

    def test_dataset_version_defaults_to_config(self, tmp_path):
        pipe = _make_pipeline()
        pipe_path = tmp_path / "pipeline.jsonl"
        _write_jsonl([pipe], pipe_path)
        report = generate_report(pipeline_path=pipe_path)
        assert config.DATASET_VERSION in report
        assert "development benchmark" in report


class TestReportStrictMetrics:
    """v0.3.2: unsupported_claim_rate_among_accepts must be calculated correctly."""

    def test_unsupported_claim_rate_among_accepts(self, tmp_path):
        pipe = _make_pipeline(
            decision="accept",
            claims=[
                VerifiedClaim(
                    claim_id="c1", claim_text="supported claim",
                    claim_kind="fact", label="SUPPORTED",
                    evidence_pointers=[_pointer()],
                ),
                VerifiedClaim(
                    claim_id="c2", claim_text="unsupported claim",
                    claim_kind="fact", label="UNSUPPORTED",
                    evidence_pointers=[],
                ),
            ],
        )
        pipe_path = tmp_path / "pipeline.jsonl"
        _write_jsonl([pipe], pipe_path)
        report = generate_report(pipeline_path=pipe_path)
        assert "unsupported_claim_rate_among_accepts" in report
        assert "100.0%" in report


class TestReportWarnings:
    """Verify that warnings fire correctly."""

    def test_false_contradiction_warning(self, tmp_path):
        """false_contradiction_rate > 0 should trigger warning."""
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        _write_jsonl([_make_baseline()], normal_path)
        _write_jsonl([_make_baseline()], honesty_path)
        # Grounded case incorrectly flagged as contradiction
        _write_jsonl([_make_pipeline(
            category="grounded", decision="contradiction",
            claims=[VerifiedClaim(
                claim_id="c1", claim_text="conflict", claim_kind="fact",
                label="CONTRADICTS_EVIDENCE", evidence_pointers=[_pointer()],
            )],
        )], pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "false_contradiction_rate" in report
        assert "⚠️" in report
        assert "over-triggering" in report

    def test_latency_warning(self, tmp_path):
        """Pipeline p50 > 2x honesty p50 should trigger warning."""
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        _write_jsonl([_make_baseline(latency=1000.0)], normal_path)
        _write_jsonl([_make_baseline(latency=1000.0)], honesty_path)
        # Pipeline is 5x slower
        _write_jsonl([_make_pipeline(latency=5000.0)], pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "latency" in report.lower()
        assert "⚠️" in report

    def test_no_warnings_clean_run(self, tmp_path):
        """Clean results across all categories should show no warnings."""
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        baselines = [
            _make_baseline("c1", "grounded", latency=1000.0),
            _make_baseline("c2", "contradiction", latency=1000.0),
            _make_baseline("c3", "partial_answer", latency=1000.0),
        ]
        _write_jsonl(baselines, normal_path)
        _write_jsonl(baselines, honesty_path)

        pipelines = [
            _make_pipeline("c1", "grounded", "accept", latency=1500.0),
            _make_pipeline("c2", "contradiction", "contradiction",
                          claims=[VerifiedClaim(
                              claim_id="c1", claim_text="conflict", claim_kind="fact",
                              label="CONTRADICTS_EVIDENCE", evidence_pointers=[_pointer()],
                          )], latency=1500.0),
            _make_pipeline("c3", "partial_answer", "partial",
                          claims=[
                              VerifiedClaim(claim_id="c1", claim_text="known", claim_kind="fact",
                                           label="SUPPORTED", evidence_pointers=[_pointer()]),
                              VerifiedClaim(claim_id="c2", claim_text="unknown", claim_kind="fact",
                                           label="NOT_IN_EVIDENCE"),
                          ],
                          final_answer="What I can verify:\n• known\n\nWhat I cannot verify:\n• unknown",
                          latency=1500.0),
        ]
        _write_jsonl(pipelines, pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "✅ No warnings." in report


class TestReportMultipleCases:
    """Report with multiple cases across categories."""

    def test_multi_category_report(self, tmp_path):
        normal_path = tmp_path / "normal.jsonl"
        honesty_path = tmp_path / "honesty.jsonl"
        pipeline_path = tmp_path / "pipeline.jsonl"

        baselines = [
            _make_baseline("c1", "grounded", latency=100),
            _make_baseline("c2", "missing_info", answer="I don't know", latency=100),
            _make_baseline("c3", "contradiction", answer="There's a conflict", latency=100),
        ]
        _write_jsonl(baselines, normal_path)
        _write_jsonl(baselines, honesty_path)

        pipelines = [
            _make_pipeline("c1", "grounded", "accept", latency=200),
            _make_pipeline("c2", "missing_info", "needs_info",
                          claims=[VerifiedClaim(claim_id="c1", claim_text="unknown",
                                  claim_kind="fact", label="NOT_IN_EVIDENCE")],
                          latency=200),
            _make_pipeline("c3", "contradiction", "contradiction",
                          claims=[VerifiedClaim(claim_id="c1", claim_text="conflict",
                                  claim_kind="fact", label="CONTRADICTS_EVIDENCE",
                                  evidence_pointers=[_pointer()])],
                          latency=200),
        ]
        _write_jsonl(pipelines, pipeline_path)

        report = generate_report(normal_path, honesty_path, pipeline_path)
        assert "Cases: normal=3" in report
        assert "pipeline=3" in report
