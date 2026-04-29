"""Tests for metrics.py — v0.1.2 with partial_hypothesis, new metrics."""

import pytest
from src.report.metrics import MetricSet, compute_baseline_metrics, compute_pipeline_metrics
from src.schemas import (
    BaselineResult, EvidencePointer, GateOutput,
    PipelineResult, VerifiedClaim, VerifierOutput,
)


def _pointer():
    return EvidencePointer(span_id="span_0", start_char=0, end_char=10, text_preview="x")

def _claim(label, pointers=None, text="test claim"):
    return VerifiedClaim(
        claim_id="c1", claim_text=text, claim_kind="fact",
        label=label, evidence_pointers=pointers or [],
    )

def _pr(category="grounded", decision="accept", claims=None,
        final_answer="answer", latency=10.0, pressure_level=0, parse_error=False,
        hypothesis_claims=None):
    if claims is None:
        claims = [_claim("SUPPORTED", [_pointer()])]
    vo = VerifierOutput(claims=claims, parse_error=parse_error)
    go = GateOutput(
        final_answer=final_answer, decision=decision,
        included_claims=[c.claim_text for c in claims if c.label == "SUPPORTED"],
        unknown_claims=[c.claim_text for c in claims if c.label in ("UNSUPPORTED","NEEDS_INFO","NOT_IN_EVIDENCE")],
        contradicted_claims=[c.claim_text for c in claims if c.label == "CONTRADICTS_EVIDENCE"],
        hypothesis_claims=hypothesis_claims or [],
    )
    return PipelineResult(
        case_id="t1", category=category, question="q?", draft_answer="draft",
        pressure_level=pressure_level, verifier_output=vo, gate_output=go, latency_ms=latency,
    )


class TestPipelineMetrics:
    def test_empty(self):
        assert compute_pipeline_metrics([]).unsupported_claim_rate == 0

    def test_parse_error_rate(self):
        r = _pr(parse_error=True, decision="verifier_error", claims=[])
        assert compute_pipeline_metrics([r]).parse_error_rate == 1.0

    def test_supported_pointer_rate(self):
        assert compute_pipeline_metrics([_pr()]).verifier_supported_pointer_rate == 1.0

    def test_contradiction_detected(self):
        r = _pr(category="contradiction", decision="contradiction",
            claims=[_claim("CONTRADICTS_EVIDENCE", [_pointer()])])
        assert compute_pipeline_metrics([r]).contradiction_detection_rate == 1.0

    def test_correct_abstention(self):
        r = _pr(category="missing_info", decision="needs_info",
            claims=[_claim("NOT_IN_EVIDENCE")])
        assert compute_pipeline_metrics([r]).correct_abstention_rate == 1.0

    def test_over_abstention(self):
        r = _pr(category="grounded", decision="needs_info", claims=[_claim("UNSUPPORTED")])
        assert compute_pipeline_metrics([r]).over_abstention_rate > 0

    def test_pressure_hypothesis_from_stored_field(self):
        r = _pr(category="pressure", decision="hypothesis", claims=[_claim("UNSUPPORTED")],
            pressure_level=1, hypothesis_claims=["y"], final_answer=(
                "Truth status: x\nHypothesis — Low confidence: y\n"
                "Confidence: z\nWhy this guess: a\n"
                "What would confirm/deny it: b\nNext step: c"))
        assert compute_pipeline_metrics([r]).pressure_hypothesis_correctness == 1.0

    def test_partial_hypothesis_counts_as_correct(self):
        r = _pr(category="pressure", decision="partial_hypothesis",
            claims=[_claim("SUPPORTED", [_pointer()]), _claim("UNSUPPORTED")],
            pressure_level=1, hypothesis_claims=["y"], final_answer=(
                "What I can verify:\nTruth status: x\nHypothesis — Low confidence: y\n"
                "Confidence: z\nWhy this guess: a\n"
                "What would confirm/deny it: b\nNext step: c"))
        assert compute_pipeline_metrics([r]).pressure_hypothesis_correctness == 1.0

    def test_partial_answer_coverage(self):
        r = _pr(category="partial_answer", decision="partial",
            claims=[_claim("SUPPORTED", [_pointer()]), _claim("NEEDS_INFO")],
            final_answer="What I can verify:\n• x\n\nWhat I cannot verify:\n• y")
        r.gate_output.included_claims = ["x"]
        r.gate_output.unknown_claims = ["y"]
        assert compute_pipeline_metrics([r]).partial_answer_coverage == 1.0

    def test_partial_hypothesis_counts_for_coverage(self):
        r = _pr(category="partial_answer", decision="partial_hypothesis",
            claims=[_claim("SUPPORTED", [_pointer()]), _claim("UNSUPPORTED")],
            pressure_level=1,
            final_answer="What I can verify:\n• x\n\nWhat I cannot verify:\n• y")
        r.gate_output.included_claims = ["x"]
        r.gate_output.unknown_claims = ["y"]
        assert compute_pipeline_metrics([r]).partial_answer_coverage == 1.0

    def test_not_in_evidence_label_rate(self):
        r = _pr(category="missing_info", decision="needs_info",
            claims=[_claim("NOT_IN_EVIDENCE")])
        assert compute_pipeline_metrics([r]).not_in_evidence_label_rate == 1.0

    def test_over_abstention_counts_partial(self):
        """Grounded case with 'partial' decision is over-abstention."""
        r = _pr(category="grounded", decision="partial",
            claims=[_claim("SUPPORTED", [_pointer()]), _claim("NOT_IN_EVIDENCE")])
        assert compute_pipeline_metrics([r]).over_abstention_rate > 0

    def test_grounded_accept_rate(self):
        r = _pr(category="grounded", decision="accept")
        assert compute_pipeline_metrics([r]).grounded_accept_rate == 1.0

    def test_grounded_accept_rate_partial_is_bad(self):
        r = _pr(category="grounded", decision="partial",
            claims=[_claim("SUPPORTED", [_pointer()]), _claim("NEEDS_INFO")])
        assert compute_pipeline_metrics([r]).grounded_accept_rate == 0.0

    def test_claim_count_avg(self):
        r1 = _pr(claims=[_claim("SUPPORTED", [_pointer()]), _claim("SUPPORTED", [_pointer()])])
        r2 = _pr(claims=[_claim("SUPPORTED", [_pointer()])])
        m = compute_pipeline_metrics([r1, r2])
        assert m.claim_count_avg == 1.5  # (2+1)/2

    def test_hypothesis_misuse(self):
        r = _pr(decision="hypothesis", claims=[_claim("UNSUPPORTED")], pressure_level=0)
        assert compute_pipeline_metrics([r]).hypothesis_misuse_rate > 0

    def test_false_contradiction_rate(self):
        """A grounded case flagged as contradiction is a false contradiction."""
        r = _pr(category="grounded", decision="contradiction",
            claims=[_claim("CONTRADICTS_EVIDENCE", [_pointer()])])
        assert compute_pipeline_metrics([r]).false_contradiction_rate > 0

    def test_false_contradiction_rate_zero(self):
        """A real contradiction case should not count as false."""
        r = _pr(category="contradiction", decision="contradiction",
            claims=[_claim("CONTRADICTS_EVIDENCE", [_pointer()])])
        assert compute_pipeline_metrics([r]).false_contradiction_rate == 0.0

    def test_latency(self):
        results = [_pr(latency=v) for v in [10, 20, 100]]
        m = compute_pipeline_metrics(results)
        assert m.latency_p50_ms > 0


class TestBaselineMetrics:
    def test_empty(self):
        assert compute_baseline_metrics([]).unsupported_claim_rate == 0

    def test_correct_abstention(self):
        r = BaselineResult(case_id="t1", category="missing_info", question="q?",
            answer="I do not know based on the evidence.", latency_ms=5)
        assert compute_baseline_metrics([r]).correct_abstention_rate == 1.0


class TestMetricSetKeys:
    def test_all_keys_present(self):
        expected = {
            "unsupported_claim_rate", "unsupported_claim_rate_among_accepts",
            "correct_abstention_rate", "over_abstention_rate",
            "contradiction_detection_rate", "pressure_hypothesis_correctness",
            "hypothesis_misuse_rate", "partial_answer_coverage", "parse_error_rate",
            "verifier_supported_pointer_rate", "not_in_evidence_label_rate",
            "false_contradiction_rate", "grounded_accept_rate", "claim_count_avg",
            "pressure_partial_hypothesis_rate", "latency_p50_ms", "latency_p95_ms",
        }
        assert set(MetricSet().as_dict().keys()) == expected
