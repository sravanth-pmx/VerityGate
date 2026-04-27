"""Tests for schemas.py — validation rules."""

import pytest
from pydantic import ValidationError
from src.schemas import (
    EvidencePointer,
    VerifiedClaim,
    VerifierOutput,
)


def _pointer(sid="span_0"):
    return EvidencePointer(span_id=sid, start_char=0, end_char=10, text_preview="mock")


class TestVerifiedClaimValidation:
    def test_supported_with_pointers_ok(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="test", claim_kind="fact",
            label="SUPPORTED", evidence_pointers=[_pointer()],
        )
        assert c.label == "SUPPORTED"

    def test_supported_without_pointers_fails(self):
        with pytest.raises(ValidationError, match="evidence_pointers"):
            VerifiedClaim(
                claim_id="c1", claim_text="test", claim_kind="fact",
                label="SUPPORTED", evidence_pointers=[],
            )

    def test_contradicts_evidence_with_pointers_ok(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="conflict", claim_kind="fact",
            label="CONTRADICTS_EVIDENCE", evidence_pointers=[_pointer()],
        )
        assert c.label == "CONTRADICTS_EVIDENCE"

    def test_contradicts_evidence_without_pointers_fails(self):
        """CONTRADICTS_EVIDENCE must have evidence_pointers."""
        with pytest.raises(ValidationError, match="evidence_pointers"):
            VerifiedClaim(
                claim_id="c1", claim_text="conflict", claim_kind="fact",
                label="CONTRADICTS_EVIDENCE", evidence_pointers=[],
            )

    def test_not_in_evidence_is_valid(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="test", claim_kind="fact",
            label="NOT_IN_EVIDENCE",
        )
        assert c.label == "NOT_IN_EVIDENCE"

    def test_unsupported_without_pointers_ok(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="test", claim_kind="fact",
            label="UNSUPPORTED",
        )
        assert c.evidence_pointers == []

    def test_needs_info_without_pointers_ok(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="test", claim_kind="fact",
            label="NEEDS_INFO",
        )
        assert c.evidence_pointers == []

    def test_invalid_label_fails(self):
        with pytest.raises(ValidationError):
            VerifiedClaim(
                claim_id="c1", claim_text="test", claim_kind="fact",
                label="BOGUS_LABEL",
            )


class TestVerifierOutput:
    def test_parse_error_with_empty_claims(self):
        vo = VerifierOutput(claims=[], parse_error=True, raw_response_preview="broken json")
        assert vo.parse_error is True
        assert vo.claims == []
        assert vo.raw_response_preview == "broken json"

    def test_normal_output(self):
        vo = VerifierOutput(
            claims=[
                VerifiedClaim(
                    claim_id="c1", claim_text="test", claim_kind="fact",
                    label="SUPPORTED", evidence_pointers=[_pointer()],
                )
            ]
        )
        assert vo.parse_error is False
        assert len(vo.claims) == 1

    def test_filter_stats_default_empty(self):
        vo = VerifierOutput(claims=[])
        assert vo.filter_stats == {}

    def test_filter_stats_stored(self):
        vo = VerifierOutput(
            claims=[],
            filter_stats={
                "pre_total": 5, "pre_meta_removed": 1,
                "pre_dedup_removed": 1, "pre_total_out": 3,
                "post_total": 3, "post_meta_removed": 0,
                "post_dedup_removed": 0, "post_irrelevant_removed": 1,
                "post_total_out": 2,
            }
        )
        assert vo.filter_stats["pre_total"] == 5
        assert vo.filter_stats["post_irrelevant_removed"] == 1
