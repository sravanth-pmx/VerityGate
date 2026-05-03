"""Tests for span_matcher.py — deterministic claim labeling."""

import pytest
from src.span_matcher import label_claim_against_spans, relabel_claims
from src.schemas import EvidencePointer, EvidenceSpan, VerifiedClaim


def _span(text, sid="span_0"):
    return EvidenceSpan(span_id=sid, text=text, start_char=0, end_char=len(text))

def _pointer():
    return EvidencePointer(span_id="span_0", start_char=0, end_char=10, text_preview="x")


class TestLabelClaimAgainstSpans:
    def test_exact_substring_supported(self):
        spans = [_span("The meeting is scheduled for Tuesday at 3pm.")]
        label, ptr, _ = label_claim_against_spans("The meeting is scheduled for Tuesday at 3pm", spans)
        assert label == "SUPPORTED"
        assert ptr is not None

    def test_fuzzy_match_supported(self):
        spans = [_span("HR records indicate 47 new employees were hired in Q1 2024.")]
        label, ptr, _ = label_claim_against_spans("47 new employees were hired in Q1 2024", spans)
        assert label == "SUPPORTED"
        assert ptr is not None

    def test_absence_phrase_not_in_evidence(self):
        spans = [_span("The company was founded in 2010.")]
        label, ptr, _ = label_claim_against_spans(
            "The evidence does not specify the stock price", spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    def test_deferral_not_in_evidence(self):
        spans = [_span("The budget has not been finalized.")]
        label, ptr, _ = label_claim_against_spans(
            "The budget has not been finalized", spans)
        assert label == "NOT_IN_EVIDENCE"

    def test_not_listed_not_in_evidence(self):
        spans = [_span("Current medications are not listed.")]
        label, ptr, _ = label_claim_against_spans(
            "Current medications are not listed in the evidence", spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    def test_not_documented_not_in_evidence(self):
        spans = [_span("The allergy section says not documented in transferred records.")]
        label, ptr, _ = label_claim_against_spans(
            "Allergies are not documented in transferred records", spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    def test_not_shown_not_in_evidence(self):
        spans = [_span("Shipping cost is waived but the original shipping fee is not shown.")]
        label, ptr, _ = label_claim_against_spans(
            "The original shipping fee is not shown", spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    @pytest.mark.parametrize("claim", [
        "Blood pressure was not recorded in the note",
        "The deposit amount is left blank",
        "The final grade column is not populated",
        "The judge assignment field reads unassigned",
        "The phone number is intentionally redacted",
        "Target CPA: TBD",
        "Lease end date: Missing from the abstract",
        "Battery life data is missing",
    ])
    def test_placeholder_values_not_in_evidence(self, claim):
        spans = [_span(claim)]
        label, ptr, _ = label_claim_against_spans(claim, spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    @pytest.mark.parametrize("claim", [
        "Remedy instructions will be mailed later",
        "Renewal premium will be calculated next quarter",
        "Renewal premium is not yet calculated",
        "Final exam date is not yet scheduled",
        "Reference feedback has not been collected",
        "The bacteria count is still being processed",
        "Battery life testing still pending",
    ])
    def test_deferral_values_not_in_evidence(self, claim):
        spans = [_span(claim)]
        label, ptr, _ = label_claim_against_spans(claim, spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    def test_no_slot_included_not_in_evidence(self):
        spans = [_span("No vote totals are included.")]
        label, ptr, _ = label_claim_against_spans(
            "No vote totals are included in the evidence", spans)
        assert label == "NOT_IN_EVIDENCE"
        assert ptr is None

    def test_supported_included_fact_stays_supported(self):
        spans = [_span("The package included a power adapter and cable.")]
        label, ptr, _ = label_claim_against_spans(
            "The package included a power adapter", spans)
        assert label == "SUPPORTED"
        assert ptr is not None

    def test_currently_restructured_not_in_evidence(self):
        spans = [_span("Personal training rates are currently being restructured.")]
        label, ptr, _ = label_claim_against_spans(
            "Personal training rates are currently being restructured", spans)
        assert label == "NOT_IN_EVIDENCE"

    def test_no_match_returns_empty(self):
        spans = [_span("The sky is blue.")]
        label, ptr, _ = label_claim_against_spans("The defendant is guilty", spans)
        assert label == ""  # can't determine — let LLM decide
        assert ptr is None

    def test_short_claim_not_matched(self):
        spans = [_span("The car is red and fast.")]
        label, ptr, _ = label_claim_against_spans("red", spans)
        assert label == ""  # too short for substring match


class TestRelabelClaims:
    def test_downgrade_supported_absence(self):
        claims = [VerifiedClaim(
            claim_id="c1", claim_text="The evidence does not mention the cause",
            claim_kind="fact", label="SUPPORTED", evidence_pointers=[_pointer()])]
        result = relabel_claims(claims, [])
        assert result[0].label == "NOT_IN_EVIDENCE"

    def test_upgrade_unsupported_span_match(self):
        spans = [_span("Fingerprints were found at the scene.")]
        claims = [VerifiedClaim(
            claim_id="c1", claim_text="Fingerprints were found at the scene",
            claim_kind="fact", label="UNSUPPORTED")]
        result = relabel_claims(claims, spans)
        assert result[0].label == "SUPPORTED"
        assert len(result[0].evidence_pointers) == 1

    def test_leave_correct_supported(self):
        spans = [_span("Revenue was $4.2 million.")]
        claims = [VerifiedClaim(
            claim_id="c1", claim_text="Revenue was $4.2 million",
            claim_kind="fact", label="SUPPORTED", evidence_pointers=[_pointer()])]
        result = relabel_claims(claims, spans)
        assert result[0].label == "SUPPORTED"

    def test_leave_correct_unsupported(self):
        spans = [_span("The sky is blue.")]
        claims = [VerifiedClaim(
            claim_id="c1", claim_text="The defendant is guilty",
            claim_kind="fact", label="UNSUPPORTED")]
        result = relabel_claims(claims, spans)
        assert result[0].label == "UNSUPPORTED"  # no span match, stays as-is


class TestNumericConsistency:
    """Test that fuzzy matches reject claims with wrong numbers."""

    def test_matching_numbers_supported(self):
        """Claim '12% growth' + span '12% from Q2' → SUPPORTED (numbers match)."""
        spans = [_span("Q3 revenue was $4.2 million, up 12% from Q2.")]
        label, ptr, _ = label_claim_against_spans(
            "Revenue was up 12% from Q2", spans)
        assert label == "SUPPORTED"

    def test_wrong_number_not_supported(self):
        """Claim '20% growth' + span '12% growth' → not SUPPORTED (numbers differ)."""
        spans = [_span("Q3 revenue was $4.2 million, up 12% from Q2.")]
        label, ptr, _ = label_claim_against_spans(
            "Revenue was up 20% from Q2", spans)
        # Should NOT be SUPPORTED because 20% ≠ 12%
        assert label != "SUPPORTED"

    def test_wrong_dollar_amount_not_supported(self):
        """Claim '$5.0 million' + span '$4.2 million' → not SUPPORTED."""
        spans = [_span("Q3 revenue was $4.2 million, up 12% from Q2.")]
        label, ptr, _ = label_claim_against_spans(
            "Q3 revenue was $5.0 million", spans)
        assert label != "SUPPORTED"

    def test_unstated_calculated_percentage_not_supported(self):
        spans = [_span("The exam was taken by 240 students. 186 students scored 70 or above.")]
        label, ptr, _ = label_claim_against_spans(
            "77.5% of students passed the exam", spans)
        assert label == "UNSUPPORTED"
        assert ptr is None

    def test_stated_percentage_still_supported(self):
        spans = [_span("77.5% of students passed the exam.")]
        label, ptr, _ = label_claim_against_spans(
            "77.5% of students passed the exam", spans)
        assert label == "SUPPORTED"

    def test_unstated_calculated_total_not_supported(self):
        spans = [_span(
            "The event note says 40 employees attended the morning session and "
            "35 attended the afternoon session. It also says 12 people attended both sessions."
        )]
        label, ptr, _ = label_claim_against_spans(
            "63 people attended in total", spans)
        assert label == "UNSUPPORTED"
        assert ptr is None

    def test_stated_total_still_supported(self):
        spans = [_span("The event report states that 63 people attended in total.")]
        label, ptr, _ = label_claim_against_spans(
            "63 people attended in total", spans)
        assert label == "SUPPORTED"

    def test_no_numbers_in_claim_still_matches(self):
        """Text-only claim with keyword match → SUPPORTED (no numbers to check)."""
        spans = [_span("The hiring manager was Sandra Lee.")]
        label, ptr, _ = label_claim_against_spans(
            "Sandra Lee was the hiring manager", spans)
        assert label == "SUPPORTED"
