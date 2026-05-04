"""Tests for gate.py — v0.1.2 with partial_hypothesis and punctuation fixes."""

import pytest
from src.gate import apply_gate
from src.schemas import (
    EvidencePointer, EvidenceSpan, VerifiedClaim, VerifierOutput,
)


def _span(sid="span_0", text="mock evidence text"):
    return EvidenceSpan(span_id=sid, text=text, start_char=0, end_char=len(text))

def _pointer(sid="span_0"):
    return EvidencePointer(span_id=sid, start_char=0, end_char=10, text_preview="mock")

def _claim(cid, label, kind="fact", pointers=None):
    return VerifiedClaim(
        claim_id=cid, claim_text=f"claim {cid}", claim_kind=kind,
        label=label, evidence_pointers=pointers or [], notes="",
    )


class TestAccept:
    def test_reconstructs_from_claims(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "SUPPORTED", pointers=[_pointer()]),
        ])
        out = apply_gate("q?", "RAW DRAFT", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "accept"
        assert "RAW DRAFT" not in out.final_answer
        assert "claim c1" in out.final_answer

    def test_no_double_punctuation(self):
        c = VerifiedClaim(
            claim_id="c1", claim_text="The meeting is at 3pm.",
            claim_kind="fact", label="SUPPORTED",
            evidence_pointers=[_pointer()],
        )
        vo = VerifierOutput(claims=[c])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert ".." not in out.final_answer


class TestContradiction:
    def test_detected(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "CONTRADICTS_EVIDENCE", pointers=[_pointer()]),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "contradiction"

    def test_overrides_pressure(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "CONTRADICTS_EVIDENCE", pointers=[_pointer()]),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=1, spans=[_span()])
        assert out.decision == "contradiction"

    def test_draft_conflict_with_no_claims_routes_contradiction(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "When did the CEO resign?",
            "The evidence contains conflicting information about the resignation date.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "contradiction"
        assert out.contradicted_claims == ["Draft answer indicated conflicting evidence."]

    def test_no_single_consistent_answer_routes_contradiction(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "SUPPORTED", pointers=[_pointer()]),
        ])
        out = apply_gate(
            "How many units were sold?",
            "The evidence does not provide a single, consistent answer.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "contradiction"

    def test_supported_status_opposites_route_contradiction(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Remote work is allowed for this role",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Remote work is not allowed for this role",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "Is remote work allowed for this role?",
            "Remote work is not allowed for this role.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "contradiction"
        assert out.contradicted_claims

    def test_supported_same_measurement_different_values_route_contradiction(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="The sodium level was 132 mmol/L",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="The sodium level was 142 mmol/L",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "What was the measured sodium level?",
            "132 mmol/L and 142 mmol/L.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "contradiction"
        assert out.contradicted_claims


class TestNeedsInfo:
    def test_pressure_0_no_hypothesis(self):
        vo = VerifierOutput(claims=[_claim("c1", "UNSUPPORTED")])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "needs_info"
        assert "hypothesis" not in out.final_answer.lower()
        assert "guess" not in out.final_answer.lower()

    def test_not_in_evidence(self):
        vo = VerifierOutput(claims=[_claim("c1", "NOT_IN_EVIDENCE")])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "needs_info"


class TestHypothesis:
    def test_at_pressure_1_speculative(self):
        """Pressure=1 + speculative question + unsupported → hypothesis."""
        vo = VerifierOutput(claims=[_claim("c1", "UNSUPPORTED")])
        out = apply_gate("Will the product launch be successful?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "hypothesis"
        for key in ["Truth status:", "Hypothesis", "Confidence:", "Next step:"]:
            assert key in out.final_answer

    def test_not_at_pressure_0(self):
        vo = VerifierOutput(claims=[_claim("c1", "UNSUPPORTED")])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision != "hypothesis"

    def test_pressure_1_non_speculative(self):
        """Pressure=1 but non-speculative question → needs_info, not hypothesis."""
        vo = VerifierOutput(claims=[_claim("c1", "UNSUPPORTED")])
        out = apply_gate("What is the stock price?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "needs_info"
        assert "hypothesis" not in out.final_answer.lower()
        assert "guess" not in out.final_answer.lower()

    def test_pressure_1_with_contradiction(self):
        """Pressure=1 + contradiction → contradiction, never hypothesis."""
        vo = VerifierOutput(claims=[
            _claim("c1", "CONTRADICTS_EVIDENCE", pointers=[_pointer()]),
            _claim("c2", "UNSUPPORTED"),
        ])
        out = apply_gate("Will the product launch be successful?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "contradiction"

    def test_pressure_1_with_verifier_error(self):
        """Pressure=1 + verifier_error → verifier_error, never hypothesis."""
        vo = VerifierOutput(claims=[], parse_error=True)
        out = apply_gate("Will the product launch be successful?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "verifier_error"


class TestPartialHypothesis:
    def test_pressure_1_speculative_with_supported_and_unknown(self):
        """Speculative pressure question + supported + unsupported → partial_hypothesis."""
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "UNSUPPORTED"),
        ])
        out = apply_gate("Will the product launch be successful?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "partial_hypothesis"
        assert "What I can verify" in out.final_answer
        assert "Truth status:" in out.final_answer
        assert "Hypothesis — Low confidence:" in out.final_answer
        assert len(out.included_claims) == 1
        assert len(out.hypothesis_claims) == 1

    def test_pressure_1_factual_with_supported_and_unknown(self):
        """Non-speculative pressure question + supported + unknown → partial (not hypothesis)."""
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "UNSUPPORTED"),
        ])
        out = apply_gate("What is the stock price?", "draft", vo,
                         pressure_level=1, spans=[_span()])
        assert out.decision == "partial"
        assert out.decision != "partial_hypothesis"

    def test_pressure_0_gives_partial_not_hypothesis(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "UNSUPPORTED"),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "partial"
        assert out.decision != "partial_hypothesis"


class TestPartial:
    def test_contains_both_sections(self):
        vo = VerifierOutput(claims=[
            _claim("c1", "SUPPORTED", pointers=[_pointer()]),
            _claim("c2", "NEEDS_INFO"),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "partial"
        assert "What I can verify" in out.final_answer
        assert "What I cannot verify" in out.final_answer

    def test_multislot_dropped_missing_slot_blocks_accept(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Medication: lisinopril",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Dosage: 10mg once daily",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "What are the patient's medication, dosage, and allergies?",
            "Medication: lisinopril\nDosage: 10mg once daily\nAllergies: not documented in transferred records",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "partial"
        assert out.unknown_claims

    def test_multislot_inline_missing_slot_blocks_accept(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Item price is $48.00",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Tax is $3.84",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c3", claim_text="Shipping cost is waived",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "What are the item price, tax, and shipping cost?",
            "Item price: $48.00 Tax: $3.84 Shipping cost: Waived (original fee not shown)",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "partial"
        assert out.unknown_claims

    def test_multislot_missing_data_blocks_accept(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="The laptop has an Intel i7 CPU",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="The laptop has 16GB RAM",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "What are the laptop CPU, RAM, and battery life?",
            "Intel i7 CPU, 16GB RAM. Battery life data is missing.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "partial"
        assert out.unknown_claims

    def test_computed_scalar_answer_blocks_accept_when_total_not_stated(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="40 employees attended the morning session",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="35 employees attended the afternoon session",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c3", claim_text="12 people attended both sessions",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        spans = [_span(
            text=(
                "The event note says 40 employees attended the morning session "
                "and 35 attended the afternoon session. It also says 12 people "
                "attended both sessions."
            )
        )]
        out = apply_gate(
            "How many people attended in total?",
            "Total unique attendees = 40 + 35 - 12 = 63",
            vo,
            pressure_level=0,
            spans=spans,
        )
        assert out.decision == "partial"
        assert out.unknown_claims

    def test_multislot_complete_slot_answer_can_accept(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Item price is $48.00",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Tax is $3.84",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
            VerifiedClaim(
                claim_id="c3", claim_text="Shipping cost is waived",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes="",
            ),
        ])
        out = apply_gate(
            "What are the item price, tax, and shipping cost?",
            "Item price: $48.00 Tax: $3.84 Shipping cost: Waived",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "accept"


class TestVerifierError:
    def test_parse_error(self):
        vo = VerifierOutput(claims=[], parse_error=True)
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "verifier_error"
        assert "hypothesis" not in out.final_answer.lower()
        assert "may still be correct" in out.final_answer

    def test_parse_error_at_pressure_1(self):
        vo = VerifierOutput(claims=[], parse_error=True)
        out = apply_gate("q?", "draft", vo, pressure_level=1, spans=[_span()])
        assert out.decision == "verifier_error"


class TestDedupPriority:
    def test_contradiction_wins_over_supported(self):
        """Two claims with same text but different labels — contradiction should win."""
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Budget is $5M",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes=""
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Budget is $5M",
                claim_kind="fact", label="CONTRADICTS_EVIDENCE",
                evidence_pointers=[_pointer()], notes=""
            ),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "contradiction"

    def test_unsupported_wins_over_supported(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Budget is $5M",
                claim_kind="fact", label="SUPPORTED",
                evidence_pointers=[_pointer()], notes=""
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Budget is $5M",
                claim_kind="fact", label="UNSUPPORTED",
                evidence_pointers=[], notes=""
            ),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "needs_info"

    def test_needs_info_wins_over_not_in_evidence(self):
        vo = VerifierOutput(claims=[
            VerifiedClaim(
                claim_id="c1", claim_text="Budget is $5M",
                claim_kind="fact", label="NEEDS_INFO",
                evidence_pointers=[], notes=""
            ),
            VerifiedClaim(
                claim_id="c2", claim_text="Budget is $5M",
                claim_kind="fact", label="NOT_IN_EVIDENCE",
                evidence_pointers=[], notes=""
            ),
        ])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        # NEEDS_INFO is higher priority than NOT_IN_EVIDENCE, so should be needs_info
        assert out.decision == "needs_info"


class TestFallback:
    def test_no_claims(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate("q?", "draft", vo, pressure_level=0, spans=[_span()])
        assert out.decision == "needs_info"

    def test_no_claims_missing_answer_preserved(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "What are the specifications?",
            "The evidence does not mention the requested specifications.",
            vo,
            pressure_level=0,
            spans=[_span()],
        )
        assert out.decision == "needs_info"
        assert out.unknown_claims

    def test_no_claims_missing_answer_under_pressure_routes_hypothesis(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "Will the product launch be successful?",
            "The evidence does not answer the question.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"
        assert out.hypothesis_claims

    def test_no_information_about_whether_preserved_under_pressure(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "Will artificial general intelligence be achieved by 2030?",
            "There is no information about whether AGI will be achieved by 2030.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"

    def test_no_claims_pending_analysis_under_pressure_routes_hypothesis(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "Is the new therapy effective?",
            "The trial summary reports a symptom-score reduction, but statistical significance analysis is pending.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"
        assert out.hypothesis_claims

    def test_no_claims_missing_scores_under_pressure_routes_hypothesis(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "Will the grant proposal be accepted?",
            "The internal review score is 87 out of 100. The funder's selection criteria and competitor scores are missing.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"
        assert out.hypothesis_claims

    def test_no_claims_terse_missing_under_pressure_routes_hypothesis(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "Should we approve the mortgage application?",
            "Missing: bank's approval threshold and underwriting exceptions.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"
        assert out.hypothesis_claims

    def test_no_claims_not_confirmed_under_pressure_routes_hypothesis(self):
        vo = VerifierOutput(claims=[])
        out = apply_gate(
            "What diagnosis best explains the symptoms?",
            "Missing. Diagnosis not confirmed.",
            vo,
            pressure_level=1,
            spans=[_span()],
        )
        assert out.decision == "hypothesis"
        assert out.hypothesis_claims
