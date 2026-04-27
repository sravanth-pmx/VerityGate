"""Tests for inference_detector.py — all 4 tiers."""

import pytest
from src.inference_detector import (
    detect_inference,
    is_speculative_question,
)


class TestTier1EpistemicHedges:
    """Claims with epistemic language should be flagged as inference."""

    def test_most_likely(self):
        is_inf, reason = detect_inference("The most likely cause is bacterial infection", "What happened?")
        assert is_inf
        assert "epistemic" in reason

    def test_consistent_with(self):
        is_inf, _ = detect_inference("Symptoms are consistent with a bacterial infection", "q?")
        assert is_inf

    def test_suggests(self):
        is_inf, _ = detect_inference("The data suggests that growth will continue", "q?")
        assert is_inf

    def test_supports_diagnosis(self):
        is_inf, _ = detect_inference(
            "The elevated WBC supports an acute inflammatory process", "q?")
        assert is_inf

    def test_appears_to(self):
        is_inf, _ = detect_inference("The patient appears to have an infection", "q?")
        assert is_inf

    def test_factual_claim_not_flagged(self):
        is_inf, _ = detect_inference("The patient has fever of 38.9°C", "q?")
        assert not is_inf

    def test_pure_number_not_flagged(self):
        is_inf, _ = detect_inference("WBC count is 12,000/µL", "q?")
        assert not is_inf

    def test_name_not_flagged(self):
        is_inf, _ = detect_inference("The CFO confirmed the figures on October 5th", "q?")
        assert not is_inf


class TestTier2LogicalLeap:
    """Claims with logical connectors should be flagged."""

    def test_therefore(self):
        is_inf, reason = detect_inference("Therefore the patient has strep throat", "q?")
        assert is_inf
        assert "logical" in reason

    def test_based_on_findings(self):
        is_inf, _ = detect_inference("Based on these findings, the infection is bacterial", "q?")
        assert is_inf

    def test_this_indicates(self):
        is_inf, _ = detect_inference("This indicates a systemic issue", "q?")
        assert is_inf


class TestTier3Deontic:
    """Recommendations should be flagged as non-factual."""

    def test_should_be(self):
        is_inf, reason = detect_inference("The patient should be started on antibiotics", "q?")
        assert is_inf
        assert "deontic" in reason

    def test_recommended(self):
        is_inf, _ = detect_inference("Treatment with amoxicillin is recommended", "q?")
        assert is_inf

    def test_factual_dosage_not_flagged(self):
        is_inf, _ = detect_inference("The dosage is 500mg", "q?")
        assert not is_inf


class TestTier4SpeculativeQuestion:
    """Speculative questions should be detected."""

    def test_should_we(self):
        assert is_speculative_question("Should we invest in this startup?")

    def test_will_the(self):
        assert is_speculative_question("Will the product launch be successful?")

    def test_is_defendant(self):
        assert is_speculative_question("Is the defendant guilty?")

    def test_what_caused(self):
        assert is_speculative_question("What caused the patient's symptoms?")

    def test_why_did(self):
        assert is_speculative_question("Why did the server crash?")

    def test_factual_not_speculative(self):
        assert not is_speculative_question("What time is the meeting?")
        assert not is_speculative_question("Who wrote the report?")
        assert not is_speculative_question("How many employees were hired?")
        assert not is_speculative_question("What was the quarterly revenue?")


class TestTier4AnsweringSpeculative:
    """Claims answering speculative questions: evaluative=flag, data=keep."""

    def test_evaluative_claim_flagged(self):
        """'adds credibility' is evaluative, should be flagged."""
        is_inf, _ = detect_inference(
            "The founder's prior exit adds credibility", "Should we invest?")
        assert is_inf

    def test_data_claim_not_flagged(self):
        """'$2M ARR' is a data restatement, should NOT be flagged."""
        is_inf, _ = detect_inference(
            "The startup has $2M ARR", "Should we invest in this startup?")
        assert not is_inf

    def test_number_with_growth_pct(self):
        """'30% month-over-month growth' is data, should NOT be flagged."""
        is_inf, _ = detect_inference(
            "The startup shows 30% month-over-month growth",
            "Should we invest in this startup?")
        assert not is_inf

    def test_infection_diagnosis_flagged(self):
        """'bacterial infection' under causal question is inference."""
        is_inf, _ = detect_inference(
            "The cause is a bacterial infection",
            "What caused the patient's symptoms?")
        assert is_inf


class TestExactFailureCases:
    """Reproduce exact claims from v7 eval that were wrong."""

    def test_case_018_claim1_data_kept(self):
        """case_018: '30% growth' is factual data — keep as SUPPORTED."""
        is_inf, _ = detect_inference(
            "The startup shows strong growth (30% month-over-month)",
            "Should we invest in this startup?")
        assert not is_inf

    def test_case_018_claim2_inference_caught(self):
        """case_018: 'suggests experience' is epistemic — flag."""
        is_inf, _ = detect_inference(
            "The founder's prior $50M exit adds credibility and suggests experience in scaling successful ventures",
            "Should we invest in this startup?")
        assert is_inf

    def test_case_020_claim1_inference_caught(self):
        """case_020: 'consistent with' is epistemic — flag."""
        is_inf, _ = detect_inference(
            "The patient's symptoms—fever, sore throat, and fatigue—are consistent with a bacterial infection",
            "What caused the patient's symptoms?")
        assert is_inf

    def test_case_020_claim2_inference_caught(self):
        """case_020: 'supports an acute inflammatory' is epistemic — flag."""
        is_inf, _ = detect_inference(
            "The elevated white blood cell count (12,000/µL) supports an acute inflammatory or infectious process",
            "What caused the patient's symptoms?")
        assert is_inf

    def test_case_020_claim3_inference_caught(self):
        """case_020: 'most likely cause' is epistemic — flag."""
        is_inf, _ = detect_inference(
            "The most likely cause of the symptoms is a bacterial infection",
            "What caused the patient's symptoms?")
        assert is_inf
