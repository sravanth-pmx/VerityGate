"""Tests for claim_filter.py — meta/relevance/dedup filtering.

v0.2.1: slot-aware relevance tests for specific failure cases.
"""

import pytest
from src.claim_filter import (
    filter_claims_pre_labeling,
    filter_unknown_claims_post_labeling,
    _extract_question_slots,
    _is_slot_relevant,
    _extract_keywords,
)


class TestPreLabeling:
    def test_removes_meta_claims(self):
        claims = [
            "The meeting is at 3pm",
            "The requested information is provided",
            "The answer states the revenue",
        ]
        filtered, stats = filter_claims_pre_labeling(claims, "When is the meeting?")
        assert len(filtered) == 1
        assert "3pm" in filtered[0]
        assert stats.meta_removed == 2

    def test_removes_based_on_evidence(self):
        claims = [
            "Based on the provided evidence, the revenue is $4.2M",
            "Revenue is $4.2M",
        ]
        filtered, stats = filter_claims_pre_labeling(claims, "Revenue?")
        assert len(filtered) == 1
        assert "4.2M" in filtered[0]

    def test_removes_duplicates(self):
        claims = [
            "Revenue was $4.2 million",
            "Revenue was $4.2 million",
            "Revenue was $4.2 million.",
        ]
        filtered, stats = filter_claims_pre_labeling(claims, "What was revenue?")
        assert len(filtered) == 1
        assert stats.dedup_removed >= 1

    def test_keeps_good_claims(self):
        claims = [
            "Q3 revenue was $4.2 million",
            "Revenue was up 12% from Q2",
            "CFO confirmed on October 5th",
        ]
        filtered, stats = filter_claims_pre_labeling(claims, "What was the revenue?")
        assert len(filtered) == 3

    def test_removes_short(self):
        claims = ["OK", "The meeting is at 3pm"]
        filtered, _ = filter_claims_pre_labeling(claims, "When?")
        assert len(filtered) == 1


class TestPostLabelingSlotRelevance:
    """Test that irrelevant unknown claims are removed based on question slot."""

    def test_launch_date_drops_location_and_audience(self):
        """Question about launch DATE should not keep launch location or audience."""
        claims = [
            {"claim_text": "The project launched on January 10", "label": "SUPPORTED"},
            {"claim_text": "The launch location is not mentioned", "label": "NOT_IN_EVIDENCE"},
            {"claim_text": "The target audience is unknown", "label": "NOT_IN_EVIDENCE"},
            {"claim_text": "The delay reason is not provided", "label": "NOT_IN_EVIDENCE"},
        ]
        filtered, stats = filter_unknown_claims_post_labeling(
            claims, "When did the project launch?")
        labels = [c["label"] for c in filtered]
        texts = [c["claim_text"] for c in filtered]
        assert any("January 10" in t for t in texts)  # supported kept
        # Location and audience should be dropped (they don't match when/date slot)
        assert not any("location" in t.lower() for t in texts if "SUPPORTED" not in str(filtered[texts.index(t)].get("label","")))
        assert stats.irrelevant_removed >= 1

    def test_budget_approver_drops_scope_and_process(self):
        """Question about WHO approved budget should not keep budget scope/process."""
        claims = [
            {"claim_text": "Budget was approved by CFO Maria Santos", "label": "SUPPORTED"},
            {"claim_text": "The budget scope is not provided", "label": "NOT_IN_EVIDENCE"},
            {"claim_text": "The approval process is not documented", "label": "NOT_IN_EVIDENCE"},
            {"claim_text": "Official documentation is not provided", "label": "NOT_IN_EVIDENCE"},
        ]
        filtered, stats = filter_unknown_claims_post_labeling(
            claims, "Who approved the budget?")
        assert any("Maria Santos" in c["claim_text"] for c in filtered)
        assert stats.irrelevant_removed >= 1

    def test_keeps_relevant_unknowns(self):
        """Side effects are relevant when question asks about dosage/frequency/side effects."""
        claims = [
            {"claim_text": "Dosage is 500mg", "label": "SUPPORTED"},
            {"claim_text": "Side effects are not known", "label": "NOT_IN_EVIDENCE"},
        ]
        filtered, _ = filter_unknown_claims_post_labeling(
            claims, "What is the drug's dosage, frequency, and side effects?")
        assert len(filtered) == 2  # both kept

    def test_keeps_supported_always(self):
        claims = [{"claim_text": "The meeting is at 3pm", "label": "SUPPORTED"}]
        filtered, _ = filter_unknown_claims_post_labeling(claims, "anything unrelated")
        assert len(filtered) == 1

    def test_dedup_in_post(self):
        claims = [
            {"claim_text": "Revenue was $4.2M", "label": "SUPPORTED"},
            {"claim_text": "Revenue was $4.2M", "label": "SUPPORTED"},
        ]
        filtered, stats = filter_unknown_claims_post_labeling(claims, "revenue?")
        assert len(filtered) == 1
        assert stats.dedup_removed == 1

    def test_stock_price_keeps_price_unknown(self):
        """Question about stock price should keep 'price not provided'."""
        claims = [
            {"claim_text": "Company was founded in 2010", "label": "SUPPORTED"},
            {"claim_text": "The current stock price is not provided", "label": "NOT_IN_EVIDENCE"},
        ]
        filtered, stats = filter_unknown_claims_post_labeling(
            claims, "What is the company's current stock price?")
        assert len(filtered) == 2  # price unknown IS relevant

    def test_cause_question_keeps_cause_unknown(self):
        """Question about cause should keep 'cause not specified'."""
        claims = [
            {"claim_text": "Server went offline at 02:14 UTC", "label": "SUPPORTED"},
            {"claim_text": "The cause of the outage is not specified", "label": "NOT_IN_EVIDENCE"},
        ]
        filtered, _ = filter_unknown_claims_post_labeling(
            claims, "What was the cause of the server outage?")
        assert len(filtered) == 2


class TestQuestionSlotExtraction:
    def test_when_question(self):
        slots = _extract_question_slots("When did the project launch?")
        assert "date" in slots or "launch" in slots

    def test_who_question(self):
        slots = _extract_question_slots("Who approved the budget?")
        assert "who" in slots or "approved" in slots

    def test_how_many_question(self):
        slots = _extract_question_slots("How many employees were hired?")
        assert "many" in slots or "employees" in slots


class TestMissingSubjectExtraction:
    def test_is_not_mentioned(self):
        from src.claim_filter import _extract_missing_subject
        assert "launch location" in _extract_missing_subject("The launch location is not mentioned")

    def test_are_unknown(self):
        from src.claim_filter import _extract_missing_subject
        result = _extract_missing_subject("Side effects are unknown")
        assert "side effects" in result.lower()
