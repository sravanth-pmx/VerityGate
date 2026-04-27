"""Tests for evidence_spans.py."""

import pytest
from src.evidence_spans import split_evidence
from src.schemas import EvidenceSpan


class TestSplitEvidence:
    def test_empty_string(self):
        assert split_evidence("") == []
        assert split_evidence("   ") == []

    def test_single_sentence(self):
        text = "The cat sat on the mat."
        spans = split_evidence(text)
        assert len(spans) == 1
        assert spans[0].span_id == "span_0"
        assert spans[0].text == text

    def test_multiple_sentences(self):
        text = "The meeting is at 3pm. It will be in Room B. Bring your laptop."
        spans = split_evidence(text)
        assert len(spans) == 3
        assert spans[0].span_id == "span_0"
        assert spans[1].span_id == "span_1"
        assert spans[2].span_id == "span_2"
        assert "3pm" in spans[0].text
        assert "Room B" in spans[1].text
        assert "laptop" in spans[2].text

    def test_paragraph_fallback(self):
        text = "First paragraph with no sentence-ending punctuation\n\nSecond paragraph also no punctuation"
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "First" in spans[0].text
        assert "Second" in spans[1].text

    def test_span_ids_sequential(self):
        text = "One. Two. Three. Four."
        spans = split_evidence(text)
        for i, span in enumerate(spans):
            assert span.span_id == f"span_{i}"

    def test_start_end_chars(self):
        text = "Hello world. Goodbye world."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert spans[0].start_char == 0
        # Each span text should be findable in the original
        for span in spans:
            assert span.text in text

    def test_question_marks(self):
        text = "Is it raining? Yes it is. Will it stop? Nobody knows."
        spans = split_evidence(text)
        assert len(spans) == 4

    def test_exclamation_marks(self):
        text = "Fire! Evacuate immediately. Call 911!"
        spans = split_evidence(text)
        assert len(spans) == 3

    def test_returns_evidence_span_objects(self):
        text = "Test sentence one. Test sentence two."
        spans = split_evidence(text)
        for span in spans:
            assert isinstance(span, EvidenceSpan)

class TestAbbreviationHandling:
    """v0.3: abbreviation-aware splitting should NOT break on Dr., U.S., etc."""

    def test_dr_prefix(self):
        text = "Dr. Smith said the temp is 98.6°F. The test was normal."
        spans = split_evidence(text)
        # Should NOT split on "Dr." — expect 2 spans, not 3+
        assert len(spans) == 2
        assert "Dr. Smith" in spans[0].text

    def test_us_abbreviation(self):
        text = "The U.S. average is 97.8°F. The global average is 98.0°F."
        spans = split_evidence(text)
        # Should NOT split on "U." or "S." — expect 2 spans
        assert len(spans) == 2
        assert "U.S." in spans[0].text

    def test_inc_abbreviation(self):
        text = "Acme Inc. reported profits. Revenue grew 15%."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "Inc." in spans[0].text

    def test_mr_mrs_prefix(self):
        text = "Mr. Johnson and Mrs. Lee attended. The meeting was productive."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "Mr. Johnson" in spans[0].text

    def test_normal_sentences_still_split(self):
        """Regular sentences without abbreviations should still split normally."""
        text = "The cat sat. The dog ran. The bird flew."
        spans = split_evidence(text)
        assert len(spans) == 3

    def test_mixed_abbreviations_and_sentences(self):
        text = "Dr. Elena Vasquez published the report. It was released on March 15, 2024."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "Dr. Elena Vasquez" in spans[0].text

    def test_phd_abbreviation(self):
        text = "Dr. Sarah Chen, Ph.D., reported the findings. The patient recovered."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "Ph.D." in spans[0].text

    def test_md_abbreviation(self):
        text = "Jane Smith, M.D., performed surgery. She was assisted by a nurse."
        spans = split_evidence(text)
        assert len(spans) == 2
        assert "M.D." in spans[0].text

    def test_mba_and_bs_abbreviations(self):
        text = "John holds an M.B.A. and a B.S. He is now a VP. They work at a firm."
        spans = split_evidence(text)
        # VP. is NOT an abbreviation, so it splits. M.B.A. and B.S. are protected.
        assert len(spans) >= 2
        assert "M.B.A." in spans[0].text
        assert "B.S." in spans[0].text
