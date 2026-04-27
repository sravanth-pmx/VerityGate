"""Tests for constants.py — shared stop words."""

from src.constants import STOP_WORDS


class TestStopWords:
    def test_is_frozenset(self):
        assert isinstance(STOP_WORDS, frozenset)

    def test_contains_common_words(self):
        for word in ["the", "a", "is", "was", "in", "on", "for", "and"]:
            assert word in STOP_WORDS

    def test_no_content_words(self):
        """Stop words should not contain meaningful content words."""
        for word in ["revenue", "temperature", "project", "meeting", "budget"]:
            assert word not in STOP_WORDS

    def test_used_by_all_modules(self):
        """All modules that use stop words should reference the shared STOP_WORDS."""
        from src.span_matcher import _STOP_WORDS as sm_sw
        from src.claim_filter import _STOP_WORDS as cf_sw
        # All should be the same object (imported from constants)
        assert sm_sw is STOP_WORDS
        assert cf_sw is STOP_WORDS
        # v0.4: contradiction_checks simplified — no longer uses STOP_WORDS
