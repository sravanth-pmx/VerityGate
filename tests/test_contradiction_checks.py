"""Tests for contradiction_checks.py — v0.4 simplified.

Only status-pair contradictions are FORCED into gate decision.
Numeric/date/money conflicts are logged as possible_conflict but
MUST NOT force gate decision=contradiction.
"""

import pytest
from src.contradiction_checks import check_contradictions, ConflictResult
from src.schemas import EvidenceSpan


def _span(sid, text, start=0):
    return EvidenceSpan(span_id=sid, text=text, start_char=start, end_char=start + len(text))


class TestForcedStatusPairContradictions:
    """These MUST be detected and forced into gate."""

    def test_open_vs_closed(self):
        spans = [
            _span("span_0", "The bridge opened to traffic on June 1."),
            _span("span_1", "The bridge failed inspection and remains closed.", start=50),
        ]
        result = check_contradictions(spans, "Is the bridge open?")
        assert len(result.forced) >= 1
        assert all(c.label == "CONTRADICTS_EVIDENCE" for c in result.forced)

    def test_approved_vs_rejected(self):
        spans = [
            _span("span_0", "Budget was approved by the CFO on April 2."),
            _span("span_1", "Budget was rejected by the board on April 3.", start=50),
        ]
        result = check_contradictions(spans, "Was the budget approved?")
        assert len(result.forced) >= 1

    def test_passed_vs_failed(self):
        spans = [
            _span("span_0", "All candidates passed the exam."),
            _span("span_1", "Three candidates failed the practical assessment.", start=50),
        ]
        result = check_contradictions(spans, "Did everyone pass?")
        assert len(result.forced) >= 1

    def test_available_vs_unavailable(self):
        spans = [
            _span("span_0", "The product is available for purchase."),
            _span("span_1", "The product is unavailable due to supply issues.", start=50),
        ]
        result = check_contradictions(spans, "Is the product available?")
        assert len(result.forced) >= 1

    def test_launched_vs_not_launched(self):
        spans = [
            _span("span_0", "The satellite was launched on January 10."),
            _span("span_1", "The satellite was not launched due to weather.", start=50),
        ]
        result = check_contradictions(spans, "Was the satellite launched?")
        assert len(result.forced) >= 1

    def test_enabled_vs_disabled(self):
        spans = [
            _span("span_0", "The feature is enabled for all users."),
            _span("span_1", "The feature was disabled by the admin.", start=50),
        ]
        result = check_contradictions(spans, "Is the feature enabled?")
        assert len(result.forced) >= 1


class TestRequestedValueConflicts:
    """Numeric/date/money conflicts are forced only for requested slots."""

    def test_date_conflict_not_forced(self):
        """Date conflicts may appear in possible, but NOT forced."""
        spans = [
            _span("span_0", "The building was completed on February 15, 2024."),
            _span("span_1", "The city certificate says completion was February 28, 2024.", start=80),
        ]
        result = check_contradictions(spans, "When was the building completed?")
        # v0.4: Date conflicts are NOT forced (too many false positives).
        # They may appear in possible for audit.
        assert len(result.forced) == 0

    def test_numeric_conflict_for_requested_units_forced(self):
        """Requested units-sold conflicts are forced."""
        spans = [
            _span("span_0", "The sales report states 1,200 units were sold."),
            _span("span_1", "Warehouse log shows 980 units dispatched.", start=60),
        ]
        result = check_contradictions(spans, "How many units were sold?")
        assert len(result.forced) >= 1

    def test_money_conflict_not_forced(self):
        """Money conflicts are NOT forced."""
        spans = [
            _span("span_0", "Revenue target is $100 million."),
            _span("span_1", "Actual revenue through Q3 is $82.4 million.", start=60),
        ]
        result = check_contradictions(spans, "What is the revenue?")
        assert len(result.forced) == 0

    def test_temperature_conflict_for_requested_temperature_forced(self):
        """Requested temperature conflicts are forced."""
        spans = [
            _span("span_0", "Sensor A recorded 23.5 degrees C."),
            _span("span_1", "Sensor B recorded 28.1 degrees C.", start=50),
        ]
        result = check_contradictions(spans, "What is the temperature?")
        assert len(result.forced) >= 1

    def test_temperature_conflict_with_mojibake_degree_symbol_forced(self):
        spans = [
            _span("span_0", "Sensor A recorded 23.5Â°C at noon."),
            _span("span_1", "Sensor B recorded 28.1Â°C at noon.", start=50),
        ]
        result = check_contradictions(spans, "What temperature was recorded?")
        assert len(result.forced) >= 1

    def test_apartment_listing_detail_conflict_forced(self):
        spans = [
            _span("span_0", "The apartment listing on the agency website shows a 2-bedroom apartment at 850 sq ft for $2,100/month."),
            _span("span_1", "The same listing on the aggregator site shows 3 bedrooms at 920 sq ft for $2,350/month.", start=120),
        ]
        result = check_contradictions(spans, "What are the details of the apartment listing?")
        assert len(result.forced) >= 1

    def test_percentage_conflict_not_forced(self):
        """Percentage conflicts are NOT forced."""
        spans = [
            _span("span_0", "Revenue grew 25% year over year."),
            _span("span_1", "Employee count grew 15% year over year.", start=60),
        ]
        result = check_contradictions(spans, "What is the growth rate?")
        assert len(result.forced) == 0


class TestNoFalsePositives:
    """Unrelated spans must not trigger either forced or possible."""

    def test_unrelated_spans(self):
        spans = [
            _span("span_0", "The startup has $2M ARR."),
            _span("span_1", "The founder exited a company for $50M.", start=80),
        ]
        result = check_contradictions(spans, "Should we invest?")
        assert len(result.forced) == 0

    def test_different_events_different_dates(self):
        spans = [
            _span("span_0", "Version 2.0 was released on January 15, 2024."),
            _span("span_1", "Version 2.1 was released on March 3, 2024.", start=60),
        ]
        result = check_contradictions(spans, "When was the last release?")
        assert len(result.forced) == 0

    def test_different_attributes_same_entity(self):
        spans = [
            _span("span_0", "The car is a 2019 model with 45,000 miles."),
            _span("span_1", "The car was serviced in 2023.", start=70),
        ]
        result = check_contradictions(spans, "What is the car's mileage?")
        assert len(result.forced) == 0


class TestEdgeCases:
    def test_single_span(self):
        result = check_contradictions([_span("span_0", "Test.")], "q?")
        assert result.forced == []
        assert result.possible == []

    def test_empty(self):
        result = check_contradictions([], "q?")
        assert result.forced == []
        assert result.possible == []

    def test_conflict_result_is_named_tuple(self):
        """API returns ConflictResult with .forced and .possible."""
        result = check_contradictions([], "q?")
        assert isinstance(result, ConflictResult)
        assert hasattr(result, "forced")
        assert hasattr(result, "possible")
