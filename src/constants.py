"""Shared constants for Project Verity-H v0.3.

Centralizes stop words and other constants used across modules.
"""

from __future__ import annotations

# ── Unified stop words ────────────────────────────────────────────────
# Used by span_matcher, claim_filter, and contradiction_checks.
# Union of all words previously defined in separate modules.

STOP_WORDS = frozenset({
    # Determiners & articles
    "the", "a", "an", "this", "that", "these", "those",
    # Be verbs
    "is", "was", "are", "were", "be", "been",
    # Have verbs
    "has", "had", "have",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "and", "or", "by", "from",
    "with", "as", "than", "per",
    # Pronouns
    "it", "its", "there", "their",
    # Common verbs / auxiliaries
    "do", "does", "did", "not", "no", "but", "also", "very",
    # Question words
    "what", "when", "where", "who", "how", "which", "much", "many", "about",
    # Evidence / meta words
    "based", "provided", "evidence", "according", "information",
    "mentioned", "specified", "given", "stated", "included",
    "known", "available",
    # Reporting verbs (used in contradiction_checks context)
    "show", "shows", "states", "says", "however",
})
