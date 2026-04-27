"""Split evidence text into indexed spans (sentence-level with paragraph fallback).

v0.3: abbreviation-aware sentence splitting to avoid breaking on Dr., U.S., etc.
"""

from __future__ import annotations

import re

from .schemas import EvidenceSpan

# ── Common abbreviations that should NOT trigger sentence splits ──────
_ABBREVIATIONS = frozenset({
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Rev",
    "Gen", "Gov", "Sgt", "Cpl", "Pvt", "Capt", "Lt", "Col", "Maj",
    "Inc", "Corp", "Ltd", "Co", "LLC", "Bros", "Dept", "Div", "Est", "Assn",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec",
    "Ave", "Blvd", "Rd", "Ln", "Ct", "Pl",
    "vs", "etc", "approx", "dept", "govt", "Vol", "No",
    "Ph.D", "M.D", "B.A", "B.S", "M.S", "M.B.A", "J.D", "D.V.M", "D.Phil",
})

# Matches "Dr.", "U.S.", "e.g.", "Ph.D." style abbreviations
_ABBREV_DOT_RE = re.compile(
    r"\b([A-Za-z]+(?:\.[A-Za-z]+)+)\."          # U.S. / e.g. / i.e. / Ph.D.
    r"|"
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\."
)

# Simple sentence-end pattern: period/question-mark/exclamation followed by
# whitespace or end-of-string.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_PLACEHOLDER = "\x00ABBR_DOT\x00"


def split_evidence(evidence_text: str) -> list[EvidenceSpan]:
    """Return a list of EvidenceSpan objects for *evidence_text*.

    Strategy:
    1. Protect abbreviation dots with placeholders.
    2. Split on sentence boundaries.
    3. Restore placeholders.
    4. If that yields only one chunk, fall back to paragraph splitting.
    5. If still one chunk, return the whole text as a single span.
    """
    text = evidence_text.strip()
    if not text:
        return []

    # --- protect abbreviation dots ---
    protected, abbrev_positions = _protect_abbreviations(text)

    # --- attempt sentence split ---
    parts = _split_and_locate(protected, _SENT_RE)
    if len(parts) > 1:
        # Restore dots and map offsets back to original text
        restored = _restore_parts(parts, text, protected, abbrev_positions)
        return _to_spans(restored)

    # --- fallback: paragraph split (on original text) ---
    para_re = re.compile(r"\n\s*\n")
    parts = _split_and_locate(text, para_re)
    if len(parts) > 1:
        return _to_spans(parts)

    # --- single span ---
    return [EvidenceSpan(span_id="span_0", text=text, start_char=0, end_char=len(text))]


# ── helpers ────────────────────────────────────────────────────────────

def _protect_abbreviations(text: str) -> tuple[str, list[int]]:
    """Replace abbreviation dots with placeholders. Return modified text and positions."""
    positions: list[int] = []
    result = list(text)

    for m in _ABBREV_DOT_RE.finditer(text):
        # The dot is at the end of the match
        dot_pos = m.end() - 1
        if dot_pos < len(result) and text[dot_pos] == ".":
            result[dot_pos] = "\x00"
            positions.append(dot_pos)

    return "".join(result), positions


def _restore_parts(
    parts: list[tuple[str, int, int]],
    original: str,
    protected: str,
    positions: list[int],
) -> list[tuple[str, int, int]]:
    """Map parts from protected text back to original text."""
    restored: list[tuple[str, int, int]] = []
    for chunk_text, start, end in parts:
        # Use the original text at these offsets
        orig_chunk = original[start:end].strip()
        restored.append((orig_chunk, start, end))
    return restored


def _split_and_locate(text: str, pattern: re.Pattern) -> list[tuple[str, int, int]]:
    """Split *text* on *pattern* and return (chunk, start, end) triples."""
    parts: list[tuple[str, int, int]] = []
    prev = 0
    for m in pattern.finditer(text):
        chunk = text[prev : m.start()]
        if chunk.strip():
            parts.append((chunk.strip(), prev, m.start()))
        prev = m.end()
    tail = text[prev:]
    if tail.strip():
        parts.append((tail.strip(), prev, len(text)))
    return parts


def _to_spans(parts: list[tuple[str, int, int]]) -> list[EvidenceSpan]:
    return [
        EvidenceSpan(span_id=f"span_{i}", text=t, start_char=s, end_char=e)
        for i, (t, s, e) in enumerate(parts)
    ]
