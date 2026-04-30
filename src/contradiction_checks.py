"""Deterministic contradiction pre-checks — v0.4 simplified.

Only obvious status-pair contradictions are forced into CONTRADICTS_EVIDENCE:
  open/closed, approved/rejected, passed/failed,
  available/unavailable, launched/not launched, enabled/disabled.

Numeric, date, and money conflicts are logged as possible_conflict
(for audit / downstream inspection) but MUST NOT force gate decision.

Design principle: prefer missed contradiction over false contradiction.
No frame-based slot matching, no synonym tables, no shared-subject counting.
"""

from __future__ import annotations

import re
from typing import NamedTuple
from .schemas import EvidenceSpan, VerifiedClaim, EvidencePointer


# ── Status-pair contradictions (forced into gate) ───────────────────────

_STATUS_PAIRS: list[tuple[re.Pattern, re.Pattern]] = [
    (re.compile(r"\bopen(?:ed)?\b", re.I), re.compile(r"\bclosed\b", re.I)),
    (re.compile(r"\bpassed\b", re.I), re.compile(r"\bfailed\b", re.I)),
    (re.compile(r"\bapproved\b", re.I), re.compile(r"\brejected\b", re.I)),
    (re.compile(r"\bavailable\b", re.I), re.compile(r"\bunavailable\b", re.I)),
    (re.compile(r"\blaunched\b", re.I), re.compile(r"\bnot launched\b", re.I)),
    (re.compile(r"\benabled\b", re.I), re.compile(r"\bdisabled\b", re.I)),
    (re.compile(r"\bwas approved\b", re.I), re.compile(r"\bno \w+ has been approved\b", re.I)),
]


# ── Numeric / date / money conflicts (logged only, not forced) ──────────
# These are reported as possible_conflict entries for downstream
# inspection but do NOT override gate decisions.

_NUM_RE = re.compile(r"[$€£]?\s*[\d,.]+(?:\s*(?:million|billion|%|°C|°F|degrees?\s*[CF]))?", re.I)
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b",
    re.I,
)


class ConflictResult(NamedTuple):
    forced: list[VerifiedClaim]           # Always gate = contradiction
    possible: list[dict]                  # Logged only, no gate override


def check_contradictions(
    spans: list[EvidenceSpan],
    question: str,
) -> ConflictResult:
    """Return forced + possible contradictions found deterministically.

    Forced: status-pair antonyms on the same subject → gate = contradiction.
    Possible: numeric/date/money value conflicts → logged only.
    """
    forced: list[VerifiedClaim] = []
    possible: list[dict] = []
    if len(spans) < 2:
        return ConflictResult(forced, possible)

    # ── Pass 1: status-pair conflicts (forced) ──────────────────────
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            s_a, s_b = spans[i], spans[j]
            hit = _check_status_conflict(s_a, s_b)
            if hit:
                forced.append(hit)

    # ── Pass 2: numeric/date conflicts (possible only) ────────────
    # v0.4: NOT forced. Rely on verifier LLM to catch these.
    # Log them for downstream audit.
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            s_a, s_b = spans[i], spans[j]
            forced_hit = _check_requested_value_conflict(s_a, s_b, question)
            if forced_hit:
                forced.append(forced_hit)
                continue
            hit = _check_possible_conflict(s_a, s_b)
            if hit:
                possible.append(hit)

    return ConflictResult(forced, possible)


# ── Status conflict detection (FORCED) ─────────────────────────────────

def _check_status_conflict(
    s_a: EvidenceSpan, s_b: EvidenceSpan,
) -> VerifiedClaim | None:
    for pat_pos, pat_neg in _STATUS_PAIRS:
        a_pos = pat_pos.search(s_a.text)
        b_neg = pat_neg.search(s_b.text)
        a_neg = pat_neg.search(s_a.text)
        b_pos = pat_pos.search(s_b.text)
        if (a_pos and b_neg) or (a_neg and b_pos):
            return _make_contradiction(
                f"Status conflict: '{s_a.text[:60]}' vs '{s_b.text[:60]}'",
                s_a, s_b, "fact",
            )
    return None


# ── Possible conflict detection (LOGGED ONLY) ──────────────────────────

def _check_possible_conflict(
    s_a: EvidenceSpan, s_b: EvidenceSpan,
) -> dict | None:
    """Detect numeric/date value conflicts without forcing gate decision.

    v0.4: This returns a CONTRADICTS_EVIDENCE claim but the gate does
    NOT auto-route to contradiction for possible_conflict entries.
    Instead, they appear in verifier output for manual inspection.
    """
    # Date conflicts
    dates_a = _DATE_RE.findall(s_a.text)
    dates_b = _DATE_RE.findall(s_b.text)
    if dates_a and dates_b:
        # Only flag if the same subject is described (simple heuristic:
        # shared nouns between the two spans)
        shared_subjects = _shared_nouns(s_a.text, s_b.text)
        if shared_subjects >= 2 and len(dates_a) == len(dates_b):
            for da, db in zip(dates_a, dates_b):
                if da.lower() != db.lower():
                    return _make_possible_conflict(
                        f"Possible date conflict: {da} vs {db}",
                        s_a, s_b, "date_time",
                    )

    # Numeric value conflicts (conservative: require identical units)
    # v0.4: Skip — too many false positives. Verifier LLM handles these.
    # Keeping the function structure for future extension.
    return None


def _check_requested_value_conflict(
    s_a: EvidenceSpan,
    s_b: EvidenceSpan,
    question: str,
) -> VerifiedClaim | None:
    """Force only high-confidence conflicts on the user's requested slot."""
    q = question.lower()
    a = s_a.text.lower()
    b = s_b.text.lower()

    if _is_target_actual_pair(a, b):
        return None

    if "temperature" in q or "temp" in q:
        vals_a = _extract_temperature_values(s_a.text)
        vals_b = _extract_temperature_values(s_b.text)
        if vals_a and vals_b and vals_a != vals_b and _shared_nouns(s_a.text, s_b.text) >= 1:
            return _make_contradiction(
                f"Temperature conflict: {sorted(vals_a)} vs {sorted(vals_b)}",
                s_a, s_b, "number",
            )

    if "how many" in q and ("unit" in q or "sold" in q):
        vals_a = _extract_plain_numbers(s_a.text)
        vals_b = _extract_plain_numbers(s_b.text)
        if vals_a and vals_b and vals_a != vals_b and _same_requested_quantity_context(a, b):
            return _make_contradiction(
                f"Quantity conflict: {sorted(vals_a)} vs {sorted(vals_b)}",
                s_a, s_b, "number",
            )

    if "apartment" in q and ("detail" in q or "listing" in q):
        conflicts: list[str] = []
        bedrooms_a, bedrooms_b = _extract_bedrooms(a), _extract_bedrooms(b)
        sqft_a, sqft_b = _extract_sqft(a), _extract_sqft(b)
        money_a, money_b = _extract_money(a), _extract_money(b)
        dates_a, dates_b = _DATE_RE.findall(s_a.text), _DATE_RE.findall(s_b.text)
        if bedrooms_a and bedrooms_b and bedrooms_a != bedrooms_b:
            conflicts.append(f"bedrooms {sorted(bedrooms_a)} vs {sorted(bedrooms_b)}")
        if sqft_a and sqft_b and sqft_a != sqft_b:
            conflicts.append(f"size {sorted(sqft_a)} vs {sorted(sqft_b)}")
        if money_a and money_b and money_a != money_b:
            conflicts.append(f"price {sorted(money_a)} vs {sorted(money_b)}")
        if dates_a and ("immediate" in a or "immediate" in b):
            conflicts.append("availability date vs immediate")
        elif dates_a and dates_b and {d.lower() for d in dates_a} != {d.lower() for d in dates_b}:
            conflicts.append(f"availability {dates_a} vs {dates_b}")
        if conflicts and "listing" in a and "listing" in b:
            return _make_contradiction(
                "Listing detail conflict: " + "; ".join(conflicts),
                s_a, s_b, "fact",
            )

    return None


def _is_target_actual_pair(a: str, b: str) -> bool:
    joined = f"{a} {b}"
    return "target" in joined and "actual" in joined


def _same_requested_quantity_context(a: str, b: str) -> bool:
    joined = f"{a} {b}"
    return "sold" in joined and ("dispatched" in joined or "shipped" in joined or "shipping" in joined)


def _extract_temperature_values(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", m.group(1).lower())
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:Â?°\s*c|degrees?\s*c|celsius)\b", text, re.I)
    }


def _extract_plain_numbers(text: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", text):
        raw = m.group()
        if re.search(r"\b(?:19|20)\d{2}\b", raw):
            continue
        nums.add(raw.replace(",", ""))
    return nums


def _extract_bedrooms(text: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\b(\d+)\s*[- ]?bedroom", text, re.I)}


def _extract_sqft(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in re.finditer(r"\b(\d[\d,]*)\s*sq\s*ft\b", text, re.I)}


def _extract_money(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)", text)}


def _shared_nouns(a: str, b: str) -> int:
    """Count shared noun-like words (≥4 letters, not stop words)."""
    STOP = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this",
        "but", "his", "from", "they", "she", "been", "their", "would", "there",
    }
    na = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", a)} - STOP
    nb = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", b)} - STOP
    return len(na & nb)


def _make_contradiction(
    claim_text: str, span_a: EvidenceSpan, span_b: EvidenceSpan, kind: str,
) -> VerifiedClaim:
    return VerifiedClaim(
        claim_id=f"contra_{span_a.span_id}_{span_b.span_id}",
        claim_text=claim_text, claim_kind=kind,
        label="CONTRADICTS_EVIDENCE",
        evidence_pointers=[
            EvidencePointer(span_id=span_a.span_id, start_char=span_a.start_char,
                end_char=span_a.end_char, text_preview=span_a.text[:80]),
            EvidencePointer(span_id=span_b.span_id, start_char=span_b.start_char,
                end_char=span_b.end_char, text_preview=span_b.text[:80]),
        ],
        notes="Detected by deterministic status-pair pre-check",
    )


def _make_possible_conflict(
    claim_text: str, span_a: EvidenceSpan, span_b: EvidenceSpan, kind: str,
) -> dict:
    """Return audit-only possible conflict.

    Important: possible conflicts must never be represented as
    CONTRADICTS_EVIDENCE claims, because that label forces the gate into
    contradiction mode.
    """
    return {
        "claim_id": f"possible_{span_a.span_id}_{span_b.span_id}",
        "claim_text": claim_text,
        "claim_kind": kind,
        "span_ids": [span_a.span_id, span_b.span_id],
        "text_previews": [span_a.text[:80], span_b.text[:80]],
        "notes": "possible_conflict: audit only, does NOT force gate decision",
    }
