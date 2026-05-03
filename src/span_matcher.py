"""Deterministic span matcher — labels claims against evidence without LLM.

This is the model-independent backbone of the verifier. It handles:
1. Exact/fuzzy substring matching → SUPPORTED
2. Absence/deferral detection → NOT_IN_EVIDENCE
3. No-match → UNSUPPORTED (needs LLM or stays as-is)

Used both as:
- Primary labeler for simple claims (works on any model size)
- Post-processor to fix LLM mislabeling
"""

from __future__ import annotations

import re

from .schemas import EvidencePointer, EvidenceSpan, VerifiedClaim
from .constants import STOP_WORDS
from .inference_detector import detect_inference

# ── Absence / deferral patterns ───────────────────────────────────────
# Claim text that says "evidence doesn't have X"
ABSENCE_IN_CLAIM = re.compile(
    r"evidence does not (?:specify|mention|include|contain|provide|state)|"
    r"not (?:provided|mentioned|specified|stated|included|listed|documented|shown|recorded|populated|calculated)(?: in the evidence)?|"
    r"is not provided|is not mentioned|is not specified|"
    r"is not documented|is not shown|is not recorded|is not populated|"
    r"(?:field\s+)?(?:reads\s+)?(?:unassigned|redacted|tbd)\b|"
    r"(?:left|marked)\s+blank|"
    r"data\s+is\s+missing|"
    r"missing from|"
    r"no (?:information|data|mention) (?:about|regarding|on|of)|"
    r"no .{1,60} (?:is |are |were |was )?(?:included|provided|stated|shown|documented)|"
    r"cannot (?:determine|confirm|verify) from|"
    r"not enough information|"
    r"does not contain information|"
    r"no evidence (?:about|for|of|regarding)|"
    r"is (?:unknown|unavailable|missing|absent)",
    re.IGNORECASE,
)

_INFERENCE_PHRASES = re.compile(
    r"\blikely due to\b|"
    r"\bprobably caused by\b|"
    r"\bappears to be caused by\b|"
    r"\bseems likely\b|"
    r"\bit is likely\b|"
    r"\bwe can approve\b|"
    r"\bshould approve\b|"
    r"\bgood investment\b|"
    r"\bgood neighborhood\b|"
    r"\bqualified for\b",
    re.IGNORECASE,
)

# Evidence text that confirms info is missing/pending
DEFERRAL_IN_EVIDENCE = re.compile(
    r"has not been finalized|"
    r"has not been (?:collected|recorded|scheduled)|"
    r"not (?:been )?finalized|"
    r"not yet (?:scheduled|calculated)|"
    r"still being (?:collected|processed)|"
    r"testing still pending|"
    r"data is (?:still )?being|"
    r"currently being (?:collected|restructured|reviewed|updated|finalized|processed)|"
    r"new .{0,40} (?:will be )?available next month|"
    r"will be (?:calculated|mailed|sent|circulated|provided|available) (?:later|next|after)|"
    r"please contact|"
    r"contact(?:ed)? (?:the |directly )?.{0,30}(?:desk|office|department)|"
    r"should be contacted|"
    r"not (?:yet )?(?:available|determined|decided|released|announced)|"
    r"pending|"
    r"under review|"
    r"to be (?:determined|announced|confirmed|decided)",
    re.IGNORECASE,
)

_STOP_WORDS = STOP_WORDS


def label_claim_against_spans(
    claim_text: str,
    spans: list[EvidenceSpan],
) -> tuple[str, EvidencePointer | None, str]:
    """Deterministically label a claim against evidence spans.

    Returns: (label, pointer_or_None, notes)
    Label is one of: SUPPORTED, NOT_IN_EVIDENCE, or empty string "" meaning
    "I don't know — let the LLM decide".
    """
    ct = claim_text.strip()
    ct_lower = ct.lower().rstrip(".")
    if _INFERENCE_PHRASES.search(ct):
        return "UNSUPPORTED", None, "claim is an inference/conclusion, not direct evidence"

    if _has_unstated_calculated_percentage(ct, spans):
        return "UNSUPPORTED", None, "calculated percentage is not explicitly stated in evidence"

    if _has_unstated_calculated_total(ct, spans):
        return "UNSUPPORTED", None, "calculated total is not explicitly stated in evidence"

    # ── 1. Claim says evidence is absent → NOT_IN_EVIDENCE ────────────
    if ABSENCE_IN_CLAIM.search(ct):
        return "NOT_IN_EVIDENCE", None, "claim states absence of evidence"

    # ── 2. Claim text matches a deferral pattern → NOT_IN_EVIDENCE ────
    if DEFERRAL_IN_EVIDENCE.search(ct):
        return "NOT_IN_EVIDENCE", None, "claim describes pending/deferred info"

    # ── 3. Exact substring match in a span → SUPPORTED ────────────────
    for span in spans:
        st_lower = span.text.lower()
        if len(ct_lower) >= 10 and ct_lower in st_lower:
            ptr = _make_pointer(span)
            return "SUPPORTED", ptr, "exact substring match"

    # ── 3b. Number + keyword match ────────────────────────────────────
    # If claim has numbers, all match a span, and ≥1 keyword overlaps
    ct_nums = _extract_comparable_nums(ct)
    ct_words = _key_words(ct_lower)
    if ct_nums and ct_words:
        for span in spans:
            span_nums = _extract_comparable_nums(span.text)
            if ct_nums and ct_nums.issubset(span_nums):
                st_words = _key_words(span.text.lower())
                if ct_words & st_words:
                    ptr = _make_pointer(span)
                    return "SUPPORTED", ptr, "number + keyword match"

    # ── 4. Fuzzy word overlap (≥80% of claim keywords in span) ────────
    if ct_words and len(ct_words) >= 2:
        for span in spans:
            st_words = _key_words(span.text.lower())
            overlap = len(ct_words & st_words) / len(ct_words)
            if overlap >= 0.80:
                # Consistency check: verify numbers/entities match
                if _numbers_consistent(ct, span.text):
                    ptr = _make_pointer(span)
                    return "SUPPORTED", ptr, f"fuzzy match ({overlap:.0%} keyword overlap)"
                else:
                    return "", None, "fuzzy match but numbers inconsistent"

    # ── 5. Can't determine → let LLM decide ──────────────────────────
    return "", None, ""


def relabel_claims(
    claims: list[VerifiedClaim],
    spans: list[EvidenceSpan],
    question: str = "",
) -> list[VerifiedClaim]:
    """Post-process LLM-labeled claims using deterministic checks.

    Rules (applied in order):
    1. SUPPORTED + absence/deferral text → downgrade to NOT_IN_EVIDENCE
    2. CONTRADICTS_EVIDENCE + absence text → downgrade to NOT_IN_EVIDENCE
    3. NOT_IN_EVIDENCE/UNSUPPORTED + span match → upgrade to SUPPORTED
    4. SUPPORTED + inference detected → downgrade to UNSUPPORTED
    """
    for claim in claims:
        det_label, det_ptr, det_notes = label_claim_against_spans(claim.claim_text, spans)

        if claim.label == "SUPPORTED" and det_label == "NOT_IN_EVIDENCE":
            # LLM wrongly called an absence claim SUPPORTED
            claim.label = "NOT_IN_EVIDENCE"
            claim.evidence_pointers = []
            claim.notes = (claim.notes or "") + f" [det: {det_notes}]"

        elif claim.label == "CONTRADICTS_EVIDENCE" and det_label == "NOT_IN_EVIDENCE":
            claim.label = "NOT_IN_EVIDENCE"
            claim.evidence_pointers = []
            claim.notes = (claim.notes or "") + f" [det: absence claim, not contradiction]"

        elif claim.label in ("NOT_IN_EVIDENCE", "UNSUPPORTED") and det_label == "SUPPORTED" and det_ptr:
            claim.label = "SUPPORTED"
            claim.evidence_pointers = [det_ptr]
            claim.notes = (claim.notes or "") + f" [det: {det_notes}]"

        elif claim.label == "SUPPORTED" and det_label == "UNSUPPORTED":
            claim.label = "UNSUPPORTED"
            claim.evidence_pointers = []
            claim.notes = (claim.notes or "") + f" [det: {det_notes}]"

        # Rule 4: inference detection — must come AFTER span match upgrades
        if claim.label == "SUPPORTED" and question:
            is_inf, inf_reason = detect_inference(claim.claim_text, question)
            if is_inf:
                claim.label = "UNSUPPORTED"
                claim.evidence_pointers = []
                claim.notes = (claim.notes or "") + f" [det: inference — {inf_reason}]"

    return claims


# ── Helpers ────────────────────────────────────────────────────────────

def _key_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w{4,}\b", text)} - _STOP_WORDS


# ── Numeric/entity consistency ────────────────────────────────────────

_NUM_RE = re.compile(r"(?<![A-Za-z])[\$]?\d[\d,.]*(?:\s*(?:%|degrees|°[CF]|million|billion|mg|hPa))?(?![A-Za-z])")

def _numbers_consistent(claim: str, span: str) -> bool:
    """Check that numbers in the claim match numbers in the span.

    If the claim mentions a number that doesn't appear in the span,
    the fuzzy match is unreliable — could be a hallucinated value.
    If the claim has no numbers, skip the check (it's a text-only claim).
    """
    claim_nums = _extract_comparable_nums(claim)
    if not claim_nums:
        return True  # no numbers to check

    span_nums = _extract_comparable_nums(span)
    if not span_nums:
        return True  # span has no numbers either, can't compare

    # Every number in the claim must appear in the span
    for cn in claim_nums:
        if cn not in span_nums:
            return False
    return True


def _extract_comparable_nums(text: str) -> set[str]:
    """Extract normalized number strings for comparison."""
    nums: set[str] = set()
    for m in _NUM_RE.finditer(text):
        # Normalize: strip $, commas, spaces
        n = m.group().strip()
        n = re.sub(r"[$,\s]", "", n)
        n = n.lower()
        if n:
            nums.add(n)
    return nums


_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")


def _has_unstated_calculated_percentage(claim_text: str, spans: list[EvidenceSpan]) -> bool:
    """Flag percentage answers when that percentage is not stated in evidence.

    This keeps explicit percentages supported, but prevents the verifier from
    accepting arithmetic such as 186 / 240 = 77.5% as if it were textual evidence.
    """
    claim_percents = {re.sub(r"\s+", "", p.lower()) for p in _PERCENT_RE.findall(claim_text)}
    if not claim_percents:
        return False

    evidence_text = " ".join(span.text for span in spans)
    evidence_percents = {re.sub(r"\s+", "", p.lower()) for p in _PERCENT_RE.findall(evidence_text)}
    if claim_percents <= evidence_percents:
        return False

    return any(word in claim_text.lower() for word in ("percentage", "percent", "passed", "rate"))


def _has_unstated_calculated_total(claim_text: str, spans: list[EvidenceSpan]) -> bool:
    """Flag total answers when the exact total is not stated in evidence.

    This prevents accepting simple arithmetic or set arithmetic as textual
    support, e.g. "40 morning, 35 afternoon, 12 both" -> "63 total".
    """
    ct = claim_text.lower()
    if not re.search(r"\b(total|in total|altogether|overall)\b", ct):
        return False

    claim_nums = _extract_comparable_nums(claim_text)
    if len(claim_nums) != 1:
        return False

    evidence_text = " ".join(span.text for span in spans)
    evidence_nums = _extract_comparable_nums(evidence_text)
    if not evidence_nums or not claim_nums.isdisjoint(evidence_nums):
        return False

    return len(evidence_nums) >= 2


def _make_pointer(span: EvidenceSpan) -> EvidencePointer:
    return EvidencePointer(
        span_id=span.span_id,
        start_char=span.start_char,
        end_char=span.end_char,
        text_preview=span.text[:80],
    )
