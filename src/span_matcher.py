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
    r"not (?:provided|mentioned|specified|stated|included)(?: in the evidence)?|"
    r"is not provided|is not mentioned|is not specified|"
    r"no (?:information|data|mention) (?:about|regarding|on|of)|"
    r"cannot (?:determine|confirm|verify) from|"
    r"not enough information|"
    r"does not contain information|"
    r"no evidence (?:about|for|of|regarding)|"
    r"is (?:unknown|unavailable|missing|absent)",
    re.IGNORECASE,
)

# Evidence text that confirms info is missing/pending
DEFERRAL_IN_EVIDENCE = re.compile(
    r"has not been finalized|"
    r"not (?:been )?finalized|"
    r"still being collected|"
    r"data is (?:still )?being|"
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


def _make_pointer(span: EvidenceSpan) -> EvidencePointer:
    return EvidencePointer(
        span_id=span.span_id,
        start_char=span.start_char,
        end_char=span.end_char,
        text_preview=span.text[:80],
    )
