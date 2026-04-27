"""Claim quality filter — removes junk, irrelevant, and duplicate claims.

v0.3: question-slot-aware relevance filtering.
v0.4: Added filter_supported_claims_by_relevance — downgrades SUPPORTED
claims that share no keywords with the question. This normalizes extraction
behavior across model sizes (large models over-extract tangentially-related
facts; this filter prunes them before the gate).

Runs BEFORE labeling to remove meta-claims and dupes (saves LLM calls).
Runs AFTER labeling to:
  1. Downgrade SUPPORTED claims with no question-keyword overlap
  2. Remove unknown claims about slots the user didn't ask for

Key improvement: detects what the question asks for (the "requested slots")
and only keeps unknown claims that match those slots. "launch location" is
irrelevant if the question asks "When did the project launch?" — even though
they share the word "launch".
"""

from __future__ import annotations

import re

from .constants import STOP_WORDS

# ── Meta-claim patterns (always remove) ───────────────────────────────
_META_PATTERNS = re.compile(
    r"requested information is provided|"
    r"the answer (?:states|provides|includes|mentions|confirms)|"
    r"the draft (?:states|provides|includes|mentions)|"
    r"final answer|"
    r"the information is provided|"
    r"the response (?:states|provides|includes)|"
    r"this (?:claim|statement) is|"
    r"the above (?:claim|information)|"
    r"as (?:stated|mentioned) (?:above|earlier|in the)|"
    r"based on the (?:provided )?evidence",
    re.IGNORECASE,
)

_STOP_WORDS = STOP_WORDS

# ── Question type → requested slot keywords ───────────────────────────
# Maps question patterns to the kind of slot being requested.
# Used to filter unknown claims that don't match the requested slot.

_QUESTION_SLOT_PATTERNS: list[tuple[re.Pattern, set[str]]] = [
    # When → time/date slots
    (re.compile(r"^when\b", re.I),
     {"date", "time", "when", "year", "month", "day", "schedule", "deadline",
      "launch", "start", "end", "publish", "open", "close"}),
    # Who → person/entity slots
    (re.compile(r"^who\b", re.I),
     {"who", "name", "person", "author", "manager", "lead", "approver",
      "approved", "wrote", "hired", "instructor", "sponsor"}),
    # Where → location slots
    (re.compile(r"^where\b", re.I),
     {"where", "location", "place", "city", "room", "address", "site"}),
    # How many / how much → quantity slots
    (re.compile(r"^how (?:many|much)\b", re.I),
     {"how", "many", "much", "number", "count", "total", "amount",
      "price", "cost", "revenue", "salary", "budget", "units", "employees"}),
    # What ... temperature/price/height/etc → specific attribute
    (re.compile(r"\b(?:temperature|temp)\b", re.I), {"temperature", "temp", "degrees", "recorded"}),
    (re.compile(r"\b(?:price|cost|stock)\b", re.I), {"price", "cost", "stock", "value"}),
    (re.compile(r"\b(?:height|tall)\b", re.I), {"height", "tall", "meters", "feet"}),
    (re.compile(r"\b(?:population)\b", re.I), {"population", "people", "residents"}),
    (re.compile(r"\b(?:language)\b", re.I), {"language", "programming", "code"}),
    (re.compile(r"\b(?:medication|drug|dosage|prescribed)\b", re.I),
     {"medication", "drug", "dosage", "prescribed", "side", "effects", "frequency"}),
    (re.compile(r"\b(?:cause|reason|why)\b", re.I), {"cause", "reason", "why"}),
    (re.compile(r"\b(?:guilty|verdict|defendant)\b", re.I),
     {"guilty", "verdict", "defendant", "crime", "evidence"}),
]


class FilterStats:
    """Track what was filtered and why."""
    __slots__ = ("total_in", "meta_removed", "dedup_removed",
                 "irrelevant_removed", "total_out")

    def __init__(self):
        self.total_in = 0
        self.meta_removed = 0
        self.dedup_removed = 0
        self.irrelevant_removed = 0
        self.total_out = 0


def filter_claims_pre_labeling(
    claim_texts: list[str],
    question: str,
) -> tuple[list[str], FilterStats]:
    """Filter claims BEFORE labeling. Removes meta-claims and exact dupes."""
    stats = FilterStats()
    stats.total_in = len(claim_texts)

    result: list[str] = []
    seen: set[str] = set()

    for ct in claim_texts:
        ct = ct.strip()
        if not ct or len(ct) < 5:
            stats.meta_removed += 1
            continue
        if _META_PATTERNS.search(ct):
            stats.meta_removed += 1
            continue
        norm = _normalize(ct)
        if norm in seen:
            stats.dedup_removed += 1
            continue
        seen.add(norm)
        result.append(ct)

    stats.total_out = len(result)
    return result, stats


def filter_unknown_claims_post_labeling(
    claims: list[dict],
    question: str,
) -> tuple[list[dict], FilterStats]:
    """Filter unknown claims AFTER labeling.

    Keeps SUPPORTED and CONTRADICTS_EVIDENCE unconditionally (with dedup).
    For unknown labels, applies slot-aware relevance check.
    """
    stats = FilterStats()
    stats.total_in = len(claims)

    q_slots = _extract_question_slots(question)
    q_keywords = _extract_keywords(question)
    result: list[dict] = []
    seen: set[str] = set()

    for c in claims:
        label = c.get("label", "")
        ct = c.get("claim_text", "")

        # Always keep SUPPORTED and CONTRADICTS_EVIDENCE (dedup only)
        if label in ("SUPPORTED", "CONTRADICTS_EVIDENCE"):
            norm = _normalize(ct)
            if norm in seen:
                stats.dedup_removed += 1
                continue
            seen.add(norm)
            result.append(c)
            continue

        # Meta check
        if _META_PATTERNS.search(ct):
            stats.meta_removed += 1
            continue

        # Dedup
        norm = _normalize(ct)
        if norm in seen:
            stats.dedup_removed += 1
            continue
        seen.add(norm)

        # Relevance check for unknown claims
        if label in ("UNSUPPORTED", "NEEDS_INFO", "NOT_IN_EVIDENCE"):
            if not _is_slot_relevant(ct, q_slots, q_keywords):
                stats.irrelevant_removed += 1
                continue

        result.append(c)

    stats.total_out = len(result)
    return result, stats


# ── Slot-aware relevance ──────────────────────────────────────────────

def _extract_question_slots(question: str) -> set[str]:
    """Extract the requested information slots from a question.

    Returns a set of slot keywords. If no specific pattern matches,
    falls back to content keywords from the question.
    """
    slots: set[str] = set()
    for pattern, slot_keywords in _QUESTION_SLOT_PATTERNS:
        if pattern.search(question):
            slots |= slot_keywords

    # Always add explicit noun phrases from the question as slots
    # (e.g., "stock price" → {"stock", "price"})
    slots |= _extract_keywords(question)
    return slots


def _is_slot_relevant(
    claim_text: str,
    question_slots: set[str],
    question_keywords: set[str],
) -> bool:
    """Check if an unknown claim is about a slot the user actually asked for.

    Stricter logic:
    1. Extract what's missing from the claim ("launch location")
    2. Check if the missing subject's ATTRIBUTE matches the question's slot
    3. Shared subject words (e.g., "project", "launch") don't count —
       the attribute must match
    """
    # Extract what's being called missing/unknown
    missing_subject = _extract_missing_subject(claim_text)
    if missing_subject:
        ms_kw = _extract_keywords(missing_subject)
        # The missing subject's attribute words must overlap with question slots
        # (not just the shared entity/subject words)
        attr_overlap = ms_kw & question_slots
        if attr_overlap:
            # But filter out cases where only the entity name matches
            # e.g. "launch" in both "launch location" and "When did project launch?"
            # We need the ATTRIBUTE part to match, not just the subject
            # Heuristic: if the missing subject has ≥2 words, the NON-entity
            # word must be in the question slots
            ms_words = list(ms_kw)
            if len(ms_words) >= 2:
                # At least one non-subject word must match question slots
                subject_words = _extract_keywords(claim_text) - ms_kw
                attr_words = ms_kw - question_keywords  # words unique to the claim
                if attr_words and not (attr_words & question_slots):
                    return False  # attribute word not in question slots
            return True
        return False

    # Fallback: check direct keyword overlap (≥2 required)
    claim_kw = _extract_keywords(claim_text)
    if not claim_kw:
        return False
    return len(claim_kw & question_slots) >= 2


def _extract_missing_subject(text: str) -> str:
    """Extract what's being described as missing/unknown in a claim.

    "The launch location is not mentioned" → "launch location"
    "Side effects are not known" → "Side effects"
    """
    patterns = [
        re.compile(r"(?:the\s+)?(.+?)\s+(?:is|are|was|were)\s+(?:not|unknown|missing|unavailable)", re.I),
        re.compile(r"(?:no\s+)(.+?)(?:\s+(?:is|are|was|were)|\s*$)", re.I),
    ]
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1).strip()
    return ""


# ── SUPPORTED claim relevance filter (v0.4 experiment) ──────────────
# Downgrades SUPPORTED claims that don't share keywords with the question.
# This normalizes extraction across model sizes — large models over-extract
# tangentially-related facts; this filter prunes them before the gate.

def filter_supported_claims_by_relevance(
    claims: list[dict],
    question: str,
) -> list[dict]:
    """Downgrade SUPPORTED claims with no question-keyword overlap.

    If a SUPPORTED claim shares zero keywords with the question (after
    removing stop words), it is downgraded to UNSUPPORTED. This prevents
    large models from inflating the SUPPORTED count with irrelevant facts.

    If the question itself has no content keywords (e.g. "What happened?"),
    the check is skipped — all claims pass through.

    Returns: modified claims list (label may be changed, notes appended).
    """
    q_kw = _extract_question_content_keywords(question)
    if not q_kw:
        # Question has no content keywords — skip check
        return claims

    for c in claims:
        if c.get("label") != "SUPPORTED":
            continue

        ct = c.get("claim_text", "")
        claim_kw = _extract_keywords(ct)
        overlap = claim_kw & q_kw
        if not overlap:
            # Downgrade to UNSUPPORTED with note
            c["label"] = "UNSUPPORTED"
            c["evidence_pointers"] = []
            notes = c.get("notes", "")
            c["notes"] = notes + " [rel: no overlap with question keywords]"

    return claims


def _extract_question_content_keywords(question: str) -> set[str]:
    """Extract content keywords from a question, but be more permissive.

    Unlike _extract_keywords, this keeps 'what', 'which', 'many', 'much'
    when they appear in the question. These are often the only content
    words in short questions, and dropping them causes the relevance
    check to skip.
    """
    text = question.lower().strip()
    words = set(re.findall(r"\b\w{3,}\b", text))
    # Remove only true stop words, keep question words
    return words - {
        "the", "this", "that", "these", "those",
        "and", "but", "or", "nor",
        "for", "with", "from", "about", "than",
        "are", "was", "were", "been", "being",
        "have", "has", "had", "having",
        "can", "could", "will", "would", "should",
        "does", "did", "doing", "done",
    }


# ── Utilities ─────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    words = set(re.findall(r"\b\w{3,}\b", text.lower()))
    return words - _STOP_WORDS


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t
