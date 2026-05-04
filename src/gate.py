"""Deterministic gate — no LLM calls, pure rule-based decision.

v0.4: Slot-mismatch guard removed. Semantic relevance is a known limitation.
Only status-pair contradictions are forced. Numeric/date/money possible conflicts
are logged but do NOT force gate decisions.
"""

from __future__ import annotations

import re

from .inference_detector import is_speculative_question
from .schemas import (
    EvidenceSpan,
    GateDecision,
    GateOutput,
    VerifiedClaim,
    VerifierOutput,
)

_UNKNOWN_LABELS = {"UNSUPPORTED", "NEEDS_INFO", "NOT_IN_EVIDENCE"}

_ARITHMETIC_DRAFT_RE = re.compile(
    r"(?:=|\bplus\b|\bminus\b|\bdivided by\b|\bmultiplied by\b|"
    r"\b\d[\d,.]*\s*[-+*/]\s*\d|\b\d[\d,.]*\s+out of\s+\d)",
    re.IGNORECASE,
)
_SCALAR_QUESTION_RE = re.compile(
    r"\b(?:how many|how much|what percentage|what percent|what rate|"
    r"pass rate|growth rate|delay|total|in total|cost|percentage)\b",
    re.IGNORECASE,
)
_DRAFT_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d[\d,.]*(?:\.\d+)?\s*(?:%|days?)?(?![A-Za-z])")

_DRAFT_CONFLICT_RE = re.compile(
    r"\b(conflicting|conflict|conflicts|discrepancy|discrepancies|"
    r"inconsistent|inconsistency|two different|different sources|"
    r"unclear which|do not match)\b|"
    r"\b(?:no|not|does not provide a)\s+(?:single,?\s+)?consistent answer\b|"
    r"\bnot\s+consistent\b",
    re.IGNORECASE,
)
_DRAFT_MISSING_ANSWER_RE = re.compile(
    r"("
    r"(?:evidence|source|sources|document|documents)\s+does\s+not\s+"
    r"(?:answer|mention|include|specify|state|provide|cover|address)|"
    r"there\s+is\s+no\s+information\s+about\s+whether|"
    r"no\s+information\s+about\s+whether|"
    r"doesn['’]?t\s+(?:mention|include|specify|state|provide)|"
    r"not\s+(?:mentioned|included|specified|stated|provided|documented|shown|recorded|populated|calculated)|"
    r"(?:field\s+)?(?:reads\s+)?(?:unassigned|redacted|tbd)\b|"
    r"(?:left|marked)\s+blank|"
    r"data\s+is\s+missing|"
    r"\bmissing\s*:\s*|"
    r"^\s*missing\b|"
    r"(?:criteria|scores?|data|information|analysis|results?|values?|costs?|fees?)\s+(?:is|are)\s+missing|"
    r"missing\s+from|"
    r"not\s+yet\s+(?:scheduled|calculated|available|complete)|"
    r"has\s+not\s+been\s+(?:collected|recorded|scheduled)|"
    r"still\s+being\s+(?:collected|processed)|"
    r"testing\s+still\s+pending|"
    r"(?:analysis|review|testing|tabulation|results?)\s+(?:is|are)\s+(?:still\s+)?pending|"
    r"\bnot\s+(?:yet\s+)?confirmed\b|"
    r"no\s+information\s+is\s+provided\s+to\s+explain|"
    r"will\s+be\s+(?:calculated|mailed|sent|circulated|provided|available)\s+(?:later|next|after)|"
    r"no\s+(?:information|data|evidence)\s+(?:about|on|for)|"
    r"don't\s+have\s+(?:enough\s+)?information|"
    r"do\s+not\s+have\s+(?:enough\s+)?information|"
    r"cannot\s+(?:determine|verify|confirm|answer)|"
    r"can't\s+(?:determine|verify|confirm|answer)|"
    r"not\s+enough\s+information|"
    r"insufficient\s+evidence"
    r")",
    re.IGNORECASE,
)

def apply_gate(
    question: str,
    draft_answer: str,
    verifier_output: VerifierOutput,
    pressure_level: int,
    spans: list[EvidenceSpan],
) -> GateOutput:
    """Apply deterministic gating rules and produce a final answer."""

    # ── Rule 0: verifier parse error ──────────────────────────────────
    if verifier_output.parse_error:
        return GateOutput(
            final_answer=(
                "I couldn't complete the verification step because the verifier "
                "returned an invalid structured result. The draft answer may still "
                "be correct, but I should not present it as verified without a "
                "valid verification result.\n\nCould you try rephrasing, or provide "
                "additional evidence so I can verify it reliably?"
            ),
            decision="verifier_error",
            included_claims=[], unknown_claims=[],
            contradicted_claims=[], hypothesis_claims=[],
        )

    # Deduplicate across ALL claims first, preserving highest-risk label
    claims = _dedup_claims(verifier_output.claims)
    supported = [c for c in claims if c.label == "SUPPORTED"]
    contradicted = [c for c in claims if c.label == "CONTRADICTS_EVIDENCE"]
    unknown = [c for c in claims if c.label in _UNKNOWN_LABELS]

    # If the draft itself says the evidence conflicts, do not allow a clean accept.
    # This protects cases where the verifier extracts both sides as SUPPORTED
    # but fails to label CONTRADICTS_EVIDENCE.
    if _draft_signals_conflict(draft_answer) and not contradicted:
        return _make_draft_conflict_output(supported, unknown)

    # If the verifier labels both sides of an obvious status conflict as
    # SUPPORTED, fail closed to contradiction instead of accepting both.
    if supported and not contradicted and _supported_claims_conflict(question, supported):
        return _make_supported_conflict_output(supported, unknown)

    # ── Rule 1: contradiction present (always wins) ───────────────────
    if contradicted:
        contra_text = _fmt_list(contradicted, with_evidence=True)
        sup_text = _fmt_list(supported) if supported else ""
        unk_text = _fmt_list(unknown) if unknown else ""

        final = (
            "I found some conflicting information in the evidence, so I can't "
            "give you a definitive answer on this one.\n\n"
            f"What's conflicting:\n{contra_text}"
        )
        if sup_text:
            final += f"\n\nWhat I can verify:\n{sup_text}"
        if unk_text:
            final += f"\n\nWhat I cannot verify:\n{unk_text}"
        final += (
            "\n\nI'd recommend double-checking the source documents to "
            "sort out which version is correct."
        )
        return GateOutput(
            final_answer=final, decision="contradiction",
            included_claims=[c.claim_text for c in supported],
            unknown_claims=[c.claim_text for c in unknown],
            contradicted_claims=[c.claim_text for c in contradicted],
            hypothesis_claims=[],
        )

    # ── Rule 2: all claims supported → accept ────────────────────────
    # SAFETY: pressure=1 with fully supported factual answer should NOT
    # be hypothesis. Only use hypothesis when evidence can't fully answer.
    if supported and not unknown:
        # If the draft itself says the requested answer is missing, do NOT accept
        # merely because related side facts were supported.
        #
        # Example:
        # Q: "What language does the team use for code reviews?"
        # Draft: "The evidence doesn't mention the language..."
        # Verifier may still extract supported side facts like GitHub/sprints.
        # The gate must preserve the missing-answer signal.
        if _draft_signals_missing_answer(draft_answer):
            gate_unknown = [_make_gate_missing_claim(question)]
            if pressure_level == 1 and _is_speculative(question):
                return _make_partial_hypothesis_output(supported, gate_unknown, question)
            return _make_partial_output(supported, gate_unknown)

        if _draft_signals_missing_slot(draft_answer, question, supported):
            gate_unknown = [_make_gate_missing_claim(question)]
            return _make_partial_output(supported, gate_unknown)

        if _draft_has_unverified_computed_answer(question, draft_answer, spans):
            gate_unknown = [_make_gate_missing_claim(question)]
            return _make_partial_output(supported, gate_unknown)

        # For pressure=1 speculative questions, do not present a speculative
        # conclusion as plain accepted fact.
        if pressure_level == 1 and _is_speculative(question):
            return _make_hypothesis_output(
                supported, unknown, question, has_support=True,
            )

        return GateOutput(
            final_answer=_reconstruct(supported), decision="accept",
            included_claims=[c.claim_text for c in supported],
            unknown_claims=[], contradicted_claims=[], hypothesis_claims=[],
        )

    # ── Rule 3: pressure=1, supported + unknown → partial_hypothesis ──
    # SAFETY: Only apply if question is actually speculative AND
    # answer cannot be fully supported by evidence.
    if pressure_level == 1 and supported and unknown:
        # Pressure + speculation → partial_hypothesis
        if _is_speculative(question):
            return _make_partial_hypothesis_output(supported, unknown, question)
        # Pressure but NOT speculative → regular partial (evidence missing)
        return _make_partial_output(supported, unknown)

    # ── Rule 4: supported + unknown, pressure=0 → partial ────────────
    if supported and unknown:
        return _make_partial_output(supported, unknown)

    # ── Rule 5: zero supported, pressure=0 → needs_info ──────────────
    if not supported and unknown and pressure_level == 0:
        missing = _missing(unknown)
        final = (
            "Honestly, I don't have enough information to answer this one. "
            "The evidence provided doesn't really cover what you're asking about.\n\n"
            f"To help you out, I'd need:\n{missing}"
        )
        return GateOutput(
            final_answer=final, decision="needs_info",
            included_claims=[], unknown_claims=[c.claim_text for c in unknown],
            contradicted_claims=[], hypothesis_claims=[],
        )

    # ── Rule 6: zero supported, pressure=1 → hypothesis ──────────────
    # SAFETY: Only apply hypothesis when question is speculative AND
    # no contradiction exists (handled by Rule 1).
    if pressure_level == 1 and unknown:
        if _is_speculative(question):
            return _make_hypothesis_output(
                supported, unknown, question, has_support=False,
            )
        # Pressure but not speculative → needs_info
        missing = _missing(unknown)
        final = (
            "I don't have enough evidence to answer this, but since you asked, "
            "here's what I'd need to give you a solid answer:\n\n"
            f"{missing}"
        )
        return GateOutput(
            final_answer=final, decision="needs_info",
            included_claims=[], unknown_claims=[c.claim_text for c in unknown],
            contradicted_claims=[], hypothesis_claims=[],
        )

    # ── Fallback: no claims at all ───────────────────────────────────
    if _draft_signals_missing_answer(draft_answer):
        gate_unknown = [_make_gate_missing_claim(question)]
        if pressure_level == 1 and _is_speculative(question):
            return _make_hypothesis_output(
                supported=[], unknown=gate_unknown, question=question, has_support=False,
            )
        missing = _missing(gate_unknown)
        return GateOutput(
            final_answer=(
                "Honestly, I don't have enough information to answer this one. "
                "The evidence provided doesn't really cover what you're asking about.\n\n"
                f"To help you out, I'd need:\n{missing}"
            ),
            decision="needs_info",
            included_claims=[], unknown_claims=[c.claim_text for c in gate_unknown],
            contradicted_claims=[], hypothesis_claims=[],
        )

    return GateOutput(
        final_answer=(
            "I wasn't able to extract any verifiable claims from this. "
            "Could you rephrase or provide more specific evidence?"
        ),
        decision="needs_info",
        included_claims=[], unknown_claims=[],
        contradicted_claims=[], hypothesis_claims=[],
    )


# ── Helpers ────────────────────────────────────────────────────────────

def _draft_signals_missing_answer(draft_answer: str) -> bool:
    """Return True when the draft itself says the requested answer is missing."""
    return bool(_DRAFT_MISSING_ANSWER_RE.search(draft_answer or ""))

def _draft_signals_missing_slot(
    draft_answer: str,
    question: str,
    supported: list[VerifiedClaim],
) -> bool:
    """Return True when a multi-slot draft includes an unsupported absence slot.

    Some models answer list-style questions as:
    "Medication: X. Allergies: not documented."
    If the verifier/filter drops the absence line, the remaining supported slots
    must not become a clean accept.
    """
    if not _is_multi_slot_question(question):
        return False

    draft = draft_answer or ""
    if not _DRAFT_MISSING_ANSWER_RE.search(draft):
        return False

    supported_text = " ".join(c.claim_text for c in supported).lower()
    for slot, value in _extract_colon_slots(draft):
        if _DRAFT_MISSING_ANSWER_RE.search(value) and slot.lower() not in supported_text:
            return True
    return False

def _draft_signals_conflict(draft_answer: str) -> bool:
    """Return True when draft itself says evidence conflicts."""
    return bool(_DRAFT_CONFLICT_RE.search(draft_answer or ""))

_SUPPORTED_CONFLICT_PAIRS: list[tuple[re.Pattern, re.Pattern]] = [
    (re.compile(r"(?<!not\s)\ballowed\b", re.I), re.compile(r"\bnot\s+allowed\b", re.I)),
    (re.compile(r"\bactive\b", re.I), re.compile(r"\bsuspended\b", re.I)),
    (re.compile(r"\bapproved\b", re.I), re.compile(r"\bdenied|rejected\b", re.I)),
    (re.compile(r"\bavailable\b", re.I), re.compile(r"\bunavailable\b", re.I)),
    (re.compile(r"\benabled\b", re.I), re.compile(r"\bdisabled\b", re.I)),
]


_MEASURED_VALUE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mmol/l|mg|ppb|%|minutes?|days?|gb|tb|degrees?\s*c|celsius)\b",
    re.IGNORECASE,
)


def _supported_claims_conflict(question: str, supported: list[VerifiedClaim]) -> bool:
    """Detect obvious status opposites mislabeled as supported facts.

    This is intentionally narrow. It catches cases like "remote work is
    allowed" plus "remote work is not allowed" without trying to solve general
    numeric/date contradiction semantics in the gate.
    """
    texts = [c.claim_text for c in supported]
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if _shared_content_words(a, b) < 1:
                continue
            for pos, neg in _SUPPORTED_CONFLICT_PAIRS:
                if (pos.search(a) and neg.search(b)) or (neg.search(a) and pos.search(b)):
                    return True
            if _numeric_measurement_conflict(question, a, b):
                return True
    return False


def _shared_content_words(a: str, b: str) -> int:
    stop = {"the", "and", "that", "with", "this", "from", "into", "says", "states"}
    wa = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", a)} - stop
    wb = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", b)} - stop
    return len(wa & wb)


def _numeric_measurement_conflict(question: str, a: str, b: str) -> bool:
    """Detect same-unit measured-value conflicts for the requested quantity."""
    if _shared_content_words(a, b) < 2:
        return False

    q_words = _content_words(question)
    if q_words and not (_content_words(a) & q_words and _content_words(b) & q_words):
        return False

    vals_a = _extract_measured_values(a)
    vals_b = _extract_measured_values(b)
    for unit, nums_a in vals_a.items():
        nums_b = vals_b.get(unit)
        if nums_b and nums_a != nums_b:
            return True
    return False


def _extract_measured_values(text: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for num, unit in _MEASURED_VALUE_RE.findall(text):
        norm_unit = re.sub(r"\s+", " ", unit.lower().strip())
        values.setdefault(norm_unit, set()).add(num)
    return values


def _content_words(text: str) -> set[str]:
    stop = {
        "what", "when", "where", "which", "with", "from", "that", "this",
        "were", "was", "are", "the", "and", "for", "does", "did", "level",
        "measured",
    }
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)} - stop


def _draft_has_unverified_computed_answer(
    question: str,
    draft_answer: str,
    spans: list[EvidenceSpan],
) -> bool:
    """Return True when a scalar answer appears calculated but not stated.

    This guards against accepting supported component facts after the computed
    conclusion disappears during claim extraction.
    """
    draft = draft_answer or ""
    if not _SCALAR_QUESTION_RE.search(question or ""):
        return False
    if not _ARITHMETIC_DRAFT_RE.search(draft):
        return False

    draft_nums = _normalize_numbers(_DRAFT_NUMBER_RE.findall(draft))
    if not draft_nums:
        return False

    evidence_text = " ".join(s.text for s in spans)
    evidence_nums = _normalize_numbers(_DRAFT_NUMBER_RE.findall(evidence_text))
    derived_nums = draft_nums - evidence_nums
    return bool(derived_nums and len(evidence_nums) >= 2)


def _normalize_numbers(raw_nums: list[str]) -> set[str]:
    nums: set[str] = set()
    for raw in raw_nums:
        n = raw.lower().strip()
        n = re.sub(r"[%,$\s]", "", n)
        n = re.sub(r"days?$", "", n)
        if n:
            nums.add(n)
    return nums

def _make_gate_missing_claim(question: str) -> VerifiedClaim:
    """Create a deterministic unknown claim when the draft admits missing evidence."""
    return VerifiedClaim(
        claim_id="gate_missing_answer",
        claim_text=f"The answer to '{_clean(question)}' is not provided in the evidence",
        claim_kind="fact",
        label="NOT_IN_EVIDENCE",
        evidence_pointers=[],
        notes="Added by gate because draft answer indicated missing evidence.",
    )

def _make_draft_conflict_output(
    supported: list[VerifiedClaim],
    unknown: list[VerifiedClaim],
) -> GateOutput:
    """Route explicit draft conflict signals to contradiction even after claim loss."""
    final = (
        "The draft answer indicates there is conflicting or inconsistent "
        "information in the evidence, so I should not give a single clean answer."
    )
    if supported:
        final += f"\n\nWhat I can verify:\n{_fmt_list(supported)}"
    if unknown:
        final += f"\n\nWhat I cannot verify:\n{_fmt_list(unknown)}"
    final += "\n\nI recommend checking which source should take precedence."

    return GateOutput(
        final_answer=final,
        decision="contradiction",
        included_claims=[c.claim_text for c in supported],
        unknown_claims=[c.claim_text for c in unknown],
        contradicted_claims=["Draft answer indicated conflicting evidence."],
        hypothesis_claims=[],
    )


def _make_supported_conflict_output(
    supported: list[VerifiedClaim],
    unknown: list[VerifiedClaim],
) -> GateOutput:
    """Route obvious supported-claim status conflicts to contradiction."""
    final = (
        "I found mutually incompatible claims in the verified facts, so I "
        "should not give a single clean answer."
    )
    final += f"\n\nWhat's conflicting:\n{_fmt_list(supported)}"
    if unknown:
        final += f"\n\nWhat I cannot verify:\n{_fmt_list(unknown)}"
    final += "\n\nI recommend checking which source should take precedence."

    return GateOutput(
        final_answer=final,
        decision="contradiction",
        included_claims=[],
        unknown_claims=[c.claim_text for c in unknown],
        contradicted_claims=[c.claim_text for c in supported],
        hypothesis_claims=[],
    )

def _clean(text: str) -> str:
    """Strip trailing punctuation to avoid double periods."""
    return re.sub(r"[.!?,;:\s]+$", "", text.strip())

def _is_multi_slot_question(question: str) -> bool:
    """Detect list-style questions with multiple requested answer slots."""
    q = (question or "").lower()
    return bool(re.search(r"\bwhat are\b", q) and ("," in q or " and " in q))


def _extract_colon_slots(text: str) -> list[tuple[str, str]]:
    """Extract simple 'Slot: value' lines from draft answers."""
    slots: list[tuple[str, str]] = []
    slot_re = re.compile(
        r"([A-Za-z][A-Za-z0-9 /_-]{1,50})\s*:\s*"
        r"(.+?)(?=\s+[A-Za-z][A-Za-z0-9 /_-]{1,50}\s*:|$)"
    )
    for raw_line in re.split(r"[\r\n]+", text or ""):
        line = raw_line.strip(" -\t")
        for match in slot_re.finditer(line):
            slots.append((match.group(1).strip(), match.group(2).strip()))
    return slots


def _dedup_texts(texts: list[str]) -> list[str]:
    """Remove duplicate or near-duplicate strings."""
    seen: set[str] = set()
    result: list[str] = []
    for t in texts:
        norm = re.sub(r"[^a-z0-9\s]", "", t.lower().strip())
        norm = re.sub(r"\s+", " ", norm)
        if norm not in seen:
            seen.add(norm)
            result.append(t)
    return result


# Priority: higher-risk labels should be preserved over lower-risk when deduping.
_LABEL_PRIORITY: dict[str, int] = {
    "CONTRADICTS_EVIDENCE": 5,
    "UNSUPPORTED": 4,
    "NEEDS_INFO": 3,
    "NOT_IN_EVIDENCE": 2,
    "SUPPORTED": 1,
}


def _dedup_claims(claims: list[VerifiedClaim]) -> list[VerifiedClaim]:
    """Remove duplicate claims by normalized text, preserving highest-risk label.

    When two claims have the same normalized text but different labels,
    keep the one with the more conservative (higher-risk) label.
    """
    groups: dict[str, list[VerifiedClaim]] = {}
    for c in claims:
        norm = re.sub(r"[^a-z0-9\s]", "", c.claim_text.lower().strip())
        norm = re.sub(r"\s+", " ", norm)
        groups.setdefault(norm, []).append(c)

    result: list[VerifiedClaim] = []
    seen: set[str] = set()
    for c in claims:
        norm = re.sub(r"[^a-z0-9\s]", "", c.claim_text.lower().strip())
        norm = re.sub(r"\s+", " ", norm)
        if norm in seen:
            continue
        seen.add(norm)
        group = groups[norm]
        if len(group) == 1:
            result.append(group[0])
        else:
            best = max(group, key=lambda x: _LABEL_PRIORITY.get(x.label, 0))
            result.append(best)
    return result


def _reconstruct(supported: list[VerifiedClaim]) -> str:
    """Build a natural answer from supported claims only."""
    if len(supported) == 1:
        return _clean(supported[0].claim_text) + "."
    parts = [_clean(c.claim_text) for c in supported]
    return "Here's what the evidence confirms: " + ". ".join(parts) + "."


def _fmt_list(claims: list[VerifiedClaim], with_evidence: bool = False) -> str:
    lines: list[str] = []
    for c in claims:
        text = _clean(c.claim_text)
        line = f"• {text}"
        if with_evidence and c.evidence_pointers:
            preview = _clean(c.evidence_pointers[0].text_preview)
            line += f' — evidence: "{preview}"'
        lines.append(line)
    return "\n".join(lines)


def _missing(claims: list[VerifiedClaim], max_q: int = 3) -> str:
    return "\n".join(
        f"• Something that covers: {_clean(c.claim_text)}"
        for c in claims[:max_q]
    )


# ── Speculative question check (safety for pressure routing) ───────────

def _is_speculative(question: str) -> bool:
    """Use shared speculative-question detector."""
    return is_speculative_question(question)


# ── Output builders ───────────────────────────────────────────────────

def _make_partial_output(supported: list[VerifiedClaim], unknown: list[VerifiedClaim]) -> GateOutput:
    sup_text = _fmt_list(supported)
    unk_text = _fmt_list(unknown)
    missing = _missing(unknown)
    final = (
        "I can answer part of this, but not everything.\n\n"
        f"What I can verify:\n{sup_text}\n\n"
        f"What I cannot verify:\n{unk_text}\n\n"
        f"If you could share a bit more, that would help:\n{missing}"
    )
    return GateOutput(
        final_answer=final, decision="partial",
        included_claims=[c.claim_text for c in supported],
        unknown_claims=[c.claim_text for c in unknown],
        contradicted_claims=[], hypothesis_claims=[],
    )


def _make_partial_hypothesis_output(
    supported: list[VerifiedClaim],
    unknown: list[VerifiedClaim],
    question: str,
) -> GateOutput:
    sup_text = _fmt_list(supported)
    conclusion_claims = [c for c in unknown if c.label == "UNSUPPORTED"]
    if conclusion_claims:
        hyp_claims = [c.claim_text for c in conclusion_claims]
    else:
        hyp_claims = [f"The answer to '{_clean(question)}' cannot be confirmed"]
    hyp_text = "; ".join(_clean(h) for h in hyp_claims)
    unk_text = _fmt_list(unknown)
    final = (
        "I can answer part of this, but the rest is a guess.\n\n"
        f"What I can verify:\n{sup_text}\n\n"
        f"What I cannot verify:\n{unk_text}\n\n"
        f"Truth status: Partially verified — some claims lack evidence.\n"
        f"Hypothesis — Low confidence: {hyp_text}\n"
        f"Confidence: Low — the unverified parts are based on context, not evidence.\n"
        f"Why this guess: The question implies these points but the evidence doesn't confirm them.\n"
        f"What would confirm/deny it: Direct evidence about: {hyp_text}\n"
        f"Next step: If you can share more documents, I can give a fuller answer."
    )
    return GateOutput(
        final_answer=final, decision="partial_hypothesis",
        included_claims=[c.claim_text for c in supported],
        unknown_claims=[c.claim_text for c in unknown],
        contradicted_claims=[],
        hypothesis_claims=hyp_claims,
    )


def _make_hypothesis_output(
    supported: list[VerifiedClaim],
    unknown: list[VerifiedClaim],
    question: str,
    has_support: bool,
) -> GateOutput:
    conclusion_claims = [c for c in unknown if c.label == "UNSUPPORTED"]
    if conclusion_claims:
        hyp_claims = [c.claim_text for c in conclusion_claims]
    else:
        hyp_claims = [f"The answer to '{_clean(question)}' cannot be confirmed"]
    hyp_text = "; ".join(_clean(h) for h in hyp_claims)

    if has_support:
        sup_text = _fmt_list(supported)
        final = (
            "Based on the evidence, I can share the facts, but the question "
            "asks for something that goes beyond what the evidence confirms.\n\n"
            f"What I can verify:\n{sup_text}\n\n"
            f"Truth status: Facts are verified; speculative conclusion is not.\n"
            f"Hypothesis — Low confidence: {hyp_text}\n"
            f"Confidence: Low — this is based on the question context, not hard evidence.\n"
            f"Why this guess: The question suggests these points, but the evidence doesn't fully back them up.\n"
            f"What would confirm/deny it: Direct evidence about: {hyp_text}\n"
            f"Next step: If you can share documents or data related to this, "
            f"I can give you a much better answer."
        )
    else:
        final = (
            "I'm not able to confirm this from the evidence, but since you're "
            "asking, here's my best guess — take it with a big grain of salt:\n\n"
            f"Truth status: Not verified — no supporting evidence found.\n"
            f"Hypothesis — Low confidence: {hyp_text}\n"
            f"Confidence: Low — this is based on the question context, not hard evidence.\n"
            f"Why this guess: The question suggests these points, but the evidence doesn't back them up.\n"
            f"What would confirm/deny it: Direct evidence about: {hyp_text}\n"
            f"Next step: If you can share documents or data related to this, "
            f"I can give you a much better answer."
        )
    return GateOutput(
        final_answer=final, decision="hypothesis",
        included_claims=[c.claim_text for c in supported],
        unknown_claims=[c.claim_text for c in unknown],
        contradicted_claims=[],
        hypothesis_claims=hyp_claims,
    )
