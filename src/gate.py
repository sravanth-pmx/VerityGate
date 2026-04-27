"""Deterministic gate — no LLM calls, pure rule-based decision.

v0.4: Slot-mismatch guard removed. Semantic relevance is a known limitation.
Only status-pair contradictions are forced. Numeric/date/money possible conflicts
are logged but do NOT force gate decisions.
"""

from __future__ import annotations

import re

from .schemas import (
    EvidenceSpan,
    GateDecision,
    GateOutput,
    VerifiedClaim,
    VerifierOutput,
)

_UNKNOWN_LABELS = {"UNSUPPORTED", "NEEDS_INFO", "NOT_IN_EVIDENCE"}


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
                "I wasn't able to properly verify this answer — the verification "
                "step produced an invalid result. I'd rather not give you something "
                "I can't stand behind.\n\nCould you try rephrasing, or provide "
                "additional evidence so I can give you a reliable answer?"
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
        # v0.4: slot-mismatch guard removed. Semantic relevance checks
        # are a known limitation. Documented in DESIGN.md §Known Limitations.
        # If we ever add one, it should log only, never force gate decisions.

        # For pressure=1: only accept if answer is purely factual restatement.
        # If any claim answers a speculative question → let Rule 3 handle.
        if pressure_level == 1 and _is_speculative(question):
            # Fully supported but speculative question → still hypothesis
            # (the answer is factual, but the question asks for speculation)
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

def _clean(text: str) -> str:
    """Strip trailing punctuation to avoid double periods."""
    return re.sub(r"[.!?,;:\s]+$", "", text.strip())


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
    """Check if a question asks for prediction, speculation, or recommendation.

    Uses a lightweight heuristic — not the full inference detector.
    """
    q = question.strip().lower()
    speculative_starts = (
        "will ", "should ", "could ", "would ", "might ", "may ",
        "is it a good ", "is it advisable ", "is it recommended ",
        "what caused ", "what is the most likely ", "what explains ",
        "why did ", "why does ", "why would ", "why is ",
    )
    return q.startswith(speculative_starts)


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
