"""Deterministic inference detector — flags claims that are inferences, not facts.

v0.3.1: Based on CogniBench cognitive statement taxonomy (arxiv:2505.20767),
GME epistemic modality taxonomy (arxiv:2106.08037), and BioScope hedge cues.

Four-tier detection:
  Tier 1: Claim-level — epistemic hedges, logical leap connectors, causal inference
  Tier 2: Logical leap connectors — therefore, thus, based on these findings
  Tier 3: Deontic/normative — should, recommended, indicated
  Tier 4: Question-level — speculative/predictive/normative question detection

Only overrides SUPPORTED → UNSUPPORTED. Never touches other labels.
Conservative: prefers missed inference over false positive on factual claims.
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════
# Tier 1: Epistemic hedges — "state of knowledge" modals (GME taxonomy)
# High precision: these almost never appear in purely factual claims
# ═══════════════════════════════════════════════════════════════════════

_EPISTEMIC_HEDGES = re.compile(
    r"\bmost likely\b|\bmost probable\b|\bprobably\b|\bpresumably\b|"
    r"\bapparently\b|\bplausibly\b|"
    # Epistemic modal constructions (NOT bare "may" — too ambiguous)
    r"\bmay (?:have|be|represent|indicate|suggest)\b|"
    r"\bmight (?:have|be|represent|indicate)\b|"
    r"\bcould (?:be|represent|indicate|have)\b|"
    # Evidential verbs — GME "state of knowledge" triggers
    r"\bsuggests?\s+(?:that |a |an |the |\w+ing |\w+ence )|"
    r"\bimplies?\b|\bpoints? to\b|"
    r"\bwould suggest\b|\bwould indicate\b|"
    r"\bappears? to (?:be|have|show|indicate)\b|"
    r"\bseems? to (?:be|have|show|indicate)\b|"
    # "supports" as epistemic — when followed by inference (not evidence citation)
    r"\bsupports?\s+(?:a |an |the )?(?:diagnosis|conclusion|hypothesis|notion|idea|interpretation|finding)\b|"
    r"\bsupports?\s+(?:a |an |the )?(?:acute|chronic|bacterial|viral|infectious|inflammatory)\b|"
    # Clinical/diagnostic inference (BioScope cues)
    r"\bconsistent with\b|\bin keeping with\b|\bcompatible with\b|"
    r"\bconcerning for\b|"
    r"\bcannot rule out\b|\bnot ruled out\b|"
    # Causal inference
    r"\blikely caused by\b|\bmost likely caused by\b|"
    r"\bbest explains?\b|\bsecondary to\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════
# Tier 2: Logical leap connectors — claim draws a conclusion from evidence
# ═══════════════════════════════════════════════════════════════════════

_LOGICAL_LEAP = re.compile(
    r"\btherefore\b|\bthus\b|\bhence\b|\bconsequently\b|"
    r"\bthis suggests?\b|\bthis indicates?\b|\bthis implies?\b|"
    r"\bthis is consistent with\b|\bthis points to\b|"
    r"\bwe can conclude\b|\bit can be concluded\b|"
    r"\bbased on (?:these|this|the) (?:findings?|evidence|data|symptoms?)\b|"
    r"\btaken together\b|\boverall\b.*\bsuggests?\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════
# Tier 3: Predictive/speculative + Deontic/normative
# ═══════════════════════════════════════════════════════════════════════

_PREDICTIVE = re.compile(
    r"\bwill likely\b|\bis likely to\b|\bis expected to\b|"
    r"\bhas a (?:strong |good |high )?chance\b|"
    r"\bpotential(?:ly)? to\b|\bprospects? (?:are|look|seem)\b|"
    r"\bpoised to\b|\bon track to\b|"
    r"\bshould (?:be able|succeed|continue|grow|improve)\b|"
    r"\bforecast|predict",
    re.IGNORECASE,
)

_DEONTIC = re.compile(
    r"\bshould (?:be |consider |start |seek |get )\b|"
    r"\brecommended?\b|\badvised?\b|\bwarranted\b|"
    r"\bis indicated\b|\bare indicated\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# Tier 4: Question-level — speculative question detection
# Source: CogniBench + Typed-RAG question taxonomy
# ═══════════════════════════════════════════════════════════════════════

_SPECULATIVE_QUESTION = re.compile(
    # Normative: "Should we invest?" "Should the defendant..."
    r"^should\s+(?:we|the|he|she|they|I)\b|"
    # Predictive: "Will the product..." "Will it rain..."
    r"^will\s+(?:the|this|it)\b|"
    # Causal inference: "What caused..." "What explains..."
    r"^what\s+(?:caused|is causing|explains|led to|is the (?:most )?likely (?:cause|diagnosis))\b|"
    # Diagnostic: "Is the defendant guilty..."
    r"^is\s+the\s+(?:defendant|patient|suspect)\s+\w+|"
    # Evaluative: "Is it advisable..."
    r"^is\s+(?:it|this)\s+(?:advisable|recommended|safe|a good)\b|"
    # Why questions seeking causal explanation
    r"^why\s+(?:did|does|would|is)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def is_speculative_question(question: str) -> bool:
    """Return True if the question is speculative/predictive/normative."""
    return bool(_SPECULATIVE_QUESTION.search(question.strip()))


def detect_inference(
    claim_text: str,
    question: str,
) -> tuple[bool, str]:
    """Detect if a SUPPORTED claim is actually an inference.

    Returns: (is_inference, reason)
    Only call this for claims with label == "SUPPORTED".
    """
    ct = claim_text.strip()

    # Tier 1: Epistemic hedges in the claim itself
    m = _EPISTEMIC_HEDGES.search(ct)
    if m:
        return True, f"epistemic hedge: '{m.group().strip()}'"

    # Tier 2: Logical leap connector
    m = _LOGICAL_LEAP.search(ct)
    if m:
        return True, f"logical leap: '{m.group().strip()}'"

    # Tier 3a: Predictive/speculative language
    m = _PREDICTIVE.search(ct)
    if m:
        return True, f"predictive: '{m.group().strip()}'"

    # Tier 3b: Deontic/normative
    m = _DEONTIC.search(ct)
    if m:
        return True, f"deontic: '{m.group().strip()}'"

    # Tier 4: Speculative question → check if claim answers it (not just restates data)
    if is_speculative_question(question):
        if _is_answering_speculative_question(ct, question):
            return True, "answers speculative question"

    return False, ""


# ── Number detection for pure-data check ──────────────────────────────
_HAS_NUMBER = re.compile(r"[\$€£]\s*\d|\d[\d,.]*\s*(?:%|°|million|billion|mg|hPa|units?|employees?)")


def _is_answering_speculative_question(claim: str, question: str) -> bool:
    """Heuristic: is this claim trying to answer a speculative question?

    We want to flag "The startup will succeed" but NOT flag
    "The startup has $2M ARR" — both may appear as claims under
    "Should we invest in this startup?"

    Key insight: claims that contain specific numbers/measurements from
    evidence are factual restatements. Claims with evaluative/conclusory
    language without specific data are answering the speculative question.
    """
    cl = claim.lower()

    # Safe harbor: claims with specific numbers/measurements are data restatements
    # e.g. "$2M ARR", "30% growth", "45,000 miles", "12,000/µL"
    if _HAS_NUMBER.search(claim):
        # But still flag if it ALSO has strong evaluative framing
        strong_eval = re.compile(
            r"\bguilty\b|\binnocent\b|\bwill\s+(?:succeed|fail|rain)\b|"
            r"\bdiagnosis\b|\bcause\s+(?:of|is)\b",
            re.IGNORECASE,
        )
        if not strong_eval.search(cl):
            return False

    # Claims with evaluative/conclusory words → answering the question
    evaluative = re.compile(
        r"\bsuccess\w*\b|\bfail\w*\b|\bguilty\b|\binnocent\b|"
        r"\bcredib\w*\b|\bviable\b|\bprofitable\b|"
        r"\bgood\s+(?:chance|position|sign)\b|"
        r"\bbad\s+(?:sign|outlook)\b|"
        r"\badds?\s+credibility\b|"
        r"\binfection\b|\bdisease\b|\bdisorder\b|\bsyndrome\b|"
        r"\bcause\s+(?:of|is)\b|\bcaused\s+by\b|"
        r"\bdiagnosis\b|\betiology\b|"
        r"\bwill\s+(?:rain|succeed|fail|grow|decline)\b",
        re.IGNORECASE,
    )
    if evaluative.search(cl):
        return True

    # Claims that frame evidence in evaluative terms
    framing = re.compile(
        r"\bshows?\s+(?:strong|weak|positive|negative|promising)\b|"
        r"\bdemonstrates?\s+(?:strong|weak|positive|negative)\b",
        re.IGNORECASE,
    )
    if framing.search(cl):
        return True

    return False
