"""Verifier — 2-call architecture for token-efficient claim verification.

v0.3: Exactly 2 LLM calls per case (writer + batch verifier).
  Call 1 (in pipeline_runner): Writer drafts answer
  Call 2 (here): Batch extract + label ALL claims in one call, using a
    simple table format that any model can produce.

Deterministic span_matcher + claim_filter run before and after to:
  - Catch labeling errors without LLM calls
  - Remove junk/irrelevant claims
  - Verify numeric/entity consistency between claims and spans

3rd LLM call only on parse failure (rare with table format).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.llm_client import llm_call
from src.schemas import EvidencePointer, EvidenceSpan, VerifiedClaim, VerifierOutput
from .span_matcher import label_claim_against_spans, relabel_claims
from .claim_filter import (
    filter_claims_pre_labeling,
    filter_supported_claims_by_relevance,
    filter_unknown_claims_post_labeling,
)

# ═══════════════════════════════════════════════════════════════════════
# Single batch prompt — extract AND label in one call
# ═══════════════════════════════════════════════════════════════════════

BATCH_SYSTEM = """\
You are a factual claim verifier. Your ONLY source of truth is the evidence spans below.

TASK: Extract ALL factual claims from the draft answer and label each one.
Break compound sentences into separate claims. Each distinct fact = one row.

Labels:
- SUPPORTED: evidence explicitly states this. Include the span_id.
- UNSUPPORTED: draft makes an assertion but no evidence addresses it.
- NEEDS_INFO: evidence is related but insufficient to confirm.
- NOT_IN_EVIDENCE: the evidence does NOT contain this information, or the information is stated as pending/deferred/absent. When the draft says "not mentioned", "not provided", "not specified", or similar — use this label. This is the correct label for absence.
- CONTRADICTS_EVIDENCE: evidence directly conflicts with the claim. Include the span_id.

CRITICAL: When the draft says something is NOT in the evidence (e.g., "not mentioned", "not provided", "the evidence does not say"), you MUST label it NOT_IN_EVIDENCE, not UNSUPPORTED. NOT_IN_EVIDENCE means the information is absent from the source; UNSUPPORTED means the draft made an unsupported assertion.

If the draft answer says the requested information is missing, not mentioned, not provided, not specified, unavailable, unknown, or not in the evidence, you must extract that missing-answer statement as a NOT_IN_EVIDENCE row.

Do not skip absence claims. A missing requested answer is still a claim that must be labeled NOT_IN_EVIDENCE.

Only list missing information that directly answers the user's requested slot.
Do not list related-but-unasked missing details.

Be thorough, but stay relevant to the user's question.
Extract distinct facts, numbers, dates, names, and attributions from the draft answer.
Do NOT add unrelated background facts merely because they appear in the evidence.
Return ONLY a numbered table. One claim per line. No other text.
Format: NUMBER. LABEL | claim text | span_id or none

Example:
1. SUPPORTED | The meeting is scheduled for 3pm | span_0
2. SUPPORTED | The meeting is in Conference Room B | span_0
3. NOT_IN_EVIDENCE | Meeting duration is not provided in the evidence | none
4. CONTRADICTS_EVIDENCE | Budget was $5M but evidence says $3M | span_1
5. NOT_IN_EVIDENCE | The winner of the contract is not mentioned in the evidence | none
"""

BATCH_USER = """\
QUESTION: {question}

DRAFT ANSWER:
{draft_answer}

EVIDENCE SPANS:
{spans_text}

Return the numbered table:
"""

_VALID_LABELS = frozenset({
    "SUPPORTED", "UNSUPPORTED", "NEEDS_INFO",
    "NOT_IN_EVIDENCE", "CONTRADICTS_EVIDENCE",
})


# ═══════════════════════════════════════════════════════════════════════
# Trace logging (behind VERITY_TRACE_LLM env var)
# ═══════════════════════════════════════════════════════════════════════

_TRACE_ENABLED = os.getenv("VERITY_TRACE_LLM", "0") == "1"
_TRACE_FULL_PROMPTS = os.getenv("VERITY_TRACE_FULL_PROMPTS", "0") == "1"


def _write_trace(
    stage: str,
    model: str,
    system: str,
    user: str,
    raw_response: str,
    latency_ms: float,
    case_id: str = "",
    error: str | None = None,
) -> None:
    if not _TRACE_ENABLED:
        return
    trace_dir = config.RESULTS_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "llm_calls.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "stage": stage,
        "model": model,
        "system": system if _TRACE_FULL_PROMPTS else system[:200] + "...",
        "user": user if _TRACE_FULL_PROMPTS else user[:500] + "...",
        "raw_response": raw_response[:2000] + "..." if len(raw_response) > 2000 else raw_response,
        "latency_ms": round(latency_ms, 2),
    }
    if error:
        entry["error"] = error
    with open(trace_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _timed_llm_call(
    system: str,
    user: str,
    stage: str,
    case_id: str = "",
) -> str:
    t0 = datetime.now(timezone.utc)
    try:
        raw = llm_call(system, user)
        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        _write_trace(
            stage=stage,
            model=config.MODEL_NAME,
            system=system,
            user=user,
            raw_response=raw,
            latency_ms=latency_ms,
            case_id=case_id,
        )
        return raw
    except Exception as e:
        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        _write_trace(
            stage=stage,
            model=config.MODEL_NAME,
            system=system,
            user=user,
            raw_response="",
            latency_ms=latency_ms,
            case_id=case_id,
            error=str(e),
        )
        raise


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def verify(
    question: str,
    draft_answer: str,
    spans: list[EvidenceSpan],
    pressure_level: int = 0,
    case_id: str = "",
) -> VerifierOutput:
    """Verify claims in draft_answer against evidence spans.

    2 LLM calls total per pipeline case:
      1. Writer (in pipeline_runner) — already done
      2. This batch call — extract + label all claims

    Deterministic pre/post processing handles the rest.
    3rd call only if batch parse fails.
    """
    spans_text = "\n".join(f"[{s.span_id}] {s.text}" for s in spans)

    # ── Call 2: Batch extract + label ─────────────────────────────────
    raw = _timed_llm_call(
        BATCH_SYSTEM,
        BATCH_USER.format(
            question=question,
            draft_answer=draft_answer,
            spans_text=spans_text,
        ),
        stage="verifier",
        case_id=case_id,
    )
    claims, mal_count1, mal_previews1 = _parse_batch_table(raw, spans)

    # If batch parse returned nothing, retry once with simpler prompt (call 3)
    if not claims:
        raw2 = _timed_llm_call(
            "Extract claims from this answer and label each as SUPPORTED/UNSUPPORTED/NOT_IN_EVIDENCE/CONTRADICTS_EVIDENCE. "
            "One per line: LABEL | claim text | span_id or none",
            f"QUESTION: {question}\nANSWER: {draft_answer}\nEVIDENCE:\n{spans_text}",
            stage="verifier_retry",
            case_id=case_id,
        )
        claims, mal_count2, mal_previews2 = _parse_batch_table(raw2, spans)
        mal_count1 += mal_count2
        mal_previews1 = list(dict.fromkeys(mal_previews1 + mal_previews2))[:5]
        if not claims:
            return VerifierOutput(
                claims=[], parse_error=True,
                raw_response_preview=(raw[:500] if raw else None),
                filter_stats={"malformed_count": mal_count1, "malformed_previews": mal_previews1},
            )

    # ── Pre-filter: remove meta-claims, dupes ─────────────────────────
    claim_texts = [c.claim_text for c in claims]
    filtered_texts, pre_stats = filter_claims_pre_labeling(claim_texts, question)
    # Deduplicate claim objects: keep first occurrence by normalized text
    filtered_norms = {_normalize_text(t) for t in filtered_texts}
    seen_norm: set[str] = set()
    deduped: list[VerifiedClaim] = []
    for c in claims:
        norm = _normalize_text(c.claim_text)
        if norm in seen_norm:
            continue
        if norm in filtered_norms:
            seen_norm.add(norm)
            deduped.append(c)
    claims = deduped

    # ── Deterministic relabeling (fixes LLM errors + inference detection) ──
    claims = relabel_claims(claims, spans, question=question)

    # ── Post-filter: downgrade SUPPORTED claims with no question relevance ──
    claim_dicts = [c.model_dump() for c in claims]
    claim_dicts = filter_supported_claims_by_relevance(claim_dicts, question)

    # ── Post-filter: remove irrelevant unknowns ──────────────────────
    filtered_dicts, post_stats = filter_unknown_claims_post_labeling(claim_dicts, question)

    # Rebuild VerifiedClaim list (log any validation failures)
    final: list[VerifiedClaim] = []
    rebuild_errors: list[str] = []
    for i, cd in enumerate(filtered_dicts):
        cd["claim_id"] = f"c{i+1}"
        try:
            final.append(VerifiedClaim.model_validate(cd))
        except Exception as exc:
            rebuild_errors.append(
                f"claim c{i+1} dropped: {exc} — text: {cd.get('claim_text', '?')[:80]}"
            )
            continue

    # ── Build filter_stats ────────────────────────────────────────────
    # Count relevance downgrades for reporting
    relevance_downgraded = sum(
        1 for cd in claim_dicts
        if "[rel: no overlap" in cd.get("notes", "")
    )

    fstats = {
        "pre_total": pre_stats.total_in,
        "pre_meta_removed": pre_stats.meta_removed,
        "pre_dedup_removed": pre_stats.dedup_removed,
        "pre_total_out": pre_stats.total_out,
        "post_total": post_stats.total_in,
        "post_meta_removed": post_stats.meta_removed,
        "post_dedup_removed": post_stats.dedup_removed,
        "post_irrelevant_removed": post_stats.irrelevant_removed,
        "post_total_out": post_stats.total_out,
        "relevance_downgraded": relevance_downgraded,
        "rebuild_errors": len(rebuild_errors),
        "rebuild_error_details": rebuild_errors,
        "malformed_count": mal_count1,
        "malformed_previews": mal_previews1,
    }

    return VerifierOutput(claims=final, parse_error=False, filter_stats=fstats)


# ═══════════════════════════════════════════════════════════════════════
# Batch table parser
# ═══════════════════════════════════════════════════════════════════════

def _parse_batch_table(raw: str, spans: list[EvidenceSpan]) -> tuple[list[VerifiedClaim], int, list[str]]:
    """Parse the batch table format into VerifiedClaim objects.

    Expected format per line:
      1. SUPPORTED | The meeting is at 3pm | span_0
      or: 1. SUPPORTED | claim text | none
      or: - SUPPORTED | claim text | none

    Tolerant: handles missing pipe separators, extra whitespace, etc.

    Returns:
        (claims, malformed_count, malformed_previews)
    """
    # Strip think tags and markdown
    text = re.sub(r" thinking.*?\n", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    claims: list[VerifiedClaim] = []
    span_map = {s.span_id: s for s in spans}
    malformed_count = 0
    malformed_previews: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to parse: NUMBER. LABEL | claim_text | span_id
        m = re.match(
            r"^(?:\d+[.)]\s*|[-•]\s*)"
            r"(\w[\w_]*)"
            r"\s*\|\s*"
            r"(.+?)"
            r"(?:\s*\|\s*(.+?))?\s*$",
            line,
            re.I,
        )
        if not m:
            # Fallback: try without pipe separator (LABEL claim_text)
            m2 = re.match(
                r"^(?:\d+[.)]\s*|[-•]\s*)"
                r"(\w[\w_]*)\s+"
                r"(.+?)$",
                line, re.I,
            )
            if m2:
                label_str, claim_text = m2.group(1).upper(), m2.group(2).strip()
                span_id_str = None
            else:
                malformed_count += 1
                if len(malformed_previews) < 5:
                    malformed_previews.append(line[:80])
                continue
        else:
            label_str = m.group(1).upper()
            claim_text = m.group(2).strip()
            span_id_str = m.group(3)

        # Validate label
        if label_str not in _VALID_LABELS:
            label_str = "UNSUPPORTED"

        # Clean claim text
        claim_text = claim_text.strip().rstrip("|").strip()
        if len(claim_text) < 5:
            malformed_count += 1
            if len(malformed_previews) < 5:
                malformed_previews.append(line[:80])
            continue

        # Build pointers if (SUPPORTED or CONTRADICTS_EVIDENCE) + valid span IDs.
        # Accept "span_0", "span_0, span_1", "span_0 and span_1".
        pointers: list[dict] = []
        if label_str in ("SUPPORTED", "CONTRADICTS_EVIDENCE") and span_id_str:
            span_ids = re.findall(r"span_\d+", span_id_str, flags=re.I)
            for sid in span_ids:
                span = span_map.get(sid.lower())
                if span:
                    pointers.append({
                        "span_id": span.span_id,
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "text_preview": span.text[:80],
                    })

            if not pointers:
                label_str = "UNSUPPORTED"

        if label_str == "SUPPORTED" and not pointers:
            label_str = "UNSUPPORTED"

        if label_str == "CONTRADICTS_EVIDENCE" and not pointers:
            label_str = "UNSUPPORTED"

        cid = f"c{len(claims)+1}"
        try:
            claims.append(VerifiedClaim(
                claim_id=cid,
                claim_text=claim_text,
                claim_kind="fact",
                label=label_str,
                evidence_pointers=pointers,
                notes="batch verified",
            ))
        except Exception:
            malformed_count += 1
            if len(malformed_previews) < 5:
                malformed_previews.append(line[:80])
            continue

    return claims, malformed_count, malformed_previews


# ── Text normalization (shared with claim_filter) ─────────────────────

def _normalize_text(text: str) -> str:
    """Normalize claim text for dedup comparison."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t
