"""
File: src/report/calibration.py
Purpose: Analytical tool designed to measure the calibration of the model's confidence 
against its actual accuracy. It uses a probe-based system to evaluate whether the model's 
internal confidence levels (extracted during verification) correlate with the factual 
correctness of its claims, identifying patterns of overconfidence or unnecessary caution 
that can inform prompt engineering and model selection.
"""

from __future__ import annotations

import re

from src.llm_client import llm_call

# ── Probe system prompt (matches verifier BATCH_SYSTEM format) ────────
_PROBE_SYSTEM = """\
You are a factual claim verifier. Your ONLY source of truth is the evidence spans below.

TASK: Extract factual claims from the draft answer and label each one.

Labels:
- SUPPORTED: evidence explicitly states this. Include the span_id.
- UNSUPPORTED: draft asserts this but no evidence addresses it.
- NEEDS_INFO: evidence is related but insufficient.
- NOT_IN_EVIDENCE: the question asks for info that is absent/pending/deferred.
- CONTRADICTS_EVIDENCE: evidence directly conflicts with the claim.

Return ONLY a numbered table. One claim per line. No other text.
Format: NUMBER. LABEL | claim text | span_id or none

Example:
1. SUPPORTED | The meeting is at 3pm | span_0
2. NOT_IN_EVIDENCE | Meeting duration is not provided | none
"""

_PROBES = [
    # Probe 1: obvious SUPPORTED — claim matches evidence exactly
    {
        "user": (
            "QUESTION: What color is the car?\n\n"
            "DRAFT ANSWER:\nThe car is red.\n\n"
            "EVIDENCE SPANS:\n"
            "[span_0] The car is red.\n\n"
            "Return the numbered table:\n"
        ),
        "expect_label": "SUPPORTED",
    },
    # Probe 2: obvious NOT_IN_EVIDENCE — question asks about price,
    # evidence only mentions color
    {
        "user": (
            "QUESTION: What is the car's price?\n\n"
            "DRAFT ANSWER:\nThe price is not mentioned in the evidence.\n\n"
            "EVIDENCE SPANS:\n"
            "[span_0] The car is red.\n\n"
            "Return the numbered table:\n"
        ),
        "expect_label": "NOT_IN_EVIDENCE",
    },
]

_VALID_LABELS = frozenset({
    "SUPPORTED", "UNSUPPORTED", "NEEDS_INFO",
    "NOT_IN_EVIDENCE", "CONTRADICTS_EVIDENCE",
})

# Line pattern matching verifier._parse_batch_table regex
_LINE_RE = re.compile(
    r"^(?:\d+[.)]\s*|[-•]\s*)"
    r"(\w[\w_]*)"
    r"\s*\|\s*"
    r"(.+?)"
    r"(?:\s*\|\s*(span_\d+|none|n/a))?\s*$",
    re.I,
)


def run_calibration() -> tuple[bool, list[str]]:
    """Run calibration probes. Returns (passed, list_of_warnings)."""
    warnings: list[str] = []
    table_ok = 0

    for i, probe in enumerate(_PROBES):
        try:
            raw = llm_call(_PROBE_SYSTEM, probe["user"])
            # Strip think tags and markdown fences
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r"^```\w*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            # Parse table lines
            parsed_labels: list[str] = []
            for line in cleaned.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = _LINE_RE.match(line)
                if m:
                    label = m.group(1).upper()
                    if label in _VALID_LABELS:
                        parsed_labels.append(label)

            if parsed_labels:
                table_ok += 1
                first_label = parsed_labels[0]
                expected = probe["expect_label"]
                if first_label != expected:
                    warnings.append(
                        f"Probe {i+1}: expected label={expected}, got label={first_label}. "
                        f"Deterministic post-processing will compensate."
                    )
            else:
                warnings.append(
                    f"Probe {i+1}: response was not a valid table. "
                    f"Preview: {cleaned[:120]!r}"
                )
        except Exception as e:
            warnings.append(f"Probe {i+1}: error — {e}")

    passed = table_ok >= 1  # At least 1 of 2 probes returned valid table
    if not passed:
        warnings.insert(
            0,
            "CALIBRATION FAILED: Model cannot produce valid verifier table. "
            "Results will be unreliable.",
        )

    return passed, warnings
