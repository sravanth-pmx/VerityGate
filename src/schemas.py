"""Pydantic schemas for Project Verity-H v0.3.1."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── Gold case ──────────────────────────────────────────────────────────
class GoldCase(BaseModel):
    id: str
    category: Literal[
        "grounded",
        "missing_info",
        "contradiction",
        "pressure",
        "filler_trap",
        "partial_answer",
    ]
    question: str
    evidence_text: str
    pressure_level: int = 0
    expected_supported_claims: list[str] = Field(default_factory=list)
    expected_unknowns: list[str] = Field(default_factory=list)
    expected_contradictions: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Evidence spans ─────────────────────────────────────────────────────
class EvidenceSpan(BaseModel):
    span_id: str
    text: str
    start_char: int
    end_char: int


# ── Verifier output ───────────────────────────────────────────────────
class EvidencePointer(BaseModel):
    span_id: str
    start_char: int
    end_char: int
    text_preview: str


ClaimKind = Literal[
    "fact",
    "number",
    "date_time",
    "attribution",
    "conditional",
    "causal",
    "definition",
    "negation",
    "comparative",
    "quantified",
]

ClaimLabel = Literal[
    "SUPPORTED",
    "UNSUPPORTED",
    "NEEDS_INFO",
    "NOT_IN_EVIDENCE",
    "CONTRADICTS_EVIDENCE",
]

# Labels that MUST have evidence_pointers to be valid
_POINTER_REQUIRED_LABELS = frozenset({"SUPPORTED", "CONTRADICTS_EVIDENCE"})


class VerifiedClaim(BaseModel):
    claim_id: str
    claim_text: str
    claim_kind: ClaimKind
    label: ClaimLabel
    evidence_pointers: list[EvidencePointer] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _grounded_labels_need_pointers(self) -> VerifiedClaim:
        if self.label in _POINTER_REQUIRED_LABELS and not self.evidence_pointers:
            raise ValueError(
                f"Claim {self.claim_id!r} is {self.label} but has no evidence_pointers"
            )
        return self


class VerifierOutput(BaseModel):
    claims: list[VerifiedClaim] = Field(default_factory=list)
    parse_error: bool = False
    raw_response_preview: str | None = None
    filter_stats: dict = Field(default_factory=dict)


# ── Gate output ────────────────────────────────────────────────────────
GateDecision = Literal[
    "accept",
    "partial",
    "partial_hypothesis",
    "needs_info",
    "contradiction",
    "hypothesis",
    "verifier_error",
]


class GateOutput(BaseModel):
    final_answer: str
    decision: GateDecision
    included_claims: list[str] = Field(default_factory=list)
    unknown_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    hypothesis_claims: list[str] = Field(default_factory=list)


# ── Pipeline result (one row in output JSONL) ─────────────────────────
class PipelineResult(BaseModel):
    case_id: str
    category: str
    question: str
    draft_answer: str
    pressure_level: int = 0
    expected_supported_claims: list[str] = Field(default_factory=list)
    expected_unknowns: list[str] = Field(default_factory=list)
    expected_contradictions: list[str] = Field(default_factory=list)
    verifier_output: VerifierOutput | None = None
    gate_output: GateOutput | None = None
    error: str | None = None
    latency_ms: float = 0.0


class BaselineResult(BaseModel):
    case_id: str
    category: str
    question: str
    answer: str
    pressure_level: int = 0
    expected_supported_claims: list[str] = Field(default_factory=list)
    expected_unknowns: list[str] = Field(default_factory=list)
    expected_contradictions: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
