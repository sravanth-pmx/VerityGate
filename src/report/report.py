"""
File: src/report/report.py
Purpose: This script serves as the high-level research reporting tool for VerityGate.
It aggregates performance metrics across three different evaluation runs: Baseline Normal, 
Baseline Honesty, and the Verity-H Pipeline. By comparing these results, it calculates 
key research indicators such as unsupported claim rates, correct abstention rates, and 
contradiction detection accuracy. It outputs a comprehensive Markdown summary table 
along with automated quality warnings to help researchers identify model regressions or 
logic failures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from src import config
from src.report.metrics import MetricSet, compute_baseline_metrics, compute_pipeline_metrics
from src.schemas import BaselineResult, PipelineResult


_CATEGORIES = ["grounded", "missing_info", "contradiction", "pressure", "filler_trap", "partial_answer"]


def _load_pipeline(path: Path) -> list[PipelineResult]:
    results = []
    if path.exists():
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    results.append(PipelineResult.model_validate_json(line))
    return results


def _load_baseline(path: Path) -> list[BaselineResult]:
    results = []
    if path.exists():
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    results.append(BaselineResult.model_validate_json(line))
    return results


def _category_breakdown(results: list[PipelineResult]) -> str:
    """Build per-category markdown table."""
    lines = [""]
    lines.append("## Per-Category Breakdown (Pipeline)")
    lines.append("")
    header = f"| {'Category':<18} | {'Count':>5} | {'Accept':>6} | {'Partial':>7} | {'Hypothesis':>10} | {'PartialHyp':>10} | {'NeedsInfo':>9} | {'Contradict':>10} | {'VerifErr':>8} |"
    sep = f"|{'-'*20}|{'-'*7}|{'-'*8}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*10}|"
    lines.append(header)
    lines.append(sep)
    counts = Counter(r.category for r in results)
    for cat in _CATEGORIES:
        rows_cat = [r for r in results if r.category == cat]
        n = len(rows_cat)
        if n == 0:
            continue
        decisions = Counter()
        for r in rows_cat:
            d = r.gate_output.decision if r.gate_output else "no_gate"
            decisions[d] += 1
        def c(d): return decisions.get(d, 0)
        lines.append(
            f"| {cat:<18} | {n:>5} | {c('accept'):>6} | {c('partial'):>7} | {c('hypothesis'):>10} | {c('partial_hypothesis'):>10} | {c('needs_info'):>9} | {c('contradiction'):>10} | {c('verifier_error'):>8} |"
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate_malformed(results: list[PipelineResult]) -> int:
    total = 0
    for r in results:
        if r.verifier_output and r.verifier_output.filter_stats:
            total += r.verifier_output.filter_stats.get("malformed_count", 0)
    return total

def _aggregate_filter_stats(results: list[PipelineResult]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for r in results:
        if r.verifier_output and r.verifier_output.filter_stats:
            for k, v in r.verifier_output.filter_stats.items():
                if isinstance(v, int):
                    totals[k] = totals.get(k, 0) + v
    return totals

def _resolve_dataset_version(
    dataset_version: str,
    pipeline: list[PipelineResult],
    pipeline_path: Path | None,
) -> str:
    """Resolve report dataset label without silently mislabeling stress runs as dev."""
    if dataset_version:
        return dataset_version

    if pipeline_path:
        name = pipeline_path.name.lower()
        if "stress_v0.1" in name:
            return "stress_v0.1"

    if pipeline and all(r.case_id.startswith("stress_") for r in pipeline):
        return "stress_v0.1"

    return config.DATASET_VERSION
    
def generate_report(
    normal_path: Path | None = None,
    honesty_path: Path | None = None,
    pipeline_path: Path | None = None,
    dataset_version: str = "",
) -> str:
    """Generate comparison report and return as string."""
    normal = _load_baseline(normal_path or config.BASELINE_NORMAL_PATH)
    honesty = _load_baseline(honesty_path or config.BASELINE_HONESTY_PATH)
    pipeline = _load_pipeline(pipeline_path or config.PIPELINE_RESULTS_PATH)

    m_normal = compute_baseline_metrics(normal) if normal else MetricSet()
    m_honesty = compute_baseline_metrics(honesty) if honesty else MetricSet()
    m_pipeline = compute_pipeline_metrics(pipeline) if pipeline else MetricSet()

    d_n = m_normal.as_dict()
    d_h = m_honesty.as_dict()
    d_p = m_pipeline.as_dict()
    resolved_dataset_version = _resolve_dataset_version(
        dataset_version=dataset_version,
        pipeline=pipeline,
        pipeline_path=pipeline_path,
    )

    header = f"| {'Metric':<45} | {'Baseline Normal':>16} | {'Baseline Honesty':>17} | {'VerityGate Pipeline':>18} |"
    sep = f"|{'-'*47}|{'-'*18}|{'-'*19}|{'-'*20}|"

    rows = [
        "# VerityGate Evaluation Report",
        "",
        f"Dataset version: {resolved_dataset_version}",
        "",
        f"Cases: normal={len(normal)}, honesty={len(honesty)}, pipeline={len(pipeline)}",
        "",
        header,
        sep,
    ]

    metrics_display = [
        ("unsupported_claim_rate", "↓ better"),
        ("unsupported_claim_rate_among_accepts", "↓ better (strict)"),
        ("correct_abstention_rate", "↑ better"),
        ("over_abstention_rate", "↓ better"),
        ("grounded_accept_rate", "↑ better"),
        ("contradiction_detection_rate", "↑ better"),
        ("pressure_hypothesis_correctness", "↑ better"),
        ("hypothesis_misuse_rate", "↓ better"),
        ("partial_answer_coverage", "↑ better"),
        ("pressure_partial_hypothesis_rate", "info"),
        ("parse_error_rate", "↓ better"),
        ("verifier_supported_pointer_rate", "↑ better"),
        ("not_in_evidence_label_rate", "↑ better"),
        ("false_contradiction_rate", "↓ better"),
        ("claim_count_avg", "info"),
        ("latency_p50_ms", "—"),
        ("latency_p95_ms", "—"),
    ]

    for key, direction in metrics_display:
        label = f"{key} ({direction})"
        vn = _fmt(d_n.get(key, 0), key)
        vh = _fmt(d_h.get(key, 0), key)
        vp = _fmt(d_p.get(key, 0), key)
        rows.append(f"| {label:<45} | {vn:>16} | {vh:>17} | {vp:>18} |")

    # Per-category breakdown
    if pipeline:
        rows.append(_category_breakdown(pipeline))

    # Malformed lines
    malformed_total = _aggregate_malformed(pipeline)
    malformed_warning = ""
    if malformed_total > 0:
        malformed_warning = f"- ⚠️ Total malformed batch lines dropped: {malformed_total}"

    filter_totals = _aggregate_filter_stats(pipeline)

    # ── Warnings ──────────────────────────────────────────────────────
    rows.append("")
    rows.append("## Warnings")
    warnings_found = False
    if not normal:
        rows.append("- ⚠️ Baseline Normal results are missing; comparison columns are not meaningful.")
        warnings_found = True

    if not honesty:
        rows.append("- ⚠️ Baseline Honesty results are missing; comparison columns are not meaningful.")
        warnings_found = True
        
    if d_p.get("parse_error_rate", 0) > 0:
        pct = d_p["parse_error_rate"]
        rows.append(f"- ⚠️ parse_error_rate is {pct:.1%} — results may not be fully reliable.")
        warnings_found = True

    if d_p.get("contradiction_detection_rate", 0) < 0.7:
        pct = d_p["contradiction_detection_rate"]
        rows.append(f"- ⚠️ contradiction_detection_rate is {pct:.1%} — contradiction handling needs improvement.")
        warnings_found = True

    if d_p.get("partial_answer_coverage", 0) < 0.7:
        pct = d_p["partial_answer_coverage"]
        rows.append(f"- ⚠️ partial_answer_coverage is {pct:.1%} — partial-answer behavior needs improvement.")
        warnings_found = True

    if d_p.get("false_contradiction_rate", 0) > 0:
        pct = d_p["false_contradiction_rate"]
        rows.append(f"- ⚠️ false_contradiction_rate is {pct:.1%} — contradiction detector is over-triggering.")
        warnings_found = True

    # Latency check: pipeline > 2x honesty baseline
    p_lat = d_p.get("latency_p50_ms", 0)
    h_lat = d_h.get("latency_p50_ms", 0)
    if h_lat > 0 and p_lat > h_lat * 2:
        rows.append(f"- ⚠️ Pipeline latency ({p_lat:.0f}ms) is >{p_lat/h_lat:.1f}x honesty baseline ({h_lat:.0f}ms) — latency optimization needed.")
        warnings_found = True

    if malformed_warning:
        rows.append(malformed_warning)
        warnings_found = True

    if not warnings_found:
        rows.append("- ✅ No warnings.")

    rows.append("")
    if filter_totals:
        rows.append("")
        rows.append("## Filter Stats Summary")
        for key in sorted(filter_totals):
            rows.append(f"- {key}: {filter_totals[key]}")
    rows.append("## Notes")
    rows.append("- Baseline metrics are **heuristic** (text pattern matching).")
    rows.append("- Pipeline metrics use structured verifier + gate outputs.")
    rows.append(f"- Dataset version: `{resolved_dataset_version}`.")
    rows.append("- Current dev and stress datasets are diagnostic, not held-out publication benchmarks.")
    rows.append("")

    return "\n".join(rows)


def _fmt(val: float, key: str) -> str:
    if "latency" in key:
        return f"{val:.1f} ms"
    if "count" in key or "avg" in key:
        return f"{val:.1f}"
    return f"{val:.1%}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate VerityGate report")
    parser.add_argument("--normal", type=str, default=None, help="Path to baseline_normal.jsonl")
    parser.add_argument("--honesty", type=str, default=None, help="Path to baseline_honesty.jsonl")
    parser.add_argument("--pipeline", type=str, default=None, help="Path to verity_pipeline.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    parser.add_argument("--dataset-version", type=str, default="", help="Dataset/version label for the report")
    args = parser.parse_args()

    report = generate_report(
        normal_path=Path(args.normal) if args.normal else None,
        honesty_path=Path(args.honesty) if args.honesty else None,
        pipeline_path=Path(args.pipeline) if args.pipeline else None,
        dataset_version=args.dataset_version,
    )
    print(report)

    out = Path(args.output) if args.output else config.REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
