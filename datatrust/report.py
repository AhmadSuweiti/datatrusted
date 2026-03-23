"""
AuditReport — the central result object returned by :func:`datatrust.audit`.

Holds all sub-results, computes a transparent trust score (0–100), and
supports export to dict, Markdown, and HTML.

Scoring logic
-------------
Start at 100 and subtract penalties for detected issues. The scoring table
is intentionally visible here so it is easy to tune.
"""

from __future__ import annotations

import html as html_lib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datatrust.models import (
    DuplicateInfo,
    LeakageHint,
    MissingInfo,
    OutlierInfo,
    SchemaReport,
    TargetInfo,
    ValidationResult,
)
from datatrust.utils import pct_str


# ---------------------------------------------------------------------------
# Scoring constants — tweak these to change the trust score behaviour
# ---------------------------------------------------------------------------

class _Penalty:
    # Missing values per column
    MISSING_LOW = 2      # 5 %  < missing ≤ 20 %
    MISSING_MED = 5      # 20 % < missing ≤ 50 %
    MISSING_HIGH = 10    # missing > 50 %

    # Full-row duplicates
    DUP_MINOR = 2        # 0 < duplicate_pct ≤ 1 %
    DUP_MOD = 5          # 1 % < duplicate_pct ≤ 5 %
    DUP_SEVERE = 10      # duplicate_pct > 5 %

    # Validation rule violations (per violation, capped)
    VIOLATION_EACH = 3
    VIOLATION_CAP = 20

    # Schema issues (per issue, capped)
    SCHEMA_EACH = 2
    SCHEMA_CAP = 10

    # Target issues
    TARGET_MISSING_LOW = 5    # 2 % < target missing ≤ 10 %
    TARGET_MISSING_HIGH = 10  # target missing > 10 %
    TARGET_IMBALANCE = 5      # imbalance ratio > 5

    # Leakage hints
    LEAKAGE_HIGH = 5
    LEAKAGE_CAP = 15

    # Outlier warnings (mild signal)
    OUTLIER_EACH = 1
    OUTLIER_CAP = 5


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    """Structured result returned by :func:`datatrust.audit`.

    All individual sub-reports are stored as attributes. The :attr:`score`
    is computed lazily from them.
    """

    shape: tuple
    """(rows, cols) of the audited DataFrame."""

    schema_report: Optional[SchemaReport] = None
    missing_info: Optional[MissingInfo] = None
    duplicate_info: Optional[DuplicateInfo] = None
    outlier_infos: List[OutlierInfo] = field(default_factory=list)
    target_info: Optional[TargetInfo] = None
    leakage_hints: List[LeakageHint] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None

    # -----------------------------------------------------------------
    # Core properties
    # -----------------------------------------------------------------

    @property
    def score(self) -> int:
        """Trust score from 0 to 100. Higher is better.

        Starts at 100 and deducts points based on detected issues.
        See :class:`_Penalty` for the scoring table.
        """
        return _compute_score(self)

    @property
    def warnings(self) -> List[str]:
        """Human-readable list of issues found during the audit."""
        return _collect_warnings(self)

    @property
    def summary(self) -> str:
        """One-paragraph plain-text summary of the audit results."""
        rows, cols = self.shape
        lines = [
            f"Dataset: {rows} rows × {cols} columns. Trust score: {self.score}/100.",
        ]

        if self.missing_info:
            n = len(self.missing_info.columns_above_threshold)
            if n:
                lines.append(
                    f"Missing values: {n} column(s) exceed the "
                    f"{pct_str(self.missing_info.threshold)} threshold."
                )
            else:
                lines.append("Missing values: all columns are within threshold.")

        if self.duplicate_info:
            d = self.duplicate_info
            if d.full_row_duplicates:
                lines.append(
                    f"Duplicates: {d.full_row_duplicates} full-row duplicate(s) "
                    f"({pct_str(d.duplicate_pct)})."
                )
            else:
                lines.append("Duplicates: none found.")

        if self.schema_report and self.schema_report.issue_count:
            lines.append(
                f"Schema: {self.schema_report.issue_count} issue(s) found "
                f"(e.g. numeric-as-string, unparsed datetimes)."
            )

        if self.validation_result and not self.validation_result.is_valid:
            lines.append(
                f"Validation: {self.validation_result.violation_count} rule violation(s)."
            )

        if self.target_info:
            ti = self.target_info
            if ti.missing_pct > 0:
                lines.append(
                    f"Target '{ti.column}': {pct_str(ti.missing_pct)} missing labels."
                )
            if ti.imbalance_ratio and ti.imbalance_ratio > 5:
                lines.append(
                    f"Target '{ti.column}': class imbalance ratio {ti.imbalance_ratio:.1f}x."
                )

        if self.leakage_hints:
            high = [h for h in self.leakage_hints if h.severity == "high"]
            if high:
                lines.append(
                    f"Leakage hints: {len(high)} high-severity hint(s) — review manually."
                )

        return " ".join(lines)

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dict (JSON-safe)."""
        result: Dict[str, Any] = {
            "shape": {"rows": self.shape[0], "cols": self.shape[1]},
            "score": self.score,
            "warnings": self.warnings,
        }

        if self.schema_report:
            result["schema"] = {
                "dtypes": self.schema_report.column_dtypes,
                "issues": [
                    {
                        "column": i.column,
                        "issue_type": i.issue_type,
                        "description": i.description,
                        "suggestion": i.suggestion,
                    }
                    for i in self.schema_report.issues
                ],
            }

        if self.missing_info:
            mi = self.missing_info
            result["missing"] = {
                "total_missing": mi.total_missing,
                "overall_pct": round(mi.overall_missing_pct, 4),
                "by_column": {
                    col: {"count": mi.missing_by_column[col], "pct": round(mi.missing_pct_by_column[col], 4)}
                    for col in mi.missing_by_column
                },
                "columns_above_threshold": mi.columns_above_threshold,
                "threshold": mi.threshold,
            }

        if self.duplicate_info:
            di = self.duplicate_info
            result["duplicates"] = {
                "full_row_duplicates": di.full_row_duplicates,
                "duplicate_pct": round(di.duplicate_pct, 4),
                "id_column_duplicates": di.id_column_duplicates,
            }

        if self.outlier_infos:
            result["outliers"] = [
                {
                    "column": o.column,
                    "outlier_count": o.outlier_count,
                    "outlier_pct": round(o.outlier_pct, 4),
                    "lower_bound": round(o.lower_bound, 4),
                    "upper_bound": round(o.upper_bound, 4),
                }
                for o in self.outlier_infos
            ]

        if self.target_info:
            ti = self.target_info
            result["target"] = {
                "column": ti.column,
                "missing_count": ti.missing_count,
                "missing_pct": round(ti.missing_pct, 4),
                "unique_count": ti.unique_count,
                "is_likely_classification": ti.is_likely_classification,
                "class_counts": ti.class_counts,
                "imbalance_ratio": ti.imbalance_ratio,
                "cardinality_warning": ti.cardinality_warning,
            }

        if self.leakage_hints:
            result["leakage_hints"] = [
                {
                    "column": h.column,
                    "hint_type": h.hint_type,
                    "severity": h.severity,
                    "description": h.description,
                }
                for h in self.leakage_hints
            ]

        if self.validation_result:
            vr = self.validation_result
            result["validation"] = {
                "is_valid": vr.is_valid,
                "violation_count": vr.violation_count,
                "violations": [
                    {
                        "rule": v.rule,
                        "column": v.column,
                        "description": v.description,
                        "affected_rows": v.affected_rows,
                        "sample_values": [str(s) for s in v.sample_values],
                    }
                    for v in vr.violations
                ],
            }

        return result

    def to_markdown(self, path: Optional[str] = None) -> str:
        """Render the report as a Markdown string.

        Parameters
        ----------
        path:
            If provided, write the Markdown to this file path.

        Returns
        -------
        str
            The rendered Markdown.
        """
        lines: List[str] = []
        rows, cols = self.shape

        lines += [
            "# datatrust Audit Report",
            "",
            f"**Trust Score: {self.score} / 100**",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {rows:,} |",
            f"| Columns | {cols:,} |",
            f"| Score | {self.score}/100 |",
            "",
        ]

        # Summary
        lines += ["## Summary", "", self.summary, ""]

        # Warnings
        if self.warnings:
            lines += ["## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Schema
        if self.schema_report and self.schema_report.issues:
            lines += ["## Schema Issues", ""]
            lines += ["| Column | Issue | Description |", "|--------|-------|-------------|"]
            for issue in self.schema_report.issues:
                lines.append(
                    f"| `{issue.column}` | {issue.issue_type} | {issue.description} |"
                )
            lines.append("")

        # Missing
        if self.missing_info:
            mi = self.missing_info
            lines += [
                "## Missing Values",
                "",
                f"- Total missing cells: {mi.total_missing:,} "
                f"({pct_str(mi.overall_missing_pct)} of all cells)",
                f"- Threshold: {pct_str(mi.threshold)}",
                "",
            ]
            above = mi.columns_above_threshold
            if above:
                lines += ["**Columns above threshold:**", ""]
                lines += ["| Column | Missing Count | Missing % |", "|--------|---------------|-----------|"]
                for col in above:
                    lines.append(
                        f"| `{col}` | {mi.missing_by_column[col]:,} "
                        f"| {pct_str(mi.missing_pct_by_column[col])} |"
                    )
                lines.append("")

        # Duplicates
        if self.duplicate_info:
            di = self.duplicate_info
            lines += [
                "## Duplicates",
                "",
                f"- Full-row duplicates: {di.full_row_duplicates:,} "
                f"({pct_str(di.duplicate_pct)})",
            ]
            if di.id_column_duplicates:
                lines.append("")
                lines += ["**ID column duplicates:**", ""]
                for col, count in di.id_column_duplicates.items():
                    lines.append(f"- `{col}`: {count} duplicate row(s)")
            lines.append("")

        # Outliers
        if self.outlier_infos:
            lines += ["## Outliers (IQR method)", ""]
            lines += ["| Column | Outlier Count | Outlier % | Lower Bound | Upper Bound |",
                      "|--------|---------------|-----------|-------------|-------------|"]
            for o in self.outlier_infos:
                lines.append(
                    f"| `{o.column}` | {o.outlier_count:,} | {pct_str(o.outlier_pct)} "
                    f"| {o.lower_bound:.4g} | {o.upper_bound:.4g} |"
                )
            lines.append("")

        # Target
        if self.target_info:
            ti = self.target_info
            lines += [
                "## Target Analysis",
                "",
                f"- Column: `{ti.column}`",
                f"- Missing labels: {ti.missing_count} ({pct_str(ti.missing_pct)})",
                f"- Unique values: {ti.unique_count}",
                f"- Likely classification: {'Yes' if ti.is_likely_classification else 'No'}",
            ]
            if ti.imbalance_ratio is not None:
                lines.append(f"- Imbalance ratio: {ti.imbalance_ratio:.2f}x")
            if ti.class_counts:
                lines += ["", "**Class distribution:**", ""]
                lines += ["| Class | Count |", "|-------|-------|"]
                for cls, cnt in list(ti.class_counts.items())[:20]:
                    lines.append(f"| {cls} | {cnt:,} |")
            lines.append("")

        # Leakage hints
        if self.leakage_hints:
            lines += ["## Leakage Hints (Heuristic)", ""]
            lines += ["| Column | Type | Severity | Description |",
                      "|--------|------|----------|-------------|"]
            for h in self.leakage_hints:
                lines.append(
                    f"| `{h.column}` | {h.hint_type} | {h.severity} | {h.description} |"
                )
            lines.append("")

        # Validation
        if self.validation_result and not self.validation_result.is_valid:
            lines += ["## Validation Violations", ""]
            lines += ["| Rule | Column | Affected Rows | Description |",
                      "|------|--------|---------------|-------------|"]
            for v in self.validation_result.violations:
                lines.append(
                    f"| {v.rule} | `{v.column}` | {v.affected_rows:,} | {v.description} |"
                )
            lines.append("")

        lines += [
            "---",
            "*Generated by [datatrust](https://github.com/yourusername/datatrust)*",
        ]

        md = "\n".join(lines)

        if path:
            Path(path).write_text(md, encoding="utf-8")

        return md

    def to_html(self, path: Optional[str] = None) -> str:
        """Render the report as a self-contained HTML string.

        Parameters
        ----------
        path:
            If provided, write the HTML to this file path.

        Returns
        -------
        str
            The rendered HTML.
        """
        html = _render_html(self)

        if path:
            Path(path).write_text(html, encoding="utf-8")

        return html


# ---------------------------------------------------------------------------
# Scoring implementation
# ---------------------------------------------------------------------------


def _compute_score(report: AuditReport) -> int:
    score = 100
    p = _Penalty

    # Missing values
    if report.missing_info:
        mi = report.missing_info
        for col in mi.columns_above_threshold:
            pct = mi.missing_pct_by_column[col]
            if pct > 0.50:
                score -= p.MISSING_HIGH
            elif pct > 0.20:
                score -= p.MISSING_MED
            else:
                score -= p.MISSING_LOW

    # Duplicates
    if report.duplicate_info:
        dp = report.duplicate_info.duplicate_pct
        if dp > 0.05:
            score -= p.DUP_SEVERE
        elif dp > 0.01:
            score -= p.DUP_MOD
        elif dp > 0:
            score -= p.DUP_MINOR

    # Schema issues
    if report.schema_report:
        schema_penalty = min(report.schema_report.issue_count * p.SCHEMA_EACH, p.SCHEMA_CAP)
        score -= schema_penalty

    # Validation violations
    if report.validation_result:
        violation_penalty = min(
            report.validation_result.violation_count * p.VIOLATION_EACH,
            p.VIOLATION_CAP,
        )
        score -= violation_penalty

    # Target issues
    if report.target_info:
        ti = report.target_info
        if ti.missing_pct > 0.10:
            score -= p.TARGET_MISSING_HIGH
        elif ti.missing_pct > 0.02:
            score -= p.TARGET_MISSING_LOW
        if ti.imbalance_ratio and ti.imbalance_ratio > 5:
            score -= p.TARGET_IMBALANCE

    # Outliers (mild)
    outlier_penalty = min(len(report.outlier_infos) * p.OUTLIER_EACH, p.OUTLIER_CAP)
    score -= outlier_penalty

    # Leakage hints (high-severity only)
    high_hints = [h for h in report.leakage_hints if h.severity == "high"]
    leakage_penalty = min(len(high_hints) * p.LEAKAGE_HIGH, p.LEAKAGE_CAP)
    score -= leakage_penalty

    return max(0, score)


def _collect_warnings(report: AuditReport) -> List[str]:
    warnings: List[str] = []

    if report.missing_info:
        mi = report.missing_info
        for col in mi.columns_above_threshold:
            pct = pct_str(mi.missing_pct_by_column[col])
            warnings.append(f"Column '{col}' is {pct} missing.")

    if report.duplicate_info:
        di = report.duplicate_info
        if di.full_row_duplicates:
            warnings.append(
                f"{di.full_row_duplicates} full-row duplicate(s) detected "
                f"({pct_str(di.duplicate_pct)})."
            )
        for col, cnt in di.id_column_duplicates.items():
            if cnt:
                warnings.append(f"ID column '{col}' has {cnt} duplicate value(s).")

    if report.schema_report:
        for issue in report.schema_report.issues:
            warnings.append(issue.description)

    if report.validation_result and not report.validation_result.is_valid:
        for v in report.validation_result.violations:
            warnings.append(v.description)

    if report.target_info:
        ti = report.target_info
        if ti.missing_pct > 0:
            warnings.append(
                f"Target column '{ti.column}' has {pct_str(ti.missing_pct)} missing labels."
            )
        if ti.imbalance_ratio and ti.imbalance_ratio > 5:
            warnings.append(
                f"Target column '{ti.column}' is imbalanced "
                f"(majority/minority ratio: {ti.imbalance_ratio:.1f}x)."
            )
        if ti.cardinality_warning:
            warnings.append(
                f"Target column '{ti.column}' has very high cardinality "
                f"({ti.unique_count} unique values). Is this a regression target?"
            )

    for hint in report.leakage_hints:
        warnings.append(f"[Leakage hint] {hint.description}")

    for o in report.outlier_infos:
        warnings.append(
            f"Column '{o.column}' has {o.outlier_count} outlier(s) "
            f"({pct_str(o.outlier_pct)}) outside [{o.lower_bound:.4g}, {o.upper_bound:.4g}]."
        )

    return warnings


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_STYLES = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }
  h2 { color: #374151; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; margin-top: 32px; }
  .score-badge { display: inline-block; padding: 6px 16px; border-radius: 20px;
                 font-size: 1.4rem; font-weight: 700; color: white; }
  .score-high   { background: #10b981; }
  .score-medium { background: #f59e0b; }
  .score-low    { background: #ef4444; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9rem; }
  th { background: #f3f4f6; text-align: left; padding: 8px 12px; border: 1px solid #d1d5db; }
  td { padding: 7px 12px; border: 1px solid #e5e7eb; vertical-align: top; }
  tr:nth-child(even) td { background: #f9fafb; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }
  .warn { color: #b45309; }
  .ok   { color: #059669; }
  .meta { color: #6b7280; font-size: 0.85rem; margin-top: 32px; }
  .summary-box { background: #eff6ff; border-left: 4px solid #3b82f6;
                 padding: 14px 18px; border-radius: 4px; margin: 16px 0; }
</style>
"""


def _score_class(score: int) -> str:
    if score >= 80:
        return "score-high"
    if score >= 60:
        return "score-medium"
    return "score-low"


def _e(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(str(text))


def _render_html(report: AuditReport) -> str:
    rows, cols = report.shape
    score = report.score

    parts: List[str] = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>datatrust Audit Report</title>",
        _HTML_STYLES,
        "</head><body>",
        "<h1>datatrust Audit Report</h1>",
        f"<p>Trust Score: <span class='score-badge {_score_class(score)}'>{score}/100</span></p>",
        f"<p><strong>{rows:,}</strong> rows &times; <strong>{cols}</strong> columns</p>",
        f"<div class='summary-box'>{_e(report.summary)}</div>",
    ]

    # Warnings
    warnings = report.warnings
    if warnings:
        parts.append("<h2>Warnings</h2><ul>")
        for w in warnings:
            parts.append(f"<li class='warn'>{_e(w)}</li>")
        parts.append("</ul>")

    # Schema issues
    if report.schema_report and report.schema_report.issues:
        parts += [
            "<h2>Schema Issues</h2>",
            "<table><tr><th>Column</th><th>Issue Type</th><th>Description</th><th>Suggestion</th></tr>",
        ]
        for i in report.schema_report.issues:
            parts.append(
                f"<tr><td><code>{_e(i.column)}</code></td><td>{_e(i.issue_type)}</td>"
                f"<td>{_e(i.description)}</td><td><code>{_e(i.suggestion)}</code></td></tr>"
            )
        parts.append("</table>")

    # Missing values
    if report.missing_info:
        mi = report.missing_info
        parts += [
            "<h2>Missing Values</h2>",
            f"<p>Total missing cells: <strong>{mi.total_missing:,}</strong> "
            f"({pct_str(mi.overall_missing_pct)} of all cells). "
            f"Threshold: {pct_str(mi.threshold)}.</p>",
        ]
        if mi.columns_above_threshold:
            parts += [
                "<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>",
            ]
            for col in mi.columns_above_threshold:
                parts.append(
                    f"<tr><td><code>{_e(col)}</code></td>"
                    f"<td>{mi.missing_by_column[col]:,}</td>"
                    f"<td>{pct_str(mi.missing_pct_by_column[col])}</td></tr>"
                )
            parts.append("</table>")

    # Duplicates
    if report.duplicate_info:
        di = report.duplicate_info
        parts += [
            "<h2>Duplicates</h2>",
            f"<p>Full-row duplicates: <strong>{di.full_row_duplicates:,}</strong> "
            f"({pct_str(di.duplicate_pct)}).</p>",
        ]
        if di.id_column_duplicates:
            parts += [
                "<table><tr><th>ID Column</th><th>Duplicate Rows</th></tr>",
            ]
            for col, cnt in di.id_column_duplicates.items():
                parts.append(
                    f"<tr><td><code>{_e(col)}</code></td><td>{cnt:,}</td></tr>"
                )
            parts.append("</table>")

    # Outliers
    if report.outlier_infos:
        parts += [
            "<h2>Outliers (IQR Method)</h2>",
            "<table><tr><th>Column</th><th>Count</th><th>%</th>"
            "<th>Lower Bound</th><th>Upper Bound</th></tr>",
        ]
        for o in report.outlier_infos:
            parts.append(
                f"<tr><td><code>{_e(o.column)}</code></td>"
                f"<td>{o.outlier_count:,}</td>"
                f"<td>{pct_str(o.outlier_pct)}</td>"
                f"<td>{o.lower_bound:.4g}</td>"
                f"<td>{o.upper_bound:.4g}</td></tr>"
            )
        parts.append("</table>")

    # Target
    if report.target_info:
        ti = report.target_info
        parts += [
            "<h2>Target Analysis</h2>",
            "<table>",
            f"<tr><th>Column</th><td><code>{_e(ti.column)}</code></td></tr>",
            f"<tr><th>Missing labels</th><td>{ti.missing_count:,} ({pct_str(ti.missing_pct)})</td></tr>",
            f"<tr><th>Unique values</th><td>{ti.unique_count}</td></tr>",
            f"<tr><th>Likely classification</th><td>{'Yes' if ti.is_likely_classification else 'No'}</td></tr>",
        ]
        if ti.imbalance_ratio is not None:
            parts.append(
                f"<tr><th>Imbalance ratio</th><td>{ti.imbalance_ratio:.2f}x</td></tr>"
            )
        parts.append("</table>")
        if ti.class_counts:
            parts += [
                "<h3>Class Distribution</h3>",
                "<table><tr><th>Class</th><th>Count</th></tr>",
            ]
            for cls, cnt in list(ti.class_counts.items())[:30]:
                parts.append(f"<tr><td>{_e(cls)}</td><td>{cnt:,}</td></tr>")
            parts.append("</table>")

    # Leakage hints
    if report.leakage_hints:
        parts += [
            "<h2>Leakage Hints <small style='font-weight:normal;color:#6b7280'>"
            "(Heuristic — review manually)</small></h2>",
            "<table><tr><th>Column</th><th>Type</th><th>Severity</th><th>Description</th></tr>",
        ]
        for h in report.leakage_hints:
            sev_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#6b7280"}.get(h.severity, "")
            parts.append(
                f"<tr><td><code>{_e(h.column)}</code></td>"
                f"<td>{_e(h.hint_type)}</td>"
                f"<td style='color:{sev_color}'>{_e(h.severity)}</td>"
                f"<td>{_e(h.description)}</td></tr>"
            )
        parts.append("</table>")

    # Validation
    if report.validation_result and not report.validation_result.is_valid:
        parts += [
            "<h2>Validation Violations</h2>",
            "<table><tr><th>Rule</th><th>Column</th><th>Affected Rows</th><th>Description</th></tr>",
        ]
        for v in report.validation_result.violations:
            parts.append(
                f"<tr><td>{_e(v.rule)}</td>"
                f"<td><code>{_e(v.column)}</code></td>"
                f"<td>{v.affected_rows:,}</td>"
                f"<td>{_e(v.description)}</td></tr>"
            )
        parts.append("</table>")

    parts.append(
        "<p class='meta'>Generated by <strong>datatrust</strong></p></body></html>"
    )

    return "\n".join(parts)
