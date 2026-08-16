"""Report generation for trial simulation results.

Produces a human-readable Markdown report and an HTML rendering of a
``TrialResult`` (or a combined trial + validation report).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insilico_trial.schemas import TrialResult


def _fmt(v: Any, nd: int = 3) -> str:
    """Format a value for display, handling None."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _metric_row(metric: str, value: dict[str, Any] | None) -> list[str]:
    if not value:
        return [metric, "N/A", "N/A", "N/A"]
    return [
        metric,
        _fmt(value.get("median")),
        _fmt(value.get("p5")),
        _fmt(value.get("p95")),
    ]


def render_trial_markdown(result: TrialResult) -> str:
    """Render a TrialResult to Markdown."""
    lines: list[str] = []
    lines.append(f"# VeriTrial Simulation Report: {result.protocol_name}")
    lines.append("")
    lines.append(f"**Run ID:** `{result.run_id}`")
    lines.append(f"**Generated:** {result.timestamp_utc.isoformat()}")
    lines.append(f"**Drug:** {result.drug_name}  |  **Population:** {result.population_name}")
    lines.append(f"**Subjects:** {result.n_subjects}  |  **Cohorts:** {result.n_cohorts}")
    lines.append("")

    lines.append("## 1. Cohort Summary")
    lines.append("")
    lines.append("| Cohort | Dose (mg) | N | DLTs | DLT rate | Escalation |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for c in result.cohort_summaries:
        lines.append(
            f"| {c['cohort']} | {c['dose_mg']:g} | {c['n']} | {c['n_dlt']} "
            f"| {c['dlt_rate']:.0%} | {c['escalation_decision']} |"
        )
    lines.append("")

    lines.append("## 2. PK Summary (by cohort)")
    lines.append("")
    lines.append("| Cohort | N | Cmax mean | AUC mean | t1/2 mean | CL/F mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for s in result.pk_summaries:
        lines.append(
            f"| {s.cohort_label} | {s.n} | {_fmt(s.cmax_mean)} | {_fmt(s.auc_mean)} "
            f"| {_fmt(s.half_life_mean)} | {_fmt(s.cl_f_mean)} |"
        )
    lines.append("")

    lines.append("## 3. Population Summary")
    lines.append("")
    ps = result.population_summary
    if ps:
        lines.append(f"- Age: {ps.mean_age:.1f} +/- {ps.std_age:.1f} years")
        lines.append(f"- Sex: {ps.n_male} M / {ps.n_female} F")
        lines.append(f"- Weight: {ps.mean_weight:.1f} +/- {ps.std_weight:.1f} kg; BMI {ps.mean_bmi:.1f}")
        lines.append(f"- eGFR (median): {ps.median_egfr:.0f} mL/min")
        lines.append(f"- CL/F: {ps.mean_cl:.3f} +/- {ps.std_cl:.3f} L/h; V/F: {ps.mean_v:.1f} +/- {ps.std_v:.1f} L")
    lines.append("")

    lines.append("## 4. Safety Summary")
    lines.append("")
    ss = result.safety_summary
    for key, value in ss.items():
        if isinstance(value, list):
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## 5. Uncertainty (median [p5 - p95])")
    lines.append("")
    for section, metrics in result.uncertainty.items():
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Metric | Median | p5 | p95 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric in ("cmax", "auc_inf", "half_life", "cl_f"):
            lines.append("| " + " | ".join(_metric_row(metric, metrics.get(metric))) + " |")
        lines.append("")

    lines.append("## 6. Provenance")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.provenance, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def render_trial_html(result: TrialResult) -> str:
    """Render a TrialResult to an HTML document."""
    def _esc(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    body_html = []
    body_html.append(f"<h1>{_esc(result.protocol_name)} — Simulation Report</h1>")
    body_html.append(f"<p><strong>Run ID:</strong> <code>{_esc(result.run_id)}</code> | "
                     f"<strong>Generated:</strong> {_esc(result.timestamp_utc.isoformat())}</p>")

    body_html.append("<h2>Cohort Summary</h2>")
    body_html.append("<table><tr><th>Cohort</th><th>Dose (mg)</th><th>N</th><th>DLTs</th>"
                     "<th>DLT rate</th><th>Escalation</th></tr>")
    for c in result.cohort_summaries:
        body_html.append(
            f"<tr><td>{c['cohort']}</td><td>{c['dose_mg']:g}</td><td>{c['n']}</td>"
            f"<td>{c['n_dlt']}</td><td>{c['dlt_rate']:.0%}</td><td>{_esc(c['escalation_decision'])}</td></tr>"
        )
    body_html.append("</table>")

    body_html.append("<h2>PK Summary (by cohort)</h2>")
    body_html.append("<table><tr><th>Cohort</th><th>N</th><th>Cmax mean</th><th>AUC mean</th>"
                     "<th>t1/2 mean</th><th>CL/F mean</th></tr>")
    for s in result.pk_summaries:
        body_html.append(
            f"<tr><td>{_esc(s.cohort_label)}</td><td>{s.n}</td><td>{_fmt(s.cmax_mean)}</td>"
            f"<td>{_fmt(s.auc_mean)}</td><td>{_fmt(s.half_life_mean)}</td><td>{_fmt(s.cl_f_mean)}</td></tr>"
        )
    body_html.append("</table>")

    ps = result.population_summary
    if ps:
        body_html.append("<h2>Population Summary</h2>")
        body_html.append(
            f"<p>Age {ps.mean_age:.1f} ± {ps.std_age:.1f} y; {ps.n_male} M / {ps.n_female} F; "
            f"weight {ps.mean_weight:.1f} ± {ps.std_weight:.1f} kg; "
            f"CL/F {ps.mean_cl:.3f} ± {ps.std_cl:.3f} L/h</p>"
        )

    body_html.append("<h2>Safety Summary</h2><ul>")
    for key, value in result.safety_summary.items():
        if not isinstance(value, list):
            body_html.append(f"<li>{_esc(str(key))}: {_esc(str(value))}</li>")
    body_html.append("</ul>")

    body_html.append("<h2>Uncertainty (median [p5-p95])</h2>")
    for section, metrics in result.uncertainty.items():
        body_html.append(f"<h3>{_esc(section)}</h3><table>"
                         "<tr><th>Metric</th><th>Median</th><th>p5</th><th>p95</th></tr>")
        for metric in ("cmax", "auc_inf", "half_life", "cl_f"):
            cells = _metric_row(metric, metrics.get(metric))
            body_html.append(f"<tr><td>{_esc(cells[0])}</td><td>{_esc(cells[1])}</td>"
                             f"<td>{_esc(cells[2])}</td><td>{_esc(cells[3])}</td></tr>")
        body_html.append("</table>")

    body_html.append(
        "<h2>Provenance</h2><pre>" + _esc(json.dumps(result.provenance, indent=2)) + "</pre>"
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"><title>{_esc(result.protocol_name)} — VeriTrial Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 40px;}}
h1 {{color: #2c3e50}} h2 {{color: #34495e; margin-top: 2em}}
table {{border-collapse: collapse; width: 100%;}}
th, td {{border: 1px solid #ccc; padding: 6px 10px; text-align: left}}
th {{background-color: #f2f2f2}}
code, pre {{background-color: #f6f6f6; padding: 2px 4px; border-radius: 3px}}
</style>
</head>
<body>
{''.join(body_html)}
</body>
</html>"""


def write_report(
    result: TrialResult,
    out_dir: str | Path = "output",
    run_label: str = "trial",
) -> dict[str, Path]:
    """Write Markdown + HTML reports and a machine-readable JSON of the result.

    Returns the paths written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / f"{run_label}_report.md"
    html_path = out / f"{run_label}_report.html"
    json_path = out / f"{run_label}_result.json"

    md_path.write_text(render_trial_markdown(result))
    html_path.write_text(render_trial_html(result))
    json_path.write_text(result.model_dump_json(indent=2))

    return {"markdown": md_path, "html": html_path, "json": json_path}


__all__ = ["render_trial_markdown", "render_trial_html", "write_report"]
