"""Automated deterministic 8D Root-Cause & CAPA Report Generator.

Synthesizes live SPC, Capability (Cp/Cpk/Pp/Ppk), Multivariate Anomaly Attribution,
and Response Surface DOE findings into a structured D1-D8 engineering report
exportable to Markdown (zero external API keys required).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.anomaly import AnomalyResult
from src.doe import OptimizationResult, RSMResult
from src.spc import CapabilityResult, IMRResult


def _capability_status(cpk: float) -> str:
    if cpk >= 1.67:
        return "EXCELLENT (Six Sigma capable)"
    if cpk >= 1.33:
        return "CAPABLE (Meets standard industrial target ≥ 1.33)"
    if cpk >= 1.00:
        return "MARGINAL (Requires tighter centering or variance reduction)"
    return "INCAPABLE (High scrap/rework risk; immediate containment required)"


def generate_8d_report(
    process_name: str,
    ctq_name: str,
    ctq_unit: str,
    imr: IMRResult,
    cap: CapabilityResult,
    anomaly: Optional[AnomalyResult] = None,
    rsm: Optional[RSMResult] = None,
    opt: Optional[OptimizationResult] = None,
) -> str:
    """Generate a structured 8D (Eight Disciplines) engineering summary in Markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = _capability_status(cap.cpk)

    # Identify centering vs spread issue
    cp_gap = cap.cp - cap.cpk
    shift_gap = cap.cpk - cap.ppk
    centering_note = (
        f"Significant off-centering detected (Cp = {cap.cp:.2f} vs Cpk = {cap.cpk:.2f}; "
        f"mean {cap.mean:.2f} vs target {cap.target:.2f} {ctq_unit}). Re-centering alone can recover capability."
        if cp_gap > 0.15
        else f"Process mean ({cap.mean:.2f} {ctq_unit}) is reasonably centered near target ({cap.target:.2f} {ctq_unit})."
    )
    stability_note = (
        f"Notable gap between short-term Cpk ({cap.cpk:.2f}) and long-term Ppk ({cap.ppk:.2f}) "
        "confirms time-dependent drift or special-cause shifts between runs."
        if shift_gap > 0.15
        else f"Short-term Cpk ({cap.cpk:.2f}) and long-term Ppk ({cap.ppk:.2f}) are closely aligned."
    )

    # Nelson rule summary
    if imr.violations:
        nelson_lines = "\n".join(
            f"- **{v.rule}**: {len(v.indices)} signal(s) — {v.description} (first at run index {v.indices[0] + 1})"
            for v in imr.violations
        )
    else:
        nelson_lines = "- No Nelson Rule 1–4 violations detected on the I-MR chart."

    # Anomaly attribution summary
    if anomaly is not None and not anomaly.excursion_table.empty:
        n_anom = int(anomaly.combined_flags.sum())
        sorted_drivers = sorted(anomaly.driver_summary.items(), key=lambda kv: kv[1], reverse=True)
        top_drivers = ", ".join(f"**{k}** ({v} runs)" for k, v in sorted_drivers if v > 0)
        anom_section = (
            f"- **Flagged Excursions:** {n_anom} runs triggered Hotelling's T² (UCL = {anomaly.t2_ucl:.2f}) "
            f"and/or Isolation Forest.\n"
            f"- **Top Sensor Drivers (Ranked by Frequency):** {top_drivers}."
        )
    else:
        anom_section = "- No multivariate sensor excursions flagged beyond threshold."

    # DOE / RSM sensitivity & setpoint recommendation
    if rsm is not None:
        sig_terms = rsm.summary_table[
            (rsm.summary_table["Term"] != "Intercept")
            & (rsm.summary_table["Significant (p<0.05)"])
        ].sort_values(by="p_Value")
        if not sig_terms.empty:
            doe_terms_str = "\n".join(
                f"- `{row.Term}`: coded effect = `{row.Coded_Coeff:+.3f}` (p = {row.p_Value:.4f})"
                for row in sig_terms.itertuples()
            )
        else:
            doe_terms_str = "- No individual quadratic/linear terms reached p < 0.05."
        rsm_fit_line = (
            f"Response Surface Model fit: **R² = {rsm.r_squared:.3f}**, "
            f"**Adj-R² = {rsm.adj_r_squared:.3f}**, **Pred-R² (PRESS) = {rsm.pred_r_squared:.3f}**."
        )
    else:
        doe_terms_str = "- Response Surface DOE not included in this run."
        rsm_fit_line = ""

    if opt is not None:
        setpoints = ", ".join(f"`{k} = {v}`" for k, v in opt.optimal_factors.items())
        opt_line = (
            f"- **Recommended Optimal Setpoints ({opt.goal.title()}):** {setpoints}\n"
            f"- **Predicted {ctq_name}:** `{opt.predicted_response} {ctq_unit}` "
            f"(Desirability = `{opt.desirability:.3f}`)"
        )
    else:
        opt_line = "- Run RSM Optimization to generate verified setpoint recommendations."

    normality_warning = (
        ""
        if cap.is_normal
        else (
            f"\n> **Statistical Caution:** Shapiro-Wilk normality test rejected normality "
            f"(p = {cap.normality_p_value:.4f} < 0.05). Standard normal-theory Cp/Cpk and DPMO "
            "may be biased; inspect for bimodal shifts or use non-parametric percentiles."
        )
    )

    return f"""# 8D Root-Cause & Process Capability Report

**Process:** {process_name}  
**Primary CTQ:** `{ctq_name}` ({ctq_unit}) | **Spec Limits:** LSL = {cap.lsl}, Target = {cap.target}, USL = {cap.usl}  
**Generated:** {date_str}  
**Overall Capability Status:** **{status}**

---

## D1 — Team & Scope
- **Scope:** Statistical Process Control (I-MR), Process Capability (n = {cap.n} runs), Multivariate Sensor Diagnostics, and Response Surface DOE Optimization for `{ctq_name}`.
- **Recommended Cross-Functional Team:** Process Engineer (Lead), Quality/Metrology Engineer, Equipment Maintenance Technician, Production Supervisor.

## D2 — Problem Description (Quantified)
| Metric | Value | Benchmark |
|---|---|---|
| Process Mean ± Overall Std | {cap.mean:.2f} ± {cap.std_overall:.2f} {ctq_unit} | Target {cap.target:.2f} |
| Short-Term Capability (Cp / Cpk) | **{cap.cp:.2f} / {cap.cpk:.2f}** | ≥ 1.33 |
| Long-Term Performance (Pp / Ppk) | **{cap.pp:.2f} / {cap.ppk:.2f}** | ≥ 1.33 |
| Observed Yield / Expected Yield | {cap.observed_yield_pct:.2f}% / {cap.expected_yield_pct:.2f}% | ≥ 99.73% |
| Estimated DPMO / Sigma Level | {cap.dpmo:,.0f} DPMO / {cap.sigma_level:.2f}σ | ≤ 63 DPMO (4.5σ+) |
| Out-of-Control Signals (I-MR) | {len(imr.ooc_indices)} run(s) flagged | 0 |
{normality_warning}

**Diagnostic Interpretation:**
- {centering_note}
- {stability_note}

## D3 — Interim Containment Actions (ICA)
1. **Quarantine & 100% Inspection:** Hold parts from all {len(imr.ooc_indices)} out-of-control run(s) and multivariate excursion lots for dimensional/metrology disposition before shipment.
2. **Tighten Alarm Limits:** Temporarily enforce ±2.5σ warning limits on `{ctq_name}` ({imr.x_bar - 2.5 * imr.sigma_within:.2f} to {imr.x_bar + 2.5 * imr.sigma_within:.2f} {ctq_unit}) until drift is stabilized.

## D4 — Root-Cause Analysis (Statistical Evidence)
### Control Chart Signals (Nelson Rules)
{nelson_lines}

### Multivariate Sensor Attribution
{anom_section}

### Statistically Significant Process Factors (RSM DOE)
{rsm_fit_line}
{doe_terms_str}

## D5 — Permanent Corrective Actions (PCA)
{opt_line}
- **Hardware / Maintenance Action:** Inspect and calibrate the top sensor driver(s) identified in D4 (e.g., preventive replacement schedule for wear components and closed-loop sensor interlocks).
- **SOP Update:** Lock revised parameter windows and pre-run verification checks into the shop-floor work instruction.

## D6 — Implementation & Validation Plan
1. Execute a **30-run confirmation build** at the D5 optimal setpoints.
2. Verify **zero Nelson Rule 1–4 violations** on the I-MR chart and confirm **Cpk ≥ 1.33** and **Ppk ≥ 1.33**.
3. Perform paired metrology validation against reference standards before closing containment.

## D7 — Prevent Recurrence (Control Plan)
- Deploy real-time **Hotelling's T²** monitoring across process sensors with automated interlock alerts prior to CTQ out-of-spec excursion.
- Schedule periodic preventive maintenance aligned with the observed drift onset interval.

## D8 — Closure & Lessons Learned
- Archive the validated RSM model, control limits, and 8D record in the digital manufacturing log for cross-line transferability.
"""
