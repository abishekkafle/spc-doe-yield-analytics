"""Unit tests verifying SPC, Capability, RSM DOE, Anomaly Detection, and 8D Reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly import detect_multivariate_anomalies
from src.data_generator import generate_doe_ccd_data, generate_production_data
from src.doe import fit_response_surface, optimize_response_surface
from src.report_8d import generate_8d_report
from src.spc import compute_capability, compute_imr_chart, compute_xbar_r_chart


def test_imr_and_capability_in_control():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=100.0, scale=2.0, size=200)
    imr = compute_imr_chart(data)
    cap = compute_capability(data, lsl=90.0, usl=110.0, target=100.0, sigma_within=imr.sigma_within)

    # Expected Cp ~ (110-90)/(6*2) = 1.67
    assert 1.35 < cap.cp < 2.05
    assert abs(cap.cp - cap.cpk) < 0.25
    assert cap.observed_yield_pct > 99.0
    assert cap.is_normal is True


def test_nelson_rules_detect_shift_and_outlier():
    # Stable baseline + obvious +6 sigma spike + 12 points shifted above mean
    base = np.full(40, 50.0) + np.sin(np.arange(40)) * 0.2
    base[10] = 65.0  # Rule 1 spike
    base[20:35] = 52.5  # Rule 2 sustained shift
    imr = compute_imr_chart(base)
    rule_names = [v.rule for v in imr.violations]
    assert any("Rule 1" in r for r in rule_names)
    assert any("Rule 2" in r for r in rule_names)
    assert 10 in imr.ooc_indices


def test_xbar_r_chart_subgroups():
    df, spec = generate_production_data("cold_spray", n_runs=100, inject_anomalies=False)
    res = compute_xbar_r_chart(df[spec.primary_ctq], subgroup_size=5)
    assert len(res.x_bars) == 20
    assert res.ucl_x > res.x_double_bar > res.lcl_x
    assert res.ucl_r > res.r_bar >= res.lcl_r


def test_rsm_doe_fit_and_optimization():
    doe_df, spec = generate_doe_ccd_data("cold_spray", seed=42)
    factors = spec.process_factors[:3]
    rsm = fit_response_surface(doe_df, factors=factors, response=spec.primary_ctq)

    assert rsm.r_squared > 0.90
    assert rsm.pred_r_squared > 0.70
    assert len(rsm.summary_table) == 10  # 1 + 3 + 3 + 3

    opt = optimize_response_surface(rsm, goal="target", target=spec.target, lsl=spec.lsl, usl=spec.usl)
    assert opt.success
    assert abs(opt.predicted_response - spec.target) < 5.0
    assert opt.desirability > 0.85


def test_multivariate_anomaly_detection_recovers_injected_excursions():
    df, spec = generate_production_data("cold_spray", n_runs=180, seed=42, inject_anomalies=True)
    res = detect_multivariate_anomalies(df, features=spec.process_factors, alpha=0.01, contamination=0.08)

    flagged_runs = set(res.excursion_table["Run_ID"].tolist())
    # Injected heater spike at runs 162-164 and pressure drop at 95-102 must be caught
    assert any(r in flagged_runs for r in (95, 96, 97, 98, 99, 100))
    assert any(r in flagged_runs for r in (162, 163, 164))
    assert sum(res.driver_summary.values()) == len(res.excursion_table)


def test_8d_report_end_to_end():
    df, spec = generate_production_data("semiconductor_etch", n_runs=180, seed=7, inject_anomalies=True)
    doe_df, _ = generate_doe_ccd_data("semiconductor_etch", seed=7)

    imr = compute_imr_chart(df[spec.primary_ctq])
    cap = compute_capability(df[spec.primary_ctq], spec.lsl, spec.usl, spec.target, imr.sigma_within)
    anom = detect_multivariate_anomalies(df, spec.process_factors)
    rsm = fit_response_surface(doe_df, spec.process_factors[:3], spec.primary_ctq)
    opt = optimize_response_surface(rsm, goal="target", target=spec.target, lsl=spec.lsl, usl=spec.usl)

    report = generate_8d_report(spec.name, spec.primary_ctq, spec.ctq_unit, imr, cap, anom, rsm, opt)
    for section in ("## D1", "## D2", "## D3", "## D4", "## D5", "## D6", "## D7", "## D8"):
        assert section in report
    assert spec.primary_ctq in report
