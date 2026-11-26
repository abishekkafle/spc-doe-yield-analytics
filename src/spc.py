"""Statistical Process Control (SPC) and Process Capability analytics.

Implements AIAG / Montgomery standard calculations:
- Individuals & Moving Range (I-MR) control charts
- X-bar & R subgroup control charts (subgroups n=2..10)
- Nelson Rules 1-4 for out-of-control signal detection
- Short-term capability (Cp, Cpk) using within-subgroup sigma
- Long-term performance (Pp, Ppk) using overall sample sigma
- Yield %, DPMO, Sigma Level (with 1.5-sigma shift convention), and Shapiro-Wilk normality check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Standard AIAG / Montgomery control chart constants for subgroup sizes n = 2..10
CONTROL_CONSTANTS: Dict[int, Dict[str, float]] = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267, "d2": 1.128},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574, "d2": 1.693},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282, "d2": 2.059},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114, "d2": 2.326},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004, "d2": 2.534},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078},
}


@dataclass
class NelsonViolation:
    rule: str
    description: str
    indices: List[int]


@dataclass
class IMRResult:
    values: np.ndarray
    moving_ranges: np.ndarray
    x_bar: float
    mr_bar: float
    sigma_within: float
    ucl_i: float
    lcl_i: float
    ucl_mr: float
    lcl_mr: float
    violations: List[NelsonViolation]
    ooc_indices: List[int]


@dataclass
class XbarRResult:
    subgroup_labels: List[str]
    x_bars: np.ndarray
    ranges: np.ndarray
    subgroup_size: int
    x_double_bar: float
    r_bar: float
    sigma_within: float
    ucl_x: float
    lcl_x: float
    ucl_r: float
    lcl_r: float
    ooc_x_indices: List[int]
    ooc_r_indices: List[int]


@dataclass
class CapabilityResult:
    n: int
    mean: float
    std_overall: float
    std_within: float
    lsl: float
    usl: float
    target: float
    cp: float
    cpk: float
    cpu: float
    cpl: float
    pp: float
    ppk: float
    observed_yield_pct: float
    expected_yield_pct: float
    dpmo: float
    sigma_level: float
    normality_p_value: float
    is_normal: bool


def detect_nelson_rules(values: np.ndarray, mean: float, sigma: float) -> List[NelsonViolation]:
    """Detect Nelson Rules 1-4 on an ordered process series."""
    n = len(values)
    violations: List[NelsonViolation] = []
    if sigma <= 0 or n == 0:
        return violations

    z = (values - mean) / sigma

    # Rule 1: Any single point beyond ±3 sigma
    r1 = np.where(np.abs(z) > 3.0)[0].tolist()
    if r1:
        violations.append(
            NelsonViolation(
                rule="Rule 1 (Beyond 3σ)",
                description="Single point outside ±3σ control limits (special-cause excursion).",
                indices=r1,
            )
        )

    # Rule 2: 9 or more consecutive points on the same side of the centerline
    r2: List[int] = []
    run_len = 0
    prev_sign = 0
    for i, val in enumerate(z):
        sign = 1 if val > 0 else (-1 if val < 0 else 0)
        if sign != 0 and sign == prev_sign:
            run_len += 1
        else:
            run_len = 1 if sign != 0 else 0
            prev_sign = sign
        if run_len >= 9:
            r2.append(i)
    if r2:
        violations.append(
            NelsonViolation(
                rule="Rule 2 (9 Same Side)",
                description="9+ consecutive points on one side of centerline (sustained mean shift).",
                indices=r2,
            )
        )

    # Rule 3: 6 or more consecutive points steadily increasing or decreasing
    r3: List[int] = []
    inc_len = 1
    dec_len = 1
    for i in range(1, n):
        if values[i] > values[i - 1]:
            inc_len += 1
            dec_len = 1
        elif values[i] < values[i - 1]:
            dec_len += 1
            inc_len = 1
        else:
            inc_len = 1
            dec_len = 1
        if inc_len >= 6 or dec_len >= 6:
            r3.append(i)
    if r3:
        violations.append(
            NelsonViolation(
                rule="Rule 3 (6 Trending)",
                description="6+ consecutive points strictly increasing or decreasing (tool wear / drift).",
                indices=r3,
            )
        )

    # Rule 4: 14 or more consecutive points alternating up and down
    r4: List[int] = []
    alt_len = 1
    prev_dir = 0
    for i in range(1, n):
        curr_dir = 1 if values[i] > values[i - 1] else (-1 if values[i] < values[i - 1] else 0)
        if curr_dir != 0 and curr_dir == -prev_dir:
            alt_len += 1
        else:
            alt_len = 2 if curr_dir != 0 else 1
        prev_dir = curr_dir
        if alt_len >= 14:
            r4.append(i)
    if r4:
        violations.append(
            NelsonViolation(
                rule="Rule 4 (14 Alternating)",
                description="14+ points alternating up/down (over-adjustment or dual-stream mixing).",
                indices=r4,
            )
        )

    return violations


def compute_imr_chart(series: pd.Series | np.ndarray) -> IMRResult:
    """Compute Individuals and Moving Range (I-MR) control chart statistics."""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        raise ValueError("I-MR chart requires at least 3 non-null observations.")

    d2 = CONTROL_CONSTANTS[2]["d2"]  # 1.128
    d4 = CONTROL_CONSTANTS[2]["D4"]  # 3.267

    mr = np.abs(np.diff(values))
    x_bar = float(np.mean(values))
    mr_bar = float(np.mean(mr))
    sigma_within = mr_bar / d2

    ucl_i = x_bar + 3.0 * sigma_within
    lcl_i = x_bar - 3.0 * sigma_within
    ucl_mr = d4 * mr_bar
    lcl_mr = 0.0

    violations = detect_nelson_rules(values, x_bar, sigma_within)
    ooc_set = sorted({idx for v in violations for idx in v.indices})

    return IMRResult(
        values=values,
        moving_ranges=mr,
        x_bar=x_bar,
        mr_bar=mr_bar,
        sigma_within=sigma_within,
        ucl_i=ucl_i,
        lcl_i=lcl_i,
        ucl_mr=ucl_mr,
        lcl_mr=lcl_mr,
        violations=violations,
        ooc_indices=ooc_set,
    )


def compute_xbar_r_chart(
    series: pd.Series | np.ndarray,
    subgroup_size: int = 5,
) -> XbarRResult:
    """Compute X-bar and R subgroup control charts."""
    if subgroup_size not in CONTROL_CONSTANTS:
        raise ValueError(f"subgroup_size must be between 2 and 10; got {subgroup_size}.")

    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    n_groups = len(values) // subgroup_size
    if n_groups < 2:
        raise ValueError("Need at least 2 complete subgroups for an X-bar/R chart.")

    trimmed = values[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
    x_bars = trimmed.mean(axis=1)
    ranges = trimmed.max(axis=1) - trimmed.min(axis=1)

    c = CONTROL_CONSTANTS[subgroup_size]
    x_double_bar = float(np.mean(x_bars))
    r_bar = float(np.mean(ranges))
    sigma_within = r_bar / c["d2"]

    ucl_x = x_double_bar + c["A2"] * r_bar
    lcl_x = x_double_bar - c["A2"] * r_bar
    ucl_r = c["D4"] * r_bar
    lcl_r = c["D3"] * r_bar

    ooc_x = np.where((x_bars > ucl_x) | (x_bars < lcl_x))[0].tolist()
    ooc_r = np.where((ranges > ucl_r) | (ranges < lcl_r))[0].tolist()
    labels = [f"SG-{i + 1}" for i in range(n_groups)]

    return XbarRResult(
        subgroup_labels=labels,
        x_bars=x_bars,
        ranges=ranges,
        subgroup_size=subgroup_size,
        x_double_bar=x_double_bar,
        r_bar=r_bar,
        sigma_within=sigma_within,
        ucl_x=ucl_x,
        lcl_x=lcl_x,
        ucl_r=ucl_r,
        lcl_r=lcl_r,
        ooc_x_indices=ooc_x,
        ooc_r_indices=ooc_r,
    )


def compute_capability(
    series: pd.Series | np.ndarray,
    lsl: float,
    usl: float,
    target: Optional[float] = None,
    sigma_within: Optional[float] = None,
) -> CapabilityResult:
    """Compute Cp, Cpk, Pp, Ppk, Yield, DPMO, Sigma Level, and normality test."""
    if usl <= lsl:
        raise ValueError("USL must be strictly greater than LSL.")

    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 3:
        raise ValueError("Capability analysis requires at least 3 observations.")

    if target is None:
        target = 0.5 * (lsl + usl)

    mean = float(np.mean(values))
    std_overall = float(np.std(values, ddof=1))

    if sigma_within is None:
        mr_bar = float(np.mean(np.abs(np.diff(values))))
        sigma_within = mr_bar / CONTROL_CONSTANTS[2]["d2"]

    # Guard against zero variance
    eps = 1e-12
    sw = max(sigma_within, eps)
    so = max(std_overall, eps)

    cp = (usl - lsl) / (6.0 * sw)
    cpu = (usl - mean) / (3.0 * sw)
    cpl = (mean - lsl) / (3.0 * sw)
    cpk = min(cpu, cpl)

    pp = (usl - lsl) / (6.0 * so)
    ppk = min((usl - mean) / (3.0 * so), (mean - lsl) / (3.0 * so))

    # Observed vs expected yield (normal assumption using overall std)
    in_spec = np.sum((values >= lsl) & (values <= usl))
    observed_yield_pct = 100.0 * float(in_spec) / n

    p_below = float(stats.norm.cdf(lsl, loc=mean, scale=so))
    p_above = float(1.0 - stats.norm.cdf(usl, loc=mean, scale=so))
    p_defect = min(max(p_below + p_above, 1e-12), 1.0 - 1e-12)
    expected_yield_pct = 100.0 * (1.0 - p_defect)
    dpmo = p_defect * 1_000_000.0

    # Conventional long-term sigma level (Z_lt + 1.5 shift)
    z_lt = float(stats.norm.isf(p_defect))
    sigma_level = max(0.0, z_lt + 1.5)

    # Shapiro-Wilk normality check (cap at 5000 samples per scipy docs)
    sample_for_test = values[:5000]
    _, norm_p = stats.shapiro(sample_for_test)

    return CapabilityResult(
        n=n,
        mean=mean,
        std_overall=std_overall,
        std_within=sigma_within,
        lsl=lsl,
        usl=usl,
        target=target,
        cp=cp,
        cpk=cpk,
        cpu=cpu,
        cpl=cpl,
        pp=pp,
        ppk=ppk,
        observed_yield_pct=observed_yield_pct,
        expected_yield_pct=expected_yield_pct,
        dpmo=dpmo,
        sigma_level=sigma_level,
        normality_p_value=float(norm_p),
        is_normal=bool(norm_p >= 0.05),
    )
