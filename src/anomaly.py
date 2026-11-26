"""Multivariate process anomaly detection and sensor-level root-cause attribution.

Combines two complementary detectors:
1. Hotelling's T^2 statistic (classical multivariate SPC with F-distribution UCL)
2. Isolation Forest (non-parametric detector for nonlinear excursions)

For every flagged run, computes standardized sensor Z-scores and identifies the
primary driving parameter(s) so engineers get actionable diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    features: List[str]
    t2_scores: np.ndarray
    t2_ucl: float
    t2_flags: np.ndarray
    iforest_scores: np.ndarray
    iforest_flags: np.ndarray
    combined_flags: np.ndarray
    excursion_table: pd.DataFrame
    driver_summary: Dict[str, int]


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    features: List[str],
    run_id_col: str = "Run_ID",
    alpha: float = 0.01,
    contamination: float = 0.06,
    baseline_fraction: float = 0.45,
) -> AnomalyResult:
    """Run Hotelling's T^2 and Isolation Forest on process sensor columns.

    Uses the first `baseline_fraction` of runs as the in-control Phase I reference
    baseline for Hotelling's T^2 mean/covariance estimation so late-stage drift
    does not mask itself.
    """
    if len(features) < 2:
        raise ValueError("Multivariate anomaly detection requires at least 2 features.")

    clean = df[features].dropna().copy()
    n, p = clean.shape
    if n < max(20, p + 5):
        raise ValueError(f"Need at least {max(20, p + 5)} complete rows for multivariate detection.")

    X = clean.to_numpy(dtype=float)

    # Phase I baseline for Hotelling's T^2
    m = max(p + 5, int(n * baseline_fraction))
    X_base = X[:m]
    mu = np.mean(X_base, axis=0)
    cov = np.cov(X_base, rowvar=False)
    # Small ridge regularization for numerical stability
    cov_reg = cov + np.eye(p) * 1e-6 * np.trace(cov) / p
    cov_inv = np.linalg.inv(cov_reg)

    diff = X - mu
    t2_scores = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)

    # Phase II Hotelling's T^2 UCL from F-distribution:
    # UCL = p * (m + 1) * (m - 1) / (m * (m - p)) * F_{alpha, p, m - p}
    f_crit = float(stats.f.ppf(1.0 - alpha, p, m - p))
    t2_ucl = (p * (m + 1) * (m - 1)) / (m * (m - p)) * f_crit
    t2_flags = t2_scores > t2_ucl

    # Isolation Forest on standardized features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=float(np.clip(contamination, 0.01, 0.25)),
        random_state=42,
    )
    preds = iso.fit_predict(X_scaled)
    iforest_flags = preds == -1
    # Invert decision_function so higher = more anomalous
    iforest_scores = -iso.decision_function(X_scaled)

    combined_flags = t2_flags | iforest_flags

    # Sensor-level Z-score attribution relative to baseline
    base_std = np.maximum(np.std(X_base, axis=0, ddof=1), 1e-9)
    z_mat = (X - mu) / base_std

    run_ids = (
        df.loc[clean.index, run_id_col].to_numpy()
        if run_id_col in df.columns
        else np.arange(1, n + 1)
    )

    records = []
    driver_counts: Dict[str, int] = {f: 0 for f in features}
    for i in np.where(combined_flags)[0]:
        abs_z = np.abs(z_mat[i])
        top_idx = int(np.argmax(abs_z))
        top_feat = features[top_idx]
        top_z = float(z_mat[i, top_idx])
        driver_counts[top_feat] += 1

        # Also note secondary driver if |Z| > 2.0
        order = np.argsort(abs_z)[::-1]
        drivers = [f"{features[j]} ({z_mat[i, j]:+.1f}σ)" for j in order[:2] if abs_z[j] >= 1.8]
        if not drivers:
            drivers = [f"{top_feat} ({top_z:+.1f}σ)"]

        detectors = []
        if t2_flags[i]:
            detectors.append("Hotelling T²")
        if iforest_flags[i]:
            detectors.append("Isolation Forest")

        records.append(
            {
                "Run_ID": run_ids[i],
                "Detectors": " + ".join(detectors),
                "Hotelling_T2": round(float(t2_scores[i]), 2),
                "Primary_Driver": top_feat,
                "Sensor_Attribution": ", ".join(drivers),
            }
        )

    excursion_df = pd.DataFrame(records)
    return AnomalyResult(
        features=features,
        t2_scores=t2_scores,
        t2_ucl=float(t2_ucl),
        t2_flags=t2_flags,
        iforest_scores=iforest_scores,
        iforest_flags=iforest_flags,
        combined_flags=combined_flags,
        excursion_table=excursion_df,
        driver_summary=driver_counts,
    )
