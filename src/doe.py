"""Response Surface Methodology (RSM) DOE fitting, ANOVA, and optimization.

Fits a 2nd-order coded polynomial model:
    y = b0 + sum(bi * xi) + sum(bii * xi^2) + sum(bij * xi * xj)
Computes:
- Term-level coefficients, t-stats, and p-values
- Model-level R^2, Adjusted R^2, and Predicted R^2 (via exact PRESS statistic)
- 2D contour / 3D surface evaluation grids
- Derringer-Suich numerical desirability optimization within factor bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


@dataclass
class RSMResult:
    factors: List[str]
    response: str
    factor_bounds: Dict[str, Tuple[float, float]]
    term_names: List[str]
    summary_table: pd.DataFrame
    r_squared: float
    adj_r_squared: float
    pred_r_squared: float
    rmse: float
    f_stat: float
    f_pvalue: float
    n_obs: int
    residuals: np.ndarray
    fitted_values: np.ndarray
    _model: sm.regression.linear_model.RegressionResultsWrapper

    def _encode(self, natural_point: Dict[str, float]) -> np.ndarray:
        coded = []
        for f in self.factors:
            lo, hi = self.factor_bounds[f]
            mid = 0.5 * (lo + hi)
            half = max(0.5 * (hi - lo), 1e-12)
            coded.append((natural_point[f] - mid) / half)
        return np.asarray(coded, dtype=float)

    def predict(self, natural_point: Dict[str, float]) -> float:
        """Predict response at a single natural-unit factor setting."""
        x_coded = self._encode(natural_point).reshape(1, -1)
        design_row = _build_quadratic_matrix(x_coded, self.factors)[0]
        return float(self._model.predict(design_row)[0])

    def surface_grid(
        self,
        factor_x: str,
        factor_y: str,
        fixed_values: Dict[str, float],
        n_grid: int = 35,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate response over a 2D meshgrid for contour/3D surface plotting."""
        lo_x, hi_x = self.factor_bounds[factor_x]
        lo_y, hi_y = self.factor_bounds[factor_y]
        gx = np.linspace(lo_x, hi_x, n_grid)
        gy = np.linspace(lo_y, hi_y, n_grid)
        XX, YY = np.meshgrid(gx, gy)

        pts = []
        for xv, yv in zip(XX.ravel(), YY.ravel()):
            row = dict(fixed_values)
            row[factor_x] = float(xv)
            row[factor_y] = float(yv)
            pts.append(self._encode(row))

        X_coded = np.asarray(pts)
        design_mat, _ = _build_quadratic_matrix(X_coded, self.factors)
        ZZ = np.asarray(self._model.predict(design_mat)).reshape(XX.shape)
        return XX, YY, ZZ


@dataclass
class OptimizationResult:
    goal: str
    optimal_factors: Dict[str, float]
    predicted_response: float
    desirability: float
    success: bool


def _build_quadratic_matrix(
    X_coded: np.ndarray,
    factors: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build [1, x_i, x_i^2, x_i*x_j] design matrix and readable term names."""
    n, k = X_coded.shape
    cols = [np.ones(n)]
    names = ["Intercept"]

    # Linear terms
    for i in range(k):
        cols.append(X_coded[:, i])
        names.append(factors[i])

    # Quadratic terms
    for i in range(k):
        cols.append(X_coded[:, i] ** 2)
        names.append(f"{factors[i]}^2")

    # Two-factor interaction terms
    for i in range(k):
        for j in range(i + 1, k):
            cols.append(X_coded[:, i] * X_coded[:, j])
            names.append(f"{factors[i]} * {factors[j]}")

    return np.column_stack(cols), names


def fit_response_surface(
    df: pd.DataFrame,
    factors: List[str],
    response: str,
) -> RSMResult:
    """Fit a coded 2nd-order Response Surface Model with exact PRESS Pred-R^2."""
    if len(factors) < 2:
        raise ValueError("Response surface DOE requires at least 2 factors.")

    clean = df[factors + [response]].dropna().copy()
    n = len(clean)
    k = len(factors)
    n_params = 1 + 2 * k + (k * (k - 1)) // 2
    if n <= n_params:
        raise ValueError(
            f"Need more than {n_params} observations to fit a full quadratic model for {k} factors (got {n})."
        )

    bounds: Dict[str, Tuple[float, float]] = {}
    X_coded = np.zeros((n, k), dtype=float)
    for j, f in enumerate(factors):
        lo = float(clean[f].min())
        hi = float(clean[f].max())
        if hi <= lo:
            raise ValueError(f"Factor '{f}' has zero variance in DOE table.")
        bounds[f] = (lo, hi)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        X_coded[:, j] = (clean[f].to_numpy(dtype=float) - mid) / half

    y = clean[response].to_numpy(dtype=float)
    X_mat, term_names = _build_quadratic_matrix(X_coded, factors)

    ols = sm.OLS(y, X_mat).fit()

    # Exact PRESS statistic & Predicted R^2 using hat-matrix diagonal
    influence = ols.get_influence()
    hat_diag = np.clip(influence.hat_matrix_diag, 0.0, 1.0 - 1e-9)
    press_resid = ols.resid / (1.0 - hat_diag)
    press = float(np.sum(press_resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    pred_r2 = 1.0 - (press / sst) if sst > 0 else 0.0

    summary_df = pd.DataFrame(
        {
            "Term": term_names,
            "Coded_Coeff": np.round(ols.params, 4),
            "Std_Err": np.round(ols.bse, 4),
            "t_Stat": np.round(ols.tvalues, 3),
            "p_Value": np.round(ols.pvalues, 4),
            "Significant (p<0.05)": ols.pvalues < 0.05,
        }
    )

    return RSMResult(
        factors=factors,
        response=response,
        factor_bounds=bounds,
        term_names=term_names,
        summary_table=summary_df,
        r_squared=float(ols.rsquared),
        adj_r_squared=float(ols.rsquared_adj),
        pred_r_squared=float(pred_r2),
        rmse=float(np.sqrt(ols.mse_resid)),
        f_stat=float(ols.fvalue) if ols.fvalue is not None else 0.0,
        f_pvalue=float(ols.f_pvalue) if ols.f_pvalue is not None else 1.0,
        n_obs=n,
        residuals=np.asarray(ols.resid),
        fitted_values=np.asarray(ols.fittedvalues),
        _model=ols,
    )


def optimize_response_surface(
    rsm: RSMResult,
    goal: Literal["target", "maximize", "minimize"] = "target",
    target: float = 0.0,
    lsl: float = -1.0,
    usl: float = 1.0,
) -> OptimizationResult:
    """Find optimal factor settings inside bounds using multi-start L-BFGS-B."""
    k = len(rsm.factors)
    coded_bounds = [(-1.0, 1.0)] * k

    def eval_coded(xc: np.ndarray) -> float:
        mat, _ = _build_quadratic_matrix(xc.reshape(1, -1), rsm.factors)
        return float(rsm._model.predict(mat)[0])

    def objective(xc: np.ndarray) -> float:
        y_hat = eval_coded(xc)
        if goal == "target":
            return (y_hat - target) ** 2
        if goal == "maximize":
            return -y_hat
        return y_hat

    # Multi-start from center + random interior seeds to avoid local optima
    rng = np.random.default_rng(7)
    starts = [np.zeros(k)] + [rng.uniform(-0.9, 0.9, k) for _ in range(12)]
    best_res = None
    for s in starts:
        res = minimize(objective, s, method="L-BFGS-B", bounds=coded_bounds)
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    assert best_res is not None
    x_opt_coded = np.clip(best_res.x, -1.0, 1.0)
    y_opt = eval_coded(x_opt_coded)

    opt_natural: Dict[str, float] = {}
    for j, f in enumerate(rsm.factors):
        lo, hi = rsm.factor_bounds[f]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        opt_natural[f] = round(mid + float(x_opt_coded[j]) * half, 3)

    # Derringer-Suich desirability score in [0, 1]
    if goal == "target":
        if y_opt <= lsl or y_opt >= usl:
            desirability = 0.0
        elif y_opt <= target:
            desirability = (y_opt - lsl) / max(target - lsl, 1e-12)
        else:
            desirability = (usl - y_opt) / max(usl - target, 1e-12)
    elif goal == "maximize":
        desirability = float(np.clip((y_opt - lsl) / max(usl - lsl, 1e-12), 0.0, 1.0))
    else:
        desirability = float(np.clip((usl - y_opt) / max(usl - lsl, 1e-12), 0.0, 1.0))

    return OptimizationResult(
        goal=goal,
        optimal_factors=opt_natural,
        predicted_response=round(y_opt, 3),
        desirability=round(float(desirability), 3),
        success=bool(best_res.success),
    )
