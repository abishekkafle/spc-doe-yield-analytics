"""SPC, DOE & Yield Analytics — Interactive Process Engineering Web App."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from src.anomaly import detect_multivariate_anomalies
from src.data_generator import (
    PROCESS_SPECS,
    generate_doe_ccd_data,
    generate_production_data,
)
from src.doe import fit_response_surface, optimize_response_surface
from src.report_8d import generate_8d_report
from src.spc import compute_capability, compute_imr_chart, compute_xbar_r_chart

st.set_page_config(
    page_title="SPC, DOE & Yield Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling polish
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #F4F6F9;
        border: 1px solid #E1E7ED;
        border-radius: 8px;
        padding: 12px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Sidebar
st.sidebar.title("Process Configuration")
data_mode = st.sidebar.radio(
    "Data Source",
    ["Built-in Manufacturing Dataset", "Upload Custom CSV"],
    index=0,
)

uploaded_df = None
if data_mode == "Built-in Manufacturing Dataset":
    process_key = st.sidebar.selectbox(
        "Manufacturing Process",
        options=list(PROCESS_SPECS.keys()),
        format_func=lambda k: PROCESS_SPECS[k].name,
    )
    inject_anomalies = st.sidebar.checkbox(
        "Inject Realistic Drift & Excursions",
        value=True,
        help="Toggle off to compare an in-control baseline against an out-of-control production line.",
    )
    n_runs = st.sidebar.slider("Production Runs", min_value=60, max_value=300, value=180, step=20)
    seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)

    df, spec = generate_production_data(
        process_key=process_key,
        n_runs=n_runs,
        seed=int(seed),
        inject_anomalies=inject_anomalies,
    )
    doe_df, _ = generate_doe_ccd_data(process_key=process_key, seed=int(seed) + 59)

    process_name = spec.name
    ctq_col = spec.primary_ctq
    ctq_unit = spec.ctq_unit
    sensor_cols = spec.process_factors
    default_lsl, default_target, default_usl = spec.lsl, spec.target, spec.usl
else:
    uploaded = st.sidebar.file_uploader("Upload Production CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file with at least one numeric CTQ column and 2+ numeric process sensor columns.")
        st.stop()
    df = pd.read_csv(uploaded)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 1:
        st.error("Uploaded CSV must contain at least one numeric column.")
        st.stop()
    if "Run_ID" not in df.columns:
        df.insert(0, "Run_ID", np.arange(1, len(df) + 1))
    ctq_col = st.sidebar.selectbox("Primary CTQ Column", num_cols, index=len(num_cols) - 1)
    candidate_sensors = [c for c in num_cols if c not in (ctq_col, "Run_ID", "Subgroup")]
    sensor_cols = st.sidebar.multiselect(
        "Process Sensor Columns (2+ for Multivariate & DOE)",
        candidate_sensors,
        default=candidate_sensors[: min(4, len(candidate_sensors))],
    )
    vals = df[ctq_col].dropna()
    mean_v, std_v = float(vals.mean()), float(vals.std(ddof=1))
    default_lsl = round(mean_v - 3.0 * std_v, 2)
    default_target = round(mean_v, 2)
    default_usl = round(mean_v + 3.0 * std_v, 2)
    process_name = f"Custom Process ({uploaded.name})"
    ctq_unit = "units"
    spec = None
    doe_df = df[sensor_cols[:3] + [ctq_col]].dropna() if len(sensor_cols) >= 2 else None

st.sidebar.markdown("---")
st.sidebar.subheader("Specification Limits")
col_s1, col_s2 = st.sidebar.columns(2)
lsl = col_s1.number_input("LSL", value=float(default_lsl))
usl = col_s2.number_input("USL", value=float(default_usl))
target = st.sidebar.number_input("Target", value=float(default_target))
if usl <= lsl:
    st.sidebar.error("USL must be greater than LSL.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Chart & Detector Settings")
subgroup_size = st.sidebar.slider("X-bar/R Subgroup Size (n)", 2, 10, 5)
t2_alpha = st.sidebar.select_slider("Hotelling T² Alpha (α)", options=[0.05, 0.01, 0.005, 0.001], value=0.01)
iforest_contam = st.sidebar.slider("Isolation Forest Contamination", 0.02, 0.15, 0.06, 0.01)

# ------------------------------------------------------------------ Core Computations
imr = compute_imr_chart(df[ctq_col])
xbar_r = compute_xbar_r_chart(df[ctq_col], subgroup_size=subgroup_size)
cap = compute_capability(df[ctq_col], lsl=lsl, usl=usl, target=target, sigma_within=imr.sigma_within)

anomaly_res = None
if len(sensor_cols) >= 2:
    try:
        anomaly_res = detect_multivariate_anomalies(
            df,
            features=sensor_cols,
            alpha=float(t2_alpha),
            contamination=float(iforest_contam),
        )
    except ValueError:
        anomaly_res = None

rsm_res = None
opt_res = None
if doe_df is not None and len(sensor_cols) >= 2:
    doe_factors = [c for c in sensor_cols[:3] if c in doe_df.columns]
    if len(doe_factors) >= 2 and ctq_col in doe_df.columns:
        try:
            rsm_res = fit_response_surface(doe_df, factors=doe_factors, response=ctq_col)
            opt_res = optimize_response_surface(rsm_res, goal="target", target=target, lsl=lsl, usl=usl)
        except ValueError:
            rsm_res = None

# ------------------------------------------------------------------ Header & KPIs
st.title("SPC, DOE & Yield Analytics Platform")
st.caption(f"**Active Process:** {process_name}  |  **CTQ:** `{ctq_col}` ({ctq_unit})  |  **n = {cap.n} runs**")
if spec is not None:
    with st.expander("Process Context & Known Ground-Truth Events", expanded=False):
        st.write(spec.description)
        if inject_anomalies:
            st.markdown("**Injected Ground-Truth Events (for validation):**")
            for ev in spec.ground_truth_events:
                st.markdown(f"- {ev}")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Cpk (Short-Term)", f"{cap.cpk:.2f}", delta=f"Cp {cap.cp:.2f}")
k2.metric("Ppk (Long-Term)", f"{cap.ppk:.2f}", delta=f"Pp {cap.pp:.2f}")
k3.metric("Observed Yield", f"{cap.observed_yield_pct:.2f}%", delta=f"Exp {cap.expected_yield_pct:.2f}%")
k4.metric("Sigma Level", f"{cap.sigma_level:.2f}σ", delta=f"{cap.dpmo:,.0f} DPMO", delta_color="inverse")
k5.metric("SPC OOC Signals", f"{len(imr.ooc_indices)}", delta=f"{len(imr.violations)} Nelson rules", delta_color="inverse")
k6.metric(
    "Sensor Excursions",
    f"{int(anomaly_res.combined_flags.sum())}" if anomaly_res is not None else "N/A",
    delta=f"T² UCL {anomaly_res.t2_ucl:.1f}" if anomaly_res is not None else "",
    delta_color="off",
)

# ------------------------------------------------------------------ Tabs
tab_spc, tab_cap, tab_doe, tab_anom, tab_8d = st.tabs(
    [
        "1. SPC Control Charts",
        "2. Capability (Cp/Cpk)",
        "3. Response Surface DOE",
        "4. Multivariate Anomalies",
        "5. Automated 8D Report",
    ]
)

# === TAB 1: SPC ===
with tab_spc:
    st.subheader(f"Individuals & Moving Range (I-MR) Chart — {ctq_col}")
    run_x = df["Run_ID"].to_numpy()[: len(imr.values)] if "Run_ID" in df.columns else np.arange(1, len(imr.values) + 1)

    fig_imr = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"Individuals (I) Chart — {ctq_col} ({ctq_unit})", "Moving Range (MR) Chart"),
    )
    fig_imr.add_trace(
        go.Scatter(x=run_x, y=imr.values, mode="lines+markers", name="Measured CTQ", line=dict(color="#1B4F72", width=1.8), marker=dict(size=5)),
        row=1,
        col=1,
    )
    if imr.ooc_indices:
        fig_imr.add_trace(
            go.Scatter(
                x=run_x[imr.ooc_indices],
                y=imr.values[imr.ooc_indices],
                mode="markers",
                name="Nelson Rule Violation",
                marker=dict(color="#C0392B", size=9, symbol="diamond"),
            ),
            row=1,
            col=1,
        )
    for y_val, label, dash, color in [
        (imr.ucl_i, f"UCL ({imr.ucl_i:.1f})", "dash", "#C0392B"),
        (imr.x_bar, f"CL ({imr.x_bar:.1f})", "solid", "#2E86C1"),
        (imr.lcl_i, f"LCL ({imr.lcl_i:.1f})", "dash", "#C0392B"),
        (usl, f"USL ({usl:.1f})", "dot", "#7D3C98"),
        (lsl, f"LSL ({lsl:.1f})", "dot", "#7D3C98"),
    ]:
        fig_imr.add_hline(y=y_val, line_dash=dash, line_color=color, annotation_text=label, annotation_position="right", row=1, col=1)

    fig_imr.add_trace(
        go.Scatter(x=run_x[1:], y=imr.moving_ranges, mode="lines+markers", name="Moving Range", line=dict(color="#2874A6", width=1.5), marker=dict(size=4)),
        row=2,
        col=1,
    )
    fig_imr.add_hline(y=imr.ucl_mr, line_dash="dash", line_color="#C0392B", annotation_text=f"UCL ({imr.ucl_mr:.1f})", row=2, col=1)
    fig_imr.add_hline(y=imr.mr_bar, line_dash="solid", line_color="#2E86C1", annotation_text=f"MR-bar ({imr.mr_bar:.1f})", row=2, col=1)
    fig_imr.update_layout(height=540, margin=dict(l=20, r=90, t=40, b=20), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_imr, use_container_width=True)

    c_left, c_right = st.columns([1, 1.3])
    with c_left:
        st.markdown("#### Nelson Rules Summary")
        if imr.violations:
            v_df = pd.DataFrame(
                [
                    {
                        "Rule": v.rule,
                        "Signals": len(v.indices),
                        "First Run": int(run_x[v.indices[0]]),
                        "Meaning": v.description,
                    }
                    for v in imr.violations
                ]
            )
            st.dataframe(v_df, use_container_width=True, hide_index=True)
        else:
            st.success("Process is statistically in control (0 Nelson Rule 1–4 violations).")

    with c_right:
        st.markdown(f"#### X-bar Subgroup Chart (n = {subgroup_size})")
        fig_xb = go.Figure()
        fig_xb.add_trace(go.Scatter(x=xbar_r.subgroup_labels, y=xbar_r.x_bars, mode="lines+markers", name="Subgroup Mean", line=dict(color="#117A65")))
        if xbar_r.ooc_x_indices:
            fig_xb.add_trace(
                go.Scatter(
                    x=[xbar_r.subgroup_labels[i] for i in xbar_r.ooc_x_indices],
                    y=xbar_r.x_bars[xbar_r.ooc_x_indices],
                    mode="markers",
                    name="OOC Subgroup",
                    marker=dict(color="#C0392B", size=9, symbol="x"),
                )
            )
        fig_xb.add_hline(y=xbar_r.ucl_x, line_dash="dash", line_color="#C0392B", annotation_text="UCL")
        fig_xb.add_hline(y=xbar_r.x_double_bar, line_dash="solid", line_color="#2E86C1", annotation_text="X-double-bar")
        fig_xb.add_hline(y=xbar_r.lcl_x, line_dash="dash", line_color="#C0392B", annotation_text="LCL")
        fig_xb.update_layout(height=300, margin=dict(l=10, r=40, t=20, b=20))
        st.plotly_chart(fig_xb, use_container_width=True)

# === TAB 2: CAPABILITY ===
with tab_cap:
    st.subheader(f"Process Capability Analysis — {ctq_col}")
    if not cap.is_normal:
        st.warning(
            f"Shapiro-Wilk normality test returned **p = {cap.normality_p_value:.4f} (< 0.05)**. "
            "Data deviates from normality (likely due to process shifts/drift). Interpret normal-theory Cp/Cpk with caution."
        )
    else:
        st.success(f"Shapiro-Wilk normality check passed (p = {cap.normality_p_value:.4f} ≥ 0.05).")

    vals = df[ctq_col].dropna().to_numpy()
    x_grid = np.linspace(min(vals.min(), lsl) - cap.std_overall, max(vals.max(), usl) + cap.std_overall, 250)
    pdf_overall = stats.norm.pdf(x_grid, loc=cap.mean, scale=max(cap.std_overall, 1e-9))
    pdf_within = stats.norm.pdf(x_grid, loc=cap.mean, scale=max(cap.std_within, 1e-9))

    fig_cap = go.Figure()
    fig_cap.add_trace(
        go.Histogram(x=vals, histnorm="probability density", nbinsx=28, name="Observed Distribution", marker_color="#5DADE2", opacity=0.65)
    )
    fig_cap.add_trace(go.Scatter(x=x_grid, y=pdf_within, mode="lines", name="Within-Subgroup Normal (Cpk)", line=dict(color="#1B4F72", width=2.4)))
    fig_cap.add_trace(go.Scatter(x=x_grid, y=pdf_overall, mode="lines", name="Overall Normal (Ppk)", line=dict(color="#E67E22", width=2.2, dash="dash")))
    fig_cap.add_vline(x=lsl, line_color="#C0392B", line_dash="dash", annotation_text=f"LSL ({lsl})")
    fig_cap.add_vline(x=target, line_color="#1E8449", line_dash="dot", annotation_text=f"Target ({target})")
    fig_cap.add_vline(x=usl, line_color="#C0392B", line_dash="dash", annotation_text=f"USL ({usl})")
    fig_cap.update_layout(height=420, xaxis_title=f"{ctq_col} ({ctq_unit})", yaxis_title="Probability Density", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cap, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Capability & Yield Table")
        cap_df = pd.DataFrame(
            {
                "Metric": [
                    "Sample Size (n)",
                    "Mean",
                    "Within-Subgroup Std (σ_within)",
                    "Overall Std (σ_overall)",
                    "Cp / Cpk",
                    "Pp / Ppk",
                    "Observed Yield / Expected Yield",
                    "DPMO / Sigma Level",
                ],
                "Value": [
                    f"{cap.n}",
                    f"{cap.mean:.3f} {ctq_unit}",
                    f"{cap.std_within:.3f} {ctq_unit}",
                    f"{cap.std_overall:.3f} {ctq_unit}",
                    f"{cap.cp:.3f} / {cap.cpk:.3f}",
                    f"{cap.pp:.3f} / {cap.ppk:.3f}",
                    f"{cap.observed_yield_pct:.2f}% / {cap.expected_yield_pct:.2f}%",
                    f"{cap.dpmo:,.0f} / {cap.sigma_level:.2f}σ",
                ],
            }
        )
        st.dataframe(cap_df, use_container_width=True, hide_index=True)

    with c2:
        if "Shift" in df.columns:
            st.markdown("#### Shift-to-Shift Stratification")
            fig_shift = go.Figure()
            for sh, grp in df.groupby("Shift"):
                fig_shift.add_trace(go.Box(y=grp[ctq_col], name=str(sh), boxmean="sd"))
            fig_shift.add_hline(y=lsl, line_dash="dash", line_color="#C0392B")
            fig_shift.add_hline(y=usl, line_dash="dash", line_color="#C0392B")
            fig_shift.update_layout(height=320, yaxis_title=ctq_col, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_shift, use_container_width=True)

# === TAB 3: RESPONSE SURFACE DOE ===
with tab_doe:
    st.subheader("Response Surface Methodology (RSM) & Process Window Optimization")
    if rsm_res is None:
        st.info("Provide at least 2 numeric process factors with sufficient observations to fit a 2nd-order Response Surface Model.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{rsm_res.r_squared:.3f}")
        m2.metric("Adjusted R²", f"{rsm_res.adj_r_squared:.3f}")
        m3.metric("Predicted R² (PRESS)", f"{rsm_res.pred_r_squared:.3f}")
        m4.metric("Model RMSE", f"{rsm_res.rmse:.2f} {ctq_unit}")

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown("#### Coded Quadratic Regression & Term Significance")
            st.dataframe(rsm_res.summary_table, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Numerical Desirability Optimizer")
            opt_goal = st.selectbox("Optimization Goal", ["target", "maximize", "minimize"], index=0)
            opt_live = optimize_response_surface(rsm_res, goal=opt_goal, target=target, lsl=lsl, usl=usl)
            st.success(
                f"**Predicted {ctq_col}:** `{opt_live.predicted_response} {ctq_unit}`  |  "
                f"**Desirability:** `{opt_live.desirability:.3f}`"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Process Factor": k, "Optimal Setting": v, "Design Bounds": f"{rsm_res.factor_bounds[k][0]} – {rsm_res.factor_bounds[k][1]}"}
                        for k, v in opt_live.optimal_factors.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Interactive 2D Contour & 3D Response Surface")
        p1, p2, p3 = st.columns(3)
        fx = p1.selectbox("X-Axis Factor", rsm_res.factors, index=0)
        fy_options = [f for f in rsm_res.factors if f != fx]
        fy = p2.selectbox("Y-Axis Factor", fy_options, index=0)
        fixed_vals = {f: 0.5 * sum(rsm_res.factor_bounds[f]) for f in rsm_res.factors}
        other_factors = [f for f in rsm_res.factors if f not in (fx, fy)]
        if other_factors:
            f_hold = other_factors[0]
            lo_h, hi_h = rsm_res.factor_bounds[f_hold]
            fixed_vals[f_hold] = p3.slider(f"Hold {f_hold}", float(lo_h), float(hi_h), float(fixed_vals[f_hold]))

        XX, YY, ZZ = rsm_res.surface_grid(fx, fy, fixed_vals)
        g1, g2 = st.columns(2)
        with g1:
            fig_cont = go.Figure(
                data=go.Contour(
                    x=XX[0],
                    y=YY[:, 0],
                    z=ZZ,
                    colorscale="Viridis",
                    contours=dict(showlabels=True),
                    colorbar=dict(title=ctq_unit),
                )
            )
            fig_cont.update_layout(title="2D Process Window Contour", xaxis_title=fx, yaxis_title=fy, height=400)
            st.plotly_chart(fig_cont, use_container_width=True)

        with g2:
            fig_3d = go.Figure(data=[go.Surface(x=XX, y=YY, z=ZZ, colorscale="Viridis", colorbar=dict(title=ctq_unit))])
            fig_3d.update_layout(
                title="3D Response Surface",
                scene=dict(xaxis_title=fx, yaxis_title=fy, zaxis_title=ctq_col),
                height=400,
                margin=dict(l=0, r=0, t=35, b=0),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with st.expander("View CCD Experimental Design Matrix"):
            st.dataframe(doe_df, use_container_width=True, hide_index=True)

# === TAB 4: MULTIVARIATE ANOMALIES ===
with tab_anom:
    st.subheader("Multivariate Excursion Detection (Hotelling's T² + Isolation Forest)")
    if anomaly_res is None:
        st.info("Select at least 2 numeric process sensor columns with ≥20 rows to run multivariate anomaly detection.")
    else:
        run_ids = df["Run_ID"].to_numpy()[: len(anomaly_res.t2_scores)]
        fig_t2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Hotelling's T² Multivariate Control Chart (Phase II UCL)", "Isolation Forest Anomaly Score"),
        )
        fig_t2.add_trace(
            go.Scatter(x=run_ids, y=anomaly_res.t2_scores, mode="lines+markers", name="Hotelling T²", line=dict(color="#1B4F72"), marker=dict(size=4)),
            row=1,
            col=1,
        )
        t2_idx = np.where(anomaly_res.t2_flags)[0]
        if len(t2_idx):
            fig_t2.add_trace(
                go.Scatter(
                    x=run_ids[t2_idx],
                    y=anomaly_res.t2_scores[t2_idx],
                    mode="markers",
                    name="T² Excursion",
                    marker=dict(color="#C0392B", size=8, symbol="diamond"),
                ),
                row=1,
                col=1,
            )
        fig_t2.add_hline(y=anomaly_res.t2_ucl, line_dash="dash", line_color="#C0392B", annotation_text=f"UCL ({anomaly_res.t2_ucl:.1f})", row=1, col=1)

        fig_t2.add_trace(
            go.Scatter(x=run_ids, y=anomaly_res.iforest_scores, mode="lines", name="IForest Score", line=dict(color="#117A65")),
            row=2,
            col=1,
        )
        if_idx = np.where(anomaly_res.iforest_flags)[0]
        if len(if_idx):
            fig_t2.add_trace(
                go.Scatter(
                    x=run_ids[if_idx],
                    y=anomaly_res.iforest_scores[if_idx],
                    mode="markers",
                    name="IForest Outlier",
                    marker=dict(color="#D35400", size=8, symbol="x"),
                ),
                row=2,
                col=1,
            )
        fig_t2.update_layout(height=500, margin=dict(l=20, r=40, t=35, b=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_t2, use_container_width=True)

        a1, a2 = st.columns([1, 1.6])
        with a1:
            st.markdown("#### Primary Sensor Attribution")
            drv_df = pd.DataFrame(
                [{"Sensor": k, "Excursions_Driven": v} for k, v in anomaly_res.driver_summary.items()]
            ).sort_values("Excursions_Driven", ascending=False)
            fig_drv = go.Figure(go.Bar(x=drv_df["Sensor"], y=drv_df["Excursions_Driven"], marker_color="#2874A6"))
            fig_drv.update_layout(height=300, yaxis_title="Flagged Runs", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_drv, use_container_width=True)

        with a2:
            st.markdown("#### Flagged Excursion Log")
            if not anomaly_res.excursion_table.empty:
                st.dataframe(anomaly_res.excursion_table, use_container_width=True, hide_index=True, height=300)
            else:
                st.success("No multivariate excursions detected.")

# === TAB 5: AUTOMATED 8D REPORT ===
with tab_8d:
    st.subheader("Automated 8D Root-Cause & CAPA Summary")
    report_md = generate_8d_report(
        process_name=process_name,
        ctq_name=ctq_col,
        ctq_unit=ctq_unit,
        imr=imr,
        cap=cap,
        anomaly=anomaly_res,
        rsm=rsm_res,
        opt=opt_res,
    )
    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button(
        label="Download 8D Report (.md)",
        data=report_md,
        file_name=f"8D_Report_{ctq_col}.md",
        mime="text/markdown",
    )
    b2.download_button(
        label="Download Production Data (.csv)",
        data=df.to_csv(index=False),
        file_name=f"production_data_{ctq_col}.csv",
        mime="text/csv",
    )
    st.markdown("---")
    st.markdown(report_md)
