/**
 * Interactive UI controller for the static HTML/CSS/Plotly.js SPC, DOE & Yield Analytics app.
 */

const A = window.SPCAnalytics;

const state = {
  mode: "preset", // preset | csv
  processKey: "cold_spray",
  injectAnomalies: true,
  nRuns: 180,
  subgroupSize: 5,
  alpha: 0.01,
  optGoal: "target",
  lsl: 460,
  target: 500,
  usl: 540,
  customRows: null,
  customName: "",
  customCtq: "",
  customFactors: [],
  fx: null,
  fy: null,
  holdVal: null,
  lastReportMd: "",
  lastRows: [],
};

const el = (id) => document.getElementById(id);

function loadPresetSpecs(key) {
  const spec = A.PROCESS_SPECS[key];
  state.processKey = key;
  state.lsl = spec.lsl;
  state.target = spec.target;
  state.usl = spec.usl;
  el("lslInput").value = spec.lsl;
  el("targetInput").value = spec.target;
  el("uslInput").value = spec.usl;
  state.fx = spec.factors[0];
  state.fy = spec.factors[1];
  state.holdVal = null;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 10) return null;
  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length !== headers.length) continue;
    const obj = { Run_ID: i };
    headers.forEach((h, j) => {
      const v = parts[j].trim().replace(/^"|"$/g, "");
      const num = Number(v);
      obj[h] = v !== "" && !Number.isNaN(num) ? num : v;
    });
    rows.push(obj);
  }
  const numericCols = headers.filter((h) => h !== "Run_ID" && rows.every((r) => typeof r[h] === "number"));
  return { rows, numericCols };
}

function getActiveData() {
  if (state.mode === "csv" && state.customRows) {
    return {
      rows: state.customRows,
      doeRows: state.customRows,
      processName: `Custom CSV (${state.customName})`,
      ctq: state.customCtq,
      unit: "units",
      factors: state.customFactors,
      events: [],
    };
  }
  const { rows, spec } = A.generateProductionData(state.processKey, state.nRuns, state.injectAnomalies, 42);
  const doeRows = A.generateDoeCCD(state.processKey, 101);
  return {
    rows,
    doeRows,
    processName: spec.name,
    ctq: spec.ctq,
    unit: spec.unit,
    factors: spec.factors,
    events: state.injectAnomalies ? spec.events : [],
  };
}

function renderAll() {
  const data = getActiveData();
  state.lastRows = data.rows;
  const values = data.rows.map((r) => r[data.ctq]).filter((v) => typeof v === "number" && !Number.isNaN(v));
  const runIds = data.rows.map((r, i) => r.Run_ID || i + 1);

  const lsl = state.lsl, target = state.target, usl = state.usl;
  if (usl <= lsl) return;

  const imr = A.computeIMR(values);
  const xbr = A.computeXbarR(values, state.subgroupSize);
  const cap = A.computeCapability(values, lsl, usl, target, imr.sigmaWithin);
  const anom = data.factors.length >= 2 ? A.detectHotellingAnomalies(data.rows, data.factors, state.alpha) : null;

  const doeFactors = data.factors.slice(0, 3);
  let rsm = null, opt = null;
  if (doeFactors.length >= 2 && data.doeRows.length > 10) {
    try {
      rsm = A.fitRSM(data.doeRows, doeFactors, data.ctq);
      opt = rsm.optimize(state.optGoal, target, lsl, usl);
    } catch (e) {
      rsm = null;
    }
  }

  // Update header context & KPIs
  el("activeProcessLabel").textContent = `${data.processName}  •  CTQ: ${data.ctq} (${data.unit})  •  n = ${cap.n} runs`;
  renderKPIs(cap, imr, anom);
  renderSPC(runIds, imr, xbr, data, lsl, usl);
  renderCapability(values, data, cap, lsl, target, usl);
  renderDOE(rsm, opt, data);
  renderAnomalies(runIds, anom);
  render8D(data, imr, cap, anom, rsm, opt);
}

function renderKPIs(cap, imr, anom) {
  const cpkClass = cap.cpk >= 1.33 ? "ok" : cap.cpk >= 1.0 ? "warn" : "danger";
  const ppkClass = cap.ppk >= 1.33 ? "ok" : cap.ppk >= 1.0 ? "warn" : "danger";
  const oocClass = imr.oocIndices.length === 0 ? "ok" : "danger";
  const anomCount = anom ? anom.excursions.length : 0;
  const anomClass = anomCount === 0 ? "ok" : "warn";

  el("kpiGrid").innerHTML = `
    <div class="kpi-card ${cpkClass}">
      <div class="kpi-label">Cpk (Short-Term)</div>
      <div class="kpi-value">${cap.cpk.toFixed(2)}</div>
      <div class="kpi-sub">Cp = ${cap.cp.toFixed(2)} (Goal ≥ 1.33)</div>
    </div>
    <div class="kpi-card ${ppkClass}">
      <div class="kpi-label">Ppk (Long-Term)</div>
      <div class="kpi-value">${cap.ppk.toFixed(2)}</div>
      <div class="kpi-sub">Pp = ${cap.pp.toFixed(2)}</div>
    </div>
    <div class="kpi-card ${cap.obsYield >= 98 ? "ok" : "warn"}">
      <div class="kpi-label">Observed Yield</div>
      <div class="kpi-value">${cap.obsYield.toFixed(2)}%</div>
      <div class="kpi-sub">Expected ${cap.expYield.toFixed(2)}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Sigma Level</div>
      <div class="kpi-value">${cap.sigmaLevel.toFixed(2)}σ</div>
      <div class="kpi-sub">${Math.round(cap.dpmo).toLocaleString()} DPMO</div>
    </div>
    <div class="kpi-card ${oocClass}">
      <div class="kpi-label">SPC OOC Signals</div>
      <div class="kpi-value">${imr.oocIndices.length}</div>
      <div class="kpi-sub">${imr.violations.length} Nelson rule(s)</div>
    </div>
    <div class="kpi-card ${anomClass}">
      <div class="kpi-label">Sensor Excursions</div>
      <div class="kpi-value">${anomCount}</div>
      <div class="kpi-sub">Hotelling T² UCL ${anom ? anom.ucl : "—"}</div>
    </div>
  `;
}

function renderSPC(runIds, imr, xbr, data, lsl, usl) {
  const insight =
    imr.oocIndices.length === 0
      ? `<div class="alert alert-ok"><strong>In Control:</strong> Zero Nelson Rule violations detected. Process variation is governed by common-cause noise.</div>`
      : `<div class="alert alert-danger"><strong>Special-Cause Signal Detected:</strong> ${imr.oocIndices.length} run(s) violated Nelson Rules (${imr.violations.map((v) => v.rule).join(", ")}). Check Tab 4 for the driving sensor(s).</div>`;
  el("spcInsight").innerHTML = insight;

  const traces = [
    { x: runIds, y: imr.values, mode: "lines+markers", name: data.ctq, line: { color: "#145374", width: 1.8 }, marker: { size: 4 } },
  ];
  if (imr.oocIndices.length) {
    traces.push({
      x: imr.oocIndices.map((i) => runIds[i]),
      y: imr.oocIndices.map((i) => imr.values[i]),
      mode: "markers",
      name: "Nelson Violation",
      marker: { color: "#c0392b", size: 9, symbol: "diamond" },
    });
  }

  const shapes = [
    { y: imr.uclI, color: "#c0392b", dash: "dash" },
    { y: imr.xBar, color: "#2e86c1", dash: "solid" },
    { y: imr.lclI, color: "#c0392b", dash: "dash" },
    { y: usl, color: "#7d3c98", dash: "dot" },
    { y: lsl, color: "#7d3c98", dash: "dot" },
  ].map((s) => ({
    type: "line", xref: "paper", x0: 0, x1: 1, y0: s.y, y1: s.y,
    line: { color: s.color, width: 1.5, dash: s.dash },
  }));

  Plotly.newPlot("imrChart", traces, {
    margin: { t: 24, r: 24, b: 42, l: 58 },
    xaxis: { title: "Run ID" },
    yaxis: { title: `${data.ctq} (${data.unit})` },
    shapes,
    legend: { orientation: "h", y: 1.12 },
    annotations: [
      { xref: "paper", x: 1.0, y: imr.uclI, text: `UCL ${imr.uclI.toFixed(1)}`, showarrow: false, xanchor: "right", font: { size: 11, color: "#c0392b" } },
      { xref: "paper", x: 1.0, y: imr.xBar, text: `CL ${imr.xBar.toFixed(1)}`, showarrow: false, xanchor: "right", font: { size: 11, color: "#2e86c1" } },
      { xref: "paper", x: 1.0, y: imr.lclI, text: `LCL ${imr.lclI.toFixed(1)}`, showarrow: false, xanchor: "right", font: { size: 11, color: "#c0392b" } },
    ],
  }, { responsive: true });

  // X-bar chart
  const xbTraces = [
    { x: xbr.labels, y: xbr.xBars, mode: "lines+markers", name: "Subgroup Mean", line: { color: "#0f766e", width: 2 } },
  ];
  if (xbr.oocX.length) {
    xbTraces.push({
      x: xbr.oocX.map((i) => xbr.labels[i]),
      y: xbr.oocX.map((i) => xbr.xBars[i]),
      mode: "markers",
      name: "OOC Subgroup",
      marker: { color: "#c0392b", size: 9, symbol: "x" },
    });
  }
  Plotly.newPlot("xbarChart", xbTraces, {
    margin: { t: 20, r: 20, b: 40, l: 55 },
    yaxis: { title: `X-bar (n=${state.subgroupSize})` },
    shapes: [
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: xbr.uclX, y1: xbr.uclX, line: { color: "#c0392b", dash: "dash" } },
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: xbr.xDoubleBar, y1: xbr.xDoubleBar, line: { color: "#2e86c1" } },
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: xbr.lclX, y1: xbr.lclX, line: { color: "#c0392b", dash: "dash" } },
    ],
    legend: { orientation: "h", y: 1.14 },
  }, { responsive: true });

  // Nelson table
  if (!imr.violations.length) {
    el("nelsonTable").innerHTML = `<p style="padding:12px;margin:0;color:#1e8449;font-weight:600;">No Nelson Rule 1–3 violations detected.</p>`;
  } else {
    el("nelsonTable").innerHTML = `<table><thead><tr><th>Rule</th><th>Signals</th><th>First Run</th><th>Engineering Meaning</th></tr></thead><tbody>${
      imr.violations.map((v) => `<tr><td><span class="pill pill-alert">${v.rule}</span></td><td>${v.indices.length}</td><td>#${runIds[v.indices[0]]}</td><td>${v.desc}</td></tr>`).join("")
    }</tbody></table>`;
  }
}

function renderCapability(values, data, cap, lsl, target, usl) {
  el("capInsight").innerHTML = cap.isNormal
    ? `<div class="alert alert-ok"><strong>Normality Check Passed</strong> (Jarque-Bera = ${cap.jb.toFixed(2)} < 5.99). Standard normal-theory Cp/Cpk and DPMO estimates are valid.</div>`
    : `<div class="alert alert-warn"><strong>Normality Caution</strong> (Jarque-Bera = ${cap.jb.toFixed(2)} ≥ 5.99). Process drift/shifts cause non-normal tails — inspect Cp (${cap.cp.toFixed(2)}) vs Ppk (${cap.ppk.toFixed(2)}) gap.</div>`;

  const lo = Math.min(Math.min(...values), lsl) - cap.stdOverall;
  const hi = Math.max(Math.max(...values), usl) + cap.stdOverall;
  const xs = Array.from({ length: 160 }, (_, i) => lo + ((hi - lo) * i) / 159);

  Plotly.newPlot("capChart", [
    { x: values, type: "histogram", histnorm: "probability density", nbinsx: 26, name: "Observed", marker: { color: "rgba(20,83,116,0.45)" } },
    { x: xs, y: xs.map((x) => A.normPdf(x, cap.mean, cap.stdWithin)), mode: "lines", name: "Within-Subgroup (Cpk)", line: { color: "#145374", width: 2.4 } },
    { x: xs, y: xs.map((x) => A.normPdf(x, cap.mean, cap.stdOverall)), mode: "lines", name: "Overall (Ppk)", line: { color: "#e67e22", width: 2.2, dash: "dash" } },
  ], {
    margin: { t: 25, r: 20, b: 45, l: 55 },
    xaxis: { title: `${data.ctq} (${data.unit})` },
    yaxis: { title: "Density" },
    legend: { orientation: "h", y: 1.12 },
    shapes: [
      { type: "line", yref: "paper", y0: 0, y1: 1, x0: lsl, x1: lsl, line: { color: "#c0392b", dash: "dash", width: 2 } },
      { type: "line", yref: "paper", y0: 0, y1: 1, x0: target, x1: target, line: { color: "#1e8449", dash: "dot", width: 2 } },
      { type: "line", yref: "paper", y0: 0, y1: 1, x0: usl, x1: usl, line: { color: "#c0392b", dash: "dash", width: 2 } },
    ],
  }, { responsive: true });

  el("capTable").innerHTML = `<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
    <tr><td>Sample Size (n)</td><td>${cap.n}</td></tr>
    <tr><td>Mean</td><td>${cap.mean.toFixed(3)} ${data.unit}</td></tr>
    <tr><td>Within-Subgroup σ (Short-Term)</td><td>${cap.stdWithin.toFixed(3)} ${data.unit}</td></tr>
    <tr><td>Overall σ (Long-Term)</td><td>${cap.stdOverall.toFixed(3)} ${data.unit}</td></tr>
    <tr><td>Cp / Cpk</td><td><strong>${cap.cp.toFixed(3)} / ${cap.cpk.toFixed(3)}</strong></td></tr>
    <tr><td>Pp / Ppk</td><td><strong>${cap.pp.toFixed(3)} / ${cap.ppk.toFixed(3)}</strong></td></tr>
    <tr><td>Observed / Expected Yield</td><td>${cap.obsYield.toFixed(2)}% / ${cap.expYield.toFixed(2)}%</td></tr>
    <tr><td>DPMO / Sigma Level</td><td>${Math.round(cap.dpmo).toLocaleString()} / ${cap.sigmaLevel.toFixed(2)}σ</td></tr>
  </tbody></table>`;

  // Shift stratification boxplot
  if (data.rows[0] && data.rows[0].Shift) {
    const groups = {};
    data.rows.forEach((r) => { (groups[r.Shift] = groups[r.Shift] || []).push(r[data.ctq]); });
    Plotly.newPlot("shiftChart", Object.entries(groups).map(([sh, arr]) => ({ y: arr, type: "box", name: sh, boxmean: "sd" })), {
      margin: { t: 16, r: 16, b: 36, l: 55 },
      yaxis: { title: data.ctq },
      shapes: [
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: lsl, y1: lsl, line: { color: "#c0392b", dash: "dash" } },
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: usl, y1: usl, line: { color: "#c0392b", dash: "dash" } },
      ],
    }, { responsive: true });
  }
}

function renderDOE(rsm, opt, data) {
  if (!rsm) {
    el("doeMetrics").innerHTML = `<div class="alert alert-info">Upload a CSV with 2+ numeric process factors and >10 rows to fit a Response Surface Model.</div>`;
    return;
  }

  el("doeMetrics").innerHTML = `
    <div class="alert alert-info">
      <strong>RSM Model Quality:</strong>
      R² = <strong>${rsm.r2.toFixed(3)}</strong> &nbsp;|&nbsp;
      Adj-R² = <strong>${rsm.adjR2.toFixed(3)}</strong> &nbsp;|&nbsp;
      Pred-R² (PRESS) = <strong>${rsm.predR2.toFixed(3)}</strong> &nbsp;|&nbsp;
      RMSE = <strong>${rsm.rmse.toFixed(2)} ${data.unit}</strong>
      ${opt ? `&nbsp;|&nbsp; <strong>Optimal (${opt.goal}):</strong> ${Object.entries(opt.optFactors).map(([k, v]) => `${k}=${v}`).join(", ")} → <strong>${opt.predicted} ${data.unit}</strong> (Desirability ${opt.desirability})` : ""}
    </div>
  `;

  // Populate axis selectors if needed
  const fxSel = el("doeFx"), fySel = el("doeFy");
  if (!rsm.factors.includes(state.fx)) state.fx = rsm.factors[0];
  if (!rsm.factors.includes(state.fy) || state.fy === state.fx) state.fy = rsm.factors.find((f) => f !== state.fx);
  fxSel.innerHTML = rsm.factors.map((f) => `<option value="${f}" ${f === state.fx ? "selected" : ""}>${f}</option>`).join("");
  fySel.innerHTML = rsm.factors.filter((f) => f !== state.fx).map((f) => `<option value="${f}" ${f === state.fy ? "selected" : ""}>${f}</option>`).join("");

  const third = rsm.factors.find((f) => f !== state.fx && f !== state.fy);
  const holdWrap = el("doeHoldWrap");
  if (third) {
    holdWrap.style.display = "block";
    const [lo, hi] = rsm.bounds[third];
    if (state.holdVal === null || state.holdVal < lo || state.holdVal > hi) state.holdVal = 0.5 * (lo + hi);
    el("doeHoldLabel").textContent = `Hold ${third}: ${state.holdVal.toFixed(2)}`;
    const slider = el("doeHoldSlider");
    slider.min = lo; slider.max = hi; slider.step = (hi - lo) / 40; slider.value = state.holdVal;
  } else {
    holdWrap.style.display = "none";
  }

  // Evaluate 2D grid
  const nG = 28;
  const [lox, hix] = rsm.bounds[state.fx];
  const [loy, hiy] = rsm.bounds[state.fy];
  const gx = Array.from({ length: nG }, (_, i) => lox + ((hix - lox) * i) / (nG - 1));
  const gy = Array.from({ length: nG }, (_, i) => loy + ((hiy - loy) * i) / (nG - 1));
  const Z = gy.map((yv) => gx.map((xv) => {
    const xc = rsm.factors.map((f) => {
      const [lo, hi] = rsm.bounds[f];
      const val = f === state.fx ? xv : f === state.fy ? yv : state.holdVal;
      return (val - 0.5 * (lo + hi)) / Math.max(0.5 * (hi - lo), 1e-9);
    });
    return +rsm.predictCoded(xc).toFixed(2);
  }));

  Plotly.newPlot("doeContour", [{
    x: gx, y: gy, z: Z, type: "contour", colorscale: "Viridis",
    contours: { coloring: "heatmap", showlabels: true },
    colorbar: { title: data.unit },
  }], {
    margin: { t: 24, r: 20, b: 48, l: 60 },
    xaxis: { title: state.fx },
    yaxis: { title: state.fy },
  }, { responsive: true });

  Plotly.newPlot("doeSurface", [{
    x: gx, y: gy, z: Z, type: "surface", colorscale: "Viridis", colorbar: { title: data.unit },
  }], {
    margin: { t: 10, r: 10, b: 10, l: 10 },
    scene: { xaxis: { title: state.fx }, yaxis: { title: state.fy }, zaxis: { title: data.ctq } },
  }, { responsive: true });

  el("doeTable").innerHTML = `<table><thead><tr><th>Model Term</th><th>Coded Coeff</th><th>Std Err</th><th>t-Stat</th><th>p-Value</th><th>Status</th></tr></thead><tbody>${
    rsm.summary.map((t) => `<tr><td><code>${t.term}</code></td><td>${t.coeff}</td><td>${t.se}</td><td>${t.tStat}</td><td>${t.pVal}</td><td><span class="pill ${t.sig ? "pill-sig" : "pill-ns"}">${t.sig ? "Significant" : "NS"}</span></td></tr>`).join("")
  }</tbody></table>`;
}

function renderAnomalies(runIds, anom) {
  if (!anom) return;
  const flagIdx = anom.t2Scores.map((s, i) => (s > anom.ucl ? i : -1)).filter((i) => i >= 0);

  const traces = [
    { x: runIds, y: anom.t2Scores, mode: "lines+markers", name: "Hotelling T²", line: { color: "#145374", width: 1.6 }, marker: { size: 4 } },
  ];
  if (flagIdx.length) {
    traces.push({
      x: flagIdx.map((i) => runIds[i]),
      y: flagIdx.map((i) => anom.t2Scores[i]),
      mode: "markers",
      name: "Multivariate Excursion",
      marker: { color: "#c0392b", size: 8, symbol: "diamond" },
    });
  }

  Plotly.newPlot("t2Chart", traces, {
    margin: { t: 24, r: 24, b: 42, l: 55 },
    xaxis: { title: "Run ID" },
    yaxis: { title: "Hotelling's T²" },
    legend: { orientation: "h", y: 1.12 },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: anom.ucl, y1: anom.ucl, line: { color: "#c0392b", dash: "dash", width: 2 } }],
    annotations: [{ xref: "paper", x: 1, y: anom.ucl, text: `Phase II UCL (${anom.ucl})`, showarrow: false, xanchor: "right", font: { color: "#c0392b", size: 11 } }],
  }, { responsive: true });

  const sortedDrv = Object.entries(anom.driverCounts).sort((a, b) => b[1] - a[1]);
  Plotly.newPlot("driverChart", [{
    x: sortedDrv.map((d) => d[0]),
    y: sortedDrv.map((d) => d[1]),
    type: "bar",
    marker: { color: "#2874a6" },
  }], {
    margin: { t: 18, r: 16, b: 60, l: 48 },
    yaxis: { title: "Flagged Runs" },
  }, { responsive: true });

  if (!anom.excursions.length) {
    el("excursionTable").innerHTML = `<p style="padding:12px;margin:0;color:#1e8449;font-weight:600;">No multivariate sensor excursions detected above UCL.</p>`;
  } else {
    el("excursionTable").innerHTML = `<table><thead><tr><th>Run ID</th><th>Hotelling T²</th><th>Primary Sensor Driver</th><th>Z-Score Attribution</th></tr></thead><tbody>${
      anom.excursions.map((e) => `<tr><td>#${e.runId}</td><td><strong>${e.t2}</strong></td><td><span class="pill pill-alert">${e.primary}</span></td><td>${e.attribution}</td></tr>`).join("")
    }</tbody></table>`;
  }
}

function render8D(data, imr, cap, anom, rsm, opt) {
  const md = A.build8DMarkdown(data.processName, data.ctq, data.unit, imr, cap, anom, rsm, opt);
  state.lastReportMd = md;

  // Lightweight safe Markdown -> HTML converter for clean display
  const html = md
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    .replace(/\n\n/g, "<br>");

  el("reportPreview").innerHTML = html;
}

// ------------------------------------------------------------------ Event Wiring
function init() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      el(btn.dataset.tab).classList.add("active");
      window.dispatchEvent(new Event("resize")); // trigger Plotly resize in newly shown tab
    });
  });

  // Preset selector
  el("processSelect").addEventListener("change", (e) => {
    state.mode = "preset";
    loadPresetSpecs(e.target.value);
    renderAll();
  });

  el("injectToggle").addEventListener("change", (e) => {
    state.injectAnomalies = e.target.checked;
    renderAll();
  });

  el("nRunsSlider").addEventListener("input", (e) => {
    state.nRuns = Number(e.target.value);
    el("nRunsVal").textContent = state.nRuns;
    renderAll();
  });

  el("subgroupSlider").addEventListener("input", (e) => {
    state.subgroupSize = Number(e.target.value);
    el("subgroupVal").textContent = state.subgroupSize;
    renderAll();
  });

  el("alphaSelect").addEventListener("change", (e) => {
    state.alpha = Number(e.target.value);
    renderAll();
  });

  ["lslInput", "targetInput", "uslInput"].forEach((id) => {
    el(id).addEventListener("change", () => {
      state.lsl = Number(el("lslInput").value);
      state.target = Number(el("targetInput").value);
      state.usl = Number(el("uslInput").value);
      renderAll();
    });
  });

  el("doeGoal").addEventListener("change", (e) => { state.optGoal = e.target.value; renderAll(); });
  el("doeFx").addEventListener("change", (e) => { state.fx = e.target.value; state.holdVal = null; renderAll(); });
  el("doeFy").addEventListener("change", (e) => { state.fy = e.target.value; state.holdVal = null; renderAll(); });
  el("doeHoldSlider").addEventListener("input", (e) => { state.holdVal = Number(e.target.value); renderAll(); });

  // CSV Upload
  el("csvUpload").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const parsed = parseCSV(ev.target.result);
      if (!parsed || parsed.numericCols.length < 1) {
        alert("CSV must have at least 10 rows and 1 numeric CTQ column.");
        return;
      }
      state.mode = "csv";
      state.customRows = parsed.rows;
      state.customName = file.name;
      state.customCtq = parsed.numericCols[parsed.numericCols.length - 1];
      state.customFactors = parsed.numericCols.filter((c) => c !== state.customCtq).slice(0, 4);
      const vals = parsed.rows.map((r) => r[state.customCtq]);
      const mu = vals.reduce((s, v) => s + v, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mu) ** 2, 0) / Math.max(vals.length - 1, 1));
      state.lsl = +(mu - 3 * sd).toFixed(2);
      state.target = +mu.toFixed(2);
      state.usl = +(mu + 3 * sd).toFixed(2);
      el("lslInput").value = state.lsl;
      el("targetInput").value = state.target;
      el("uslInput").value = state.usl;
      renderAll();
    };
    reader.readAsText(file);
  });

  // Downloads & Copy
  el("downloadMdBtn").addEventListener("click", () => {
    const blob = new Blob([state.lastReportMd], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `8D_Report_${state.processKey}.md`;
    a.click();
  });

  el("copyMdBtn").addEventListener("click", async () => {
    await navigator.clipboard.writeText(state.lastReportMd);
    el("copyMdBtn").textContent = "Copied!";
    setTimeout(() => (el("copyMdBtn").textContent = "Copy Markdown"), 1600);
  });

  el("downloadCsvBtn").addEventListener("click", () => {
    if (!state.lastRows.length) return;
    const cols = Object.keys(state.lastRows[0]);
    const csv = [cols.join(","), ...state.lastRows.map((r) => cols.map((c) => r[c]).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `production_data_${state.processKey}.csv`;
    a.click();
  });

  el("printBtn").addEventListener("click", () => window.print());

  loadPresetSpecs("cold_spray");
  renderAll();

  const initialTab = new URLSearchParams(window.location.search).get("tab");
  if (initialTab) {
    const btn = document.querySelector(`.tab-btn[data-tab="${initialTab}"]`);
    if (btn) btn.click();
  }
}

document.addEventListener("DOMContentLoaded", init);
