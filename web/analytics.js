/**
 * Pure client-side SPC, Process Capability, Response Surface DOE (RSM),
 * Multivariate Hotelling's T^2, and 8D Report Engine.
 * Zero backend required — deploys directly to GitHub Pages.
 */

// ------------------------------------------------------------------ Seeded PRNG
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randNormal(rng, mean = 0, std = 1) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + std * z;
}

// ------------------------------------------------------------------ Math helpers
function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function stdSample(arr) {
  const m = mean(arr);
  const ss = arr.reduce((s, v) => s + (v - m) ** 2, 0);
  return Math.sqrt(ss / Math.max(arr.length - 1, 1));
}

function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * ax);
  const y =
    1.0 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return sign * y;
}

function normCdf(x, mu = 0, sigma = 1) {
  return 0.5 * (1.0 + erf((x - mu) / (sigma * Math.SQRT2)));
}

function normPdf(x, mu = 0, sigma = 1) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Acklam inverse normal CDF approximation
function normInv(p) {
  const pp = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q, r;
  if (pp < plow) {
    q = Math.sqrt(-2 * Math.log(pp));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (pp <= phigh) {
    q = pp - 0.5;
    r = q * q;
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - pp));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// Matrix utilities for OLS & Hotelling's T^2
function matMul(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const out = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let t = 0; t < k; t++) {
      const ait = A[i][t];
      for (let j = 0; j < n; j++) out[i][j] += ait * B[t][j];
    }
  }
  return out;
}

function matTranspose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

function matInv(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > Math.abs(A[pivot][i])) pivot = r;
    }
    [A[i], A[pivot]] = [A[pivot], A[i]];
    const diag = A[i][i] || 1e-12;
    for (let j = 0; j < 2 * n; j++) A[i][j] /= diag;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = A[r][i];
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[i][j];
    }
  }
  return A.map((row) => row.slice(n));
}

// ------------------------------------------------------------------ Process Specs
const PROCESS_SPECS = {
  cold_spray: {
    key: "cold_spray",
    name: "Cold Spray Coating Deposition (SS316L on AISI 4140)",
    domain: "Coatings & Additive Repair",
    description:
      "High-pressure cold spray deposition runs tracking coating thickness (CTQ) across nozzle gas pressure, gas temperature, standoff distance, and traverse speed.",
    ctq: "Coating_Thickness_um",
    unit: "µm",
    lsl: 460,
    target: 500,
    usl: 540,
    factors: ["Gas_Pressure_MPa", "Gas_Temp_C", "Standoff_mm", "Traverse_Speed_mm_s"],
    bounds: {
      Gas_Pressure_MPa: [2.6, 3.4],
      Gas_Temp_C: [480, 560],
      Standoff_mm: [18, 32],
      Traverse_Speed_mm_s: [180, 260],
    },
    events: [
      "Runs 95–102: Powder feeder pressure drop (-0.42 MPa) causing low coating thickness.",
      "Runs 145–180: Gradual nozzle throat erosion drifting thickness upward.",
      "Runs 162–164: Gas heater controller spike (+55 °C) triggering a multivariate excursion.",
    ],
  },
  semiconductor_etch: {
    key: "semiconductor_etch",
    name: "Semiconductor Plasma Etch (SiO2 300 mm Wafer Fab)",
    domain: "Semiconductor Process Engineering",
    description:
      "300 mm wafer plasma etch lots tracking etch rate (CTQ) across RF power, chamber pressure, CF4/O2 gas ratio, and chuck temperature.",
    ctq: "Etch_Rate_nm_min",
    unit: "nm/min",
    lsl: 185,
    target: 200,
    usl: 215,
    factors: ["RF_Power_W", "Chamber_Pressure_mTorr", "Gas_Ratio_CF4_O2", "Chuck_Temp_C"],
    bounds: {
      RF_Power_W: [550, 650],
      Chamber_Pressure_mTorr: [35, 55],
      Gas_Ratio_CF4_O2: [2.5, 4.5],
      Chuck_Temp_C: [40, 60],
    },
    events: [
      "Lots 72–76: Chamber pressure valve stick (+11 mTorr) depressing etch rate.",
      "Lots 130–175: RF matching network drift causing upward etch-rate trend.",
      "Lots 152–154: Chuck chiller fault (+14 °C) triggering multivariate anomaly.",
    ],
  },
  lpbf_metal_am: {
    key: "lpbf_metal_am",
    name: "LPBF Metal Additive Manufacturing (Inconel 718)",
    domain: "Metal Additive & Precision Manufacturing",
    description:
      "Laser powder bed fusion builds tracking critical feature dimensional error (CTQ) across laser power, scan speed, hatch spacing, and preheat temperature.",
    ctq: "Dimensional_Error_um",
    unit: "µm",
    lsl: -40,
    target: 0,
    usl: 40,
    factors: ["Laser_Power_W", "Scan_Speed_mm_s", "Hatch_Spacing_um", "Preheat_Temp_C"],
    bounds: {
      Laser_Power_W: [240, 320],
      Scan_Speed_mm_s: [800, 1100],
      Hatch_Spacing_um: [90, 120],
      Preheat_Temp_C: [140, 200],
    },
    events: [
      "Builds 82–87: Protective window condensation reducing effective laser power (-38 W).",
      "Builds 140–180: Galvo thermal drift shifting dimensional error positive.",
      "Builds 160–162: Hatch/powder excursion triggering multivariate outlier.",
    ],
  },
};

function generateProductionData(key, nRuns = 180, injectAnomalies = true, seed = 42) {
  const rng = mulberry32(seed);
  const spec = PROCESS_SPECS[key];
  const rows = [];
  const shifts = ["Shift A", "Shift B", "Shift C"];

  for (let i = 0; i < nRuns; i++) {
    const runId = i + 1;
    const shift = shifts[i % 3];
    if (key === "cold_spray") {
      let p = randNormal(rng, 3.0, 0.06);
      let t = randNormal(rng, 520, 6.0);
      let s = randNormal(rng, 25.0, 0.7);
      let v = randNormal(rng, 220, 4.5);
      if (injectAnomalies) {
        if (runId >= 95 && runId <= 102) p -= 0.42;
        if (runId >= 145) {
          const d = (runId - 145) / Math.max(nRuns - 145, 1);
          s -= 2.2 * d;
          p += 0.14 * d;
        }
        if (runId >= 162 && runId <= 164) t += 55.0;
      }
      const ctq =
        500 + 52 * (p - 3.0) + 0.35 * (t - 520) - 5.5 * (s - 25) - 0.75 * (v - 220) - 18 * (p - 3.0) ** 2 + randNormal(rng, 0, 6.2);
      rows.push({
        Run_ID: runId,
        Shift: shift,
        Gas_Pressure_MPa: +p.toFixed(3),
        Gas_Temp_C: +t.toFixed(1),
        Standoff_mm: +s.toFixed(2),
        Traverse_Speed_mm_s: +v.toFixed(1),
        Coating_Thickness_um: +ctq.toFixed(2),
      });
    } else if (key === "semiconductor_etch") {
      let rf = randNormal(rng, 600, 7.0);
      let pr = randNormal(rng, 45, 1.2);
      let gr = randNormal(rng, 3.5, 0.12);
      let ct = randNormal(rng, 50, 1.1);
      if (injectAnomalies) {
        if (runId >= 72 && runId <= 76) pr += 11.0;
        if (runId >= 130) {
          const d = (runId - 130) / Math.max(nRuns - 130, 1);
          rf += 26.0 * d;
        }
        if (runId >= 152 && runId <= 154) ct += 14.0;
      }
      const ctq =
        200 + 0.34 * (rf - 600) - 1.15 * (pr - 45) + 6.5 * (gr - 3.5) + 0.45 * (ct - 50) + randNormal(rng, 0, 2.4);
      rows.push({
        Run_ID: runId,
        Shift: shift,
        RF_Power_W: +rf.toFixed(1),
        Chamber_Pressure_mTorr: +pr.toFixed(2),
        Gas_Ratio_CF4_O2: +gr.toFixed(3),
        Chuck_Temp_C: +ct.toFixed(2),
        Etch_Rate_nm_min: +ctq.toFixed(2),
      });
    } else {
      let pw = randNormal(rng, 280, 4.5);
      let sp = randNormal(rng, 950, 16.0);
      let ht = randNormal(rng, 105, 2.0);
      let ph = randNormal(rng, 170, 3.5);
      if (injectAnomalies) {
        if (runId >= 82 && runId <= 87) pw -= 38.0;
        if (runId >= 140) {
          const d = (runId - 140) / Math.max(nRuns - 140, 1);
          pw += 18.0 * d;
          ph += 12.0 * d;
        }
        if (runId >= 160 && runId <= 162) ht += 14.0;
      }
      const ctq =
        0 + 0.62 * (pw - 280) - 0.14 * (sp - 950) - 0.85 * (ht - 105) + 0.35 * (ph - 170) + randNormal(rng, 0, 6.0);
      rows.push({
        Run_ID: runId,
        Shift: shift,
        Laser_Power_W: +pw.toFixed(1),
        Scan_Speed_mm_s: +sp.toFixed(1),
        Hatch_Spacing_um: +ht.toFixed(2),
        Preheat_Temp_C: +ph.toFixed(1),
        Dimensional_Error_um: +ctq.toFixed(2),
      });
    }
  }
  return { rows, spec };
}

function generateDoeCCD(key, seed = 101) {
  const rng = mulberry32(seed);
  const spec = PROCESS_SPECS[key];
  const factors = spec.factors.slice(0, 3);
  const coded = [];
  for (const a of [-1, 1]) for (const b of [-1, 1]) for (const c of [-1, 1]) coded.push([a, b, c]);
  coded.push([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]);
  for (let i = 0; i < 6; i++) coded.push([0, 0, 0]);

  return coded.map((pt, idx) => {
    const [x1, x2, x3] = pt;
    const row = { Std_Order: idx + 1 };
    factors.forEach((f, j) => {
      const [lo, hi] = spec.bounds[f];
      const mid = 0.5 * (lo + hi);
      const half = 0.5 * (hi - lo);
      row[f] = +(mid + pt[j] * half).toFixed(3);
    });
    let y;
    if (key === "cold_spray") {
      y = 504 + 28 * x1 + 14.5 * x2 - 22 * x3 - 14 * x1 * x1 - 9.5 * x2 * x2 - 7 * x3 * x3 + 8.5 * x1 * x2 - 6 * x1 * x3 + randNormal(rng, 0, 3.2);
    } else if (key === "semiconductor_etch") {
      y = 201.5 + 12 * x1 - 9.5 * x2 + 6.2 * x3 - 5.8 * x1 * x1 - 4.5 * x2 * x2 - 3.2 * x3 * x3 - 4.2 * x1 * x2 + 2.8 * x1 * x3 + randNormal(rng, 0, 1.3);
    } else {
      y = -2 + 18 * x1 - 13.5 * x2 - 9 * x3 + 8.5 * x1 * x1 + 6 * x2 * x2 - 4 * x3 * x3 - 7.5 * x1 * x2 + 3.5 * x2 * x3 + randNormal(rng, 0, 2.4);
    }
    row[spec.ctq] = +y.toFixed(2);
    return row;
  });
}

// ------------------------------------------------------------------ SPC Engine
const CONSTANTS = {
  2: { A2: 1.88, D3: 0.0, D4: 3.267, d2: 1.128 },
  3: { A2: 1.023, D3: 0.0, D4: 2.574, d2: 1.693 },
  4: { A2: 0.729, D3: 0.0, D4: 2.282, d2: 2.059 },
  5: { A2: 0.577, D3: 0.0, D4: 2.114, d2: 2.326 },
  6: { A2: 0.483, D3: 0.0, D4: 2.004, d2: 2.534 },
  7: { A2: 0.419, D3: 0.076, D4: 1.924, d2: 2.704 },
  8: { A2: 0.373, D3: 0.136, D4: 1.864, d2: 2.847 },
  9: { A2: 0.337, D3: 0.184, D4: 1.816, d2: 2.97 },
  10: { A2: 0.308, D3: 0.223, D4: 1.777, d2: 3.078 },
};

function computeIMR(values) {
  const mr = [];
  for (let i = 1; i < values.length; i++) mr.push(Math.abs(values[i] - values[i - 1]));
  const xBar = mean(values);
  const mrBar = mean(mr);
  const sigmaWithin = mrBar / CONSTANTS[2].d2;
  const uclI = xBar + 3 * sigmaWithin;
  const lclI = xBar - 3 * sigmaWithin;
  const uclMR = CONSTANTS[2].D4 * mrBar;

  // Nelson Rules 1-4
  const z = values.map((v) => (v - xBar) / Math.max(sigmaWithin, 1e-9));
  const violations = [];

  const r1 = [];
  z.forEach((zv, i) => { if (Math.abs(zv) > 3) r1.push(i); });
  if (r1.length) violations.push({ rule: "Rule 1 (Beyond 3σ)", desc: "Single point outside ±3σ (special-cause excursion).", indices: r1 });

  const r2 = [];
  let runLen = 0, prevSign = 0;
  z.forEach((zv, i) => {
    const s = zv > 0 ? 1 : zv < 0 ? -1 : 0;
    if (s !== 0 && s === prevSign) runLen++;
    else { runLen = s !== 0 ? 1 : 0; prevSign = s; }
    if (runLen >= 9) r2.push(i);
  });
  if (r2.length) violations.push({ rule: "Rule 2 (9 Same Side)", desc: "9+ consecutive points on one side of centerline (mean shift).", indices: r2 });

  const r3 = [];
  let inc = 1, dec = 1;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[i - 1]) { inc++; dec = 1; }
    else if (values[i] < values[i - 1]) { dec++; inc = 1; }
    else { inc = 1; dec = 1; }
    if (inc >= 6 || dec >= 6) r3.push(i);
  }
  if (r3.length) violations.push({ rule: "Rule 3 (6 Trending)", desc: "6+ points steadily increasing or decreasing (wear / drift).", indices: r3 });

  const oocSet = Array.from(new Set(violations.flatMap((v) => v.indices))).sort((a, b) => a - b);
  return { values, mr, xBar, mrBar, sigmaWithin, uclI, lclI, uclMR, violations, oocIndices: oocSet };
}

function computeXbarR(values, n = 5) {
  const c = CONSTANTS[n] || CONSTANTS[5];
  const nGroups = Math.floor(values.length / n);
  const xBars = [], ranges = [], labels = [];
  for (let g = 0; g < nGroups; g++) {
    const slice = values.slice(g * n, (g + 1) * n);
    xBars.push(mean(slice));
    ranges.push(Math.max(...slice) - Math.min(...slice));
    labels.push(`SG-${g + 1}`);
  }
  const xDoubleBar = mean(xBars);
  const rBar = mean(ranges);
  const uclX = xDoubleBar + c.A2 * rBar;
  const lclX = xDoubleBar - c.A2 * rBar;
  const oocX = [];
  xBars.forEach((v, i) => { if (v > uclX || v < lclX) oocX.push(i); });
  return { labels, xBars, ranges, xDoubleBar, rBar, uclX, lclX, oocX };
}

function computeCapability(values, lsl, usl, target, sigmaWithin) {
  const n = values.length;
  const mu = mean(values);
  const stdOverall = Math.max(stdSample(values), 1e-9);
  const sw = Math.max(sigmaWithin, 1e-9);

  const cp = (usl - lsl) / (6 * sw);
  const cpk = Math.min((usl - mu) / (3 * sw), (mu - lsl) / (3 * sw));
  const pp = (usl - lsl) / (6 * stdOverall);
  const ppk = Math.min((usl - mu) / (3 * stdOverall), (mu - lsl) / (3 * stdOverall));

  const inSpec = values.filter((v) => v >= lsl && v <= usl).length;
  const obsYield = (100 * inSpec) / n;
  const pDef = Math.min(Math.max(normCdf(lsl, mu, stdOverall) + (1 - normCdf(usl, mu, stdOverall)), 1e-12), 1 - 1e-12);
  const expYield = 100 * (1 - pDef);
  const dpmo = pDef * 1e6;
  const sigmaLevel = Math.max(0, normInv(1 - pDef) + 1.5);

  // Jarque-Bera normality heuristic check (skewness & kurtosis)
  const m3 = values.reduce((s, v) => s + ((v - mu) / stdOverall) ** 3, 0) / n;
  const m4 = values.reduce((s, v) => s + ((v - mu) / stdOverall) ** 4, 0) / n;
  const jb = (n / 6) * (m3 * m3 + 0.25 * (m4 - 3) ** 2);
  const isNormal = jb < 5.99; // chi2(df=2, alpha=0.05)

  return { n, mean: mu, stdOverall, stdWithin: sw, lsl, usl, target, cp, cpk, pp, ppk, obsYield, expYield, dpmo, sigmaLevel, jb, isNormal };
}

// ------------------------------------------------------------------ RSM DOE Engine
function buildQuadRow(xc) {
  const k = xc.length;
  const row = [1];
  for (let i = 0; i < k; i++) row.push(xc[i]);
  for (let i = 0; i < k; i++) row.push(xc[i] * xc[i]);
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) row.push(xc[i] * xc[j]);
  return row;
}

function fitRSM(rows, factors, responseCol) {
  const k = factors.length;
  const bounds = {};
  factors.forEach((f) => {
    const vals = rows.map((r) => r[f]);
    bounds[f] = [Math.min(...vals), Math.max(...vals)];
  });

  const termNames = ["Intercept", ...factors, ...factors.map((f) => `${f}²`)];
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) termNames.push(`${factors[i]} × ${factors[j]}`);

  const X = rows.map((r) => {
    const xc = factors.map((f) => {
      const [lo, hi] = bounds[f];
      const mid = 0.5 * (lo + hi);
      const half = Math.max(0.5 * (hi - lo), 1e-9);
      return (r[f] - mid) / half;
    });
    return buildQuadRow(xc);
  });
  const y = rows.map((r) => r[responseCol]);
  const n = y.length;
  const p = X[0].length;

  const Xt = matTranspose(X);
  const XtX = matMul(Xt, X);
  // tiny ridge for singular safety on custom CSVs
  for (let i = 0; i < p; i++) XtX[i][i] += 1e-8;
  const XtXinv = matInv(XtX);
  const Xty = matMul(Xt, y.map((v) => [v]));
  const beta = matMul(XtXinv, Xty).map((r) => r[0]);

  const yHat = X.map((row) => row.reduce((s, v, j) => s + v * beta[j], 0));
  const resid = y.map((v, i) => v - yHat[i]);
  const sse = resid.reduce((s, e) => s + e * e, 0);
  const yMean = mean(y);
  const sst = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
  const dof = Math.max(n - p, 1);
  const mse = sse / dof;
  const rmse = Math.sqrt(mse);

  const r2 = sst > 0 ? 1 - sse / sst : 0;
  const adjR2 = 1 - ((1 - r2) * (n - 1)) / dof;

  // Exact PRESS using Hat-matrix diagonal h_ii
  let press = 0;
  for (let i = 0; i < n; i++) {
    let hii = 0;
    for (let a = 0; a < p; a++) {
      let s = 0;
      for (let b = 0; b < p; b++) s += XtXinv[a][b] * X[i][b];
      hii += X[i][a] * s;
    }
    const denom = Math.max(1 - hii, 1e-4);
    press += (resid[i] / denom) ** 2;
  }
  const predR2 = sst > 0 ? 1 - press / sst : 0;

  const summary = termNames.map((name, j) => {
    const se = Math.sqrt(Math.max(mse * XtXinv[j][j], 1e-12));
    const tStat = beta[j] / se;
    // Two-sided normal/t approximation
    const pVal = 2 * (1 - normCdf(Math.abs(tStat)));
    return { term: name, coeff: +beta[j].toFixed(3), se: +se.toFixed(3), tStat: +tStat.toFixed(2), pVal: +pVal.toFixed(4), sig: pVal < 0.05 };
  });

  function predictCoded(xc) {
    const row = buildQuadRow(xc);
    return row.reduce((s, v, j) => s + v * beta[j], 0);
  }

  function optimize(goal, target, lsl, usl) {
    // Dense grid + coordinate refinement in [-1, 1]^k
    const steps = k === 2 ? 25 : 13;
    const grid = Array.from({ length: steps }, (_, i) => -1 + (2 * i) / (steps - 1));
    let bestXc = new Array(k).fill(0);
    let bestScore = Infinity;

    const evalObj = (xc) => {
      const yh = predictCoded(xc);
      if (goal === "target") return (yh - target) ** 2;
      if (goal === "maximize") return -yh;
      return yh;
    };

    const recurse = (depth, curr) => {
      if (depth === k) {
        const sc = evalObj(curr);
        if (sc < bestScore) { bestScore = sc; bestXc = [...curr]; }
        return;
      }
      for (const v of grid) { curr[depth] = v; recurse(depth + 1, curr); }
    };
    recurse(0, new Array(k).fill(0));

    // Local coordinate polish
    for (let step = 0.08; step >= 0.005; step *= 0.5) {
      for (let d = 0; d < k; d++) {
        for (const dir of [-1, 1]) {
          const cand = [...bestXc];
          cand[d] = Math.min(1, Math.max(-1, cand[d] + dir * step));
          const sc = evalObj(cand);
          if (sc < bestScore) { bestScore = sc; bestXc = cand; }
        }
      }
    }

    const pred = predictCoded(bestXc);
    const optFactors = {};
    factors.forEach((f, j) => {
      const [lo, hi] = bounds[f];
      optFactors[f] = +(0.5 * (lo + hi) + bestXc[j] * 0.5 * (hi - lo)).toFixed(3);
    });

    let des = 0;
    if (goal === "target") {
      if (pred > lsl && pred < usl) des = pred <= target ? (pred - lsl) / Math.max(target - lsl, 1e-9) : (usl - pred) / Math.max(usl - target, 1e-9);
    } else if (goal === "maximize") {
      des = Math.min(1, Math.max(0, (pred - lsl) / Math.max(usl - lsl, 1e-9)));
    } else {
      des = Math.min(1, Math.max(0, (usl - pred) / Math.max(usl - lsl, 1e-9)));
    }
    return { goal, optFactors, predicted: +pred.toFixed(2), desirability: +des.toFixed(3) };
  }

  return { factors, bounds, summary, r2, adjR2, predR2, rmse, predictCoded, optimize };
}

// ------------------------------------------------------------------ Multivariate Hotelling's T^2
function detectHotellingAnomalies(rows, features, alpha = 0.01) {
  const n = rows.length;
  const p = features.length;
  const m = Math.max(p + 5, Math.floor(n * 0.45));
  const X = rows.map((r) => features.map((f) => r[f]));

  const mu = features.map((_, j) => mean(X.slice(0, m).map((row) => row[j])));
  const baseStd = features.map((_, j) => Math.max(stdSample(X.slice(0, m).map((row) => row[j])), 1e-9));

  const cov = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++) {
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        cov[a][b] += (X[i][a] - mu[a]) * (X[i][b] - mu[b]);
      }
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) cov[a][b] /= Math.max(m - 1, 1);
    cov[a][a] += 1e-6;
  }
  const covInv = matInv(cov);

  const t2Scores = X.map((row) => {
    const d = row.map((v, j) => v - mu[j]);
    let s = 0;
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) s += d[a] * covInv[a][b] * d[b];
    return s;
  });

  // Chi-square Wilson-Hilferty approximation for Phase II T^2 UCL
  const zAlpha = normInv(1 - alpha);
  const k = p;
  const chi2Crit = k * Math.pow(1 - 2 / (9 * k) + zAlpha * Math.sqrt(2 / (9 * k)), 3);
  const ucl = ((m + 1) * (m - 1) / (m * Math.max(m - p, 1))) * chi2Crit;

  const driverCounts = {};
  features.forEach((f) => (driverCounts[f] = 0));
  const excursions = [];

  t2Scores.forEach((score, i) => {
    if (score > ucl) {
      const zs = X[i].map((v, j) => (v - mu[j]) / baseStd[j]);
      const order = zs.map((zv, j) => ({ f: features[j], z: zv, az: Math.abs(zv) })).sort((a, b) => b.az - a.az);
      driverCounts[order[0].f]++;
      const drivers = order.slice(0, 2).filter((o) => o.az >= 1.8).map((o) => `${o.f} (${o.z >= 0 ? "+" : ""}${o.z.toFixed(1)}σ)`);
      excursions.push({
        runId: rows[i].Run_ID || i + 1,
        t2: +score.toFixed(2),
        primary: order[0].f,
        attribution: drivers.length ? drivers.join(", ") : `${order[0].f} (${order[0].z.toFixed(1)}σ)`,
      });
    }
  });

  return { t2Scores, ucl: +ucl.toFixed(2), excursions, driverCounts };
}

// ------------------------------------------------------------------ Automated 8D Report
function build8DMarkdown(processName, ctq, unit, imr, cap, anom, rsm, opt) {
  const status =
    cap.cpk >= 1.67 ? "EXCELLENT (Six Sigma capable)" :
    cap.cpk >= 1.33 ? "CAPABLE (Meets target ≥ 1.33)" :
    cap.cpk >= 1.0 ? "MARGINAL (Tighter centering/control needed)" :
    "INCAPABLE (Immediate containment required)";

  const centering = cap.cp - cap.cpk > 0.15
    ? `Off-centered process (Cp = ${cap.cp.toFixed(2)} vs Cpk = ${cap.cpk.toFixed(2)}). Re-centering mean (${cap.mean.toFixed(2)}) toward target (${cap.target}) can recover capability.`
    : `Process mean (${cap.mean.toFixed(2)} ${unit}) is well-centered near target (${cap.target} ${unit}).`;

  const stability = cap.cpk - cap.ppk > 0.15
    ? `Gap between Cpk (${cap.cpk.toFixed(2)}) and Ppk (${cap.ppk.toFixed(2)}) confirms time-dependent drift or special-cause shifts.`
    : `Short-term Cpk (${cap.cpk.toFixed(2)}) and long-term Ppk (${cap.ppk.toFixed(2)}) are aligned.`;

  const nelsonTxt = imr.violations.length
    ? imr.violations.map((v) => `- **${v.rule}**: ${v.indices.length} signal(s) — ${v.desc}`).join("\n")
    : "- Zero Nelson Rule 1–3 violations detected on I-MR chart.";

  const drivers = anom
    ? Object.entries(anom.driverCounts).filter(([, c]) => c > 0).sort((a, b) => b[1] - a[1]).map(([k, c]) => `**${k}** (${c} runs)`).join(", ")
    : "None";

  const sigTerms = rsm
    ? rsm.summary.filter((t) => t.term !== "Intercept" && t.sig).map((t) => `- \`${t.term}\`: coded effect = \`${t.coeff >= 0 ? "+" : ""}${t.coeff}\` (p = ${t.pVal})`).join("\n")
    : "- DOE not evaluated.";

  const setpoints = opt
    ? Object.entries(opt.optFactors).map(([k, v]) => `\`${k} = ${v}\``).join(", ")
    : "N/A";

  return `# 8D Root-Cause & Process Capability Report

**Process:** ${processName}
**Primary CTQ:** \`${ctq}\` (${unit}) | **Specs:** LSL = ${cap.lsl}, Target = ${cap.target}, USL = ${cap.usl}
**Overall Status:** **${status}**

## D1 — Team & Scope
- Scope: SPC (I-MR & X-bar/R), Process Capability (n = ${cap.n}), Multivariate Hotelling T² Diagnostics, and Response Surface DOE for \`${ctq}\`.
- Team: Process Engineer (Lead), Quality/Metrology Engineer, Equipment Maintenance, Production Supervisor.

## D2 — Problem Description (Quantified)
- **Mean ± Overall Std:** ${cap.mean.toFixed(2)} ± ${cap.stdOverall.toFixed(2)} ${unit}
- **Short-Term Capability (Cp / Cpk):** ${cap.cp.toFixed(2)} / ${cap.cpk.toFixed(2)} (Target ≥ 1.33)
- **Long-Term Performance (Pp / Ppk):** ${cap.pp.toFixed(2)} / ${cap.ppk.toFixed(2)}
- **Observed Yield / Expected Yield:** ${cap.obsYield.toFixed(2)}% / ${cap.expYield.toFixed(2)}% (${Math.round(cap.dpmo).toLocaleString()} DPMO, ${cap.sigmaLevel.toFixed(2)}σ)
- **Diagnostic:** ${centering} ${stability}

## D3 — Interim Containment Actions (ICA)
- Quarantine and 100% inspect parts from all **${imr.oocIndices.length}** SPC out-of-control run(s) and **${anom ? anom.excursions.length : 0}** multivariate excursion lot(s).
- Enforce temporary ±2.5σ alarm limits (${(imr.xBar - 2.5 * imr.sigmaWithin).toFixed(1)} to ${(imr.xBar + 2.5 * imr.sigmaWithin).toFixed(1)} ${unit}).

## D4 — Root-Cause Analysis (Statistical Evidence)
### SPC Signals
${nelsonTxt}

### Multivariate Sensor Attribution
- **Flagged Runs:** ${anom ? anom.excursions.length : 0} (Hotelling T² UCL = ${anom ? anom.ucl : "N/A"})
- **Top Driving Sensors:** ${drivers || "None"}

### Significant RSM DOE Factors
${rsm ? `Model Fit: R² = ${rsm.r2.toFixed(3)}, Adj-R² = ${rsm.adjR2.toFixed(3)}, Pred-R² (PRESS) = ${rsm.predR2.toFixed(3)}` : ""}
${sigTerms}

## D5 — Permanent Corrective Actions (PCA)
- **Recommended Optimal Setpoints:** ${setpoints}${opt ? ` → Predicted \`${opt.predicted} ${unit}\` (Desirability ${opt.desirability})` : ""}
- Calibrate and add preventive maintenance / interlocks on top driving sensor(s): ${drivers || "N/A"}.

## D6 — Verification & Validation
- Execute a 30-run confirmation build at D5 setpoints and confirm zero Nelson Rule violations with Cpk ≥ 1.33 and Ppk ≥ 1.33.

## D7 — Prevent Recurrence
- Enable live Hotelling's T² multivariate sensor monitoring with automated pre-excursion alarms and update shop-floor SOPs.

## D8 — Closure
- Archive validated RSM coefficients, control limits, and 8D report in the production quality record.
`;
}

window.SPCAnalytics = {
  PROCESS_SPECS,
  generateProductionData,
  generateDoeCCD,
  computeIMR,
  computeXbarR,
  computeCapability,
  fitRSM,
  detectHotellingAnomalies,
  build8DMarkdown,
  normPdf,
};
