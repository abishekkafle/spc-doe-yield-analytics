"""Realistic physics-inspired manufacturing process dataset generators.

Provides three domain-authentic datasets (with known ground-truth drift,
shift, and excursion events) plus a clean CSV ingestion helper:
1. Cold Spray Metal Coating Deposition (Coating Thickness, Porosity, Hardness)
2. Semiconductor Thin-Film Plasma Etch (Etch Rate, Non-Uniformity, Defect Density)
3. Metal Additive Manufacturing - LPBF (Dimensional Error, Relative Density, Ra)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessDatasetSpec:
    """Metadata and engineering specifications for a manufacturing process."""

    key: str
    name: str
    domain: str
    description: str
    primary_ctq: str
    ctq_unit: str
    lsl: float
    usl: float
    target: float
    process_factors: List[str]
    factor_units: Dict[str, str]
    factor_bounds: Dict[str, Tuple[float, float]]
    secondary_ctqs: List[str]
    ground_truth_events: List[str]


PROCESS_SPECS: Dict[str, ProcessDatasetSpec] = {
    "cold_spray": ProcessDatasetSpec(
        key="cold_spray",
        name="Cold Spray Coating Deposition (SS316L on AISI 4140)",
        domain="Coatings & Additive Repair",
        description=(
            "High-pressure cold spray deposition runs tracking coating thickness (CTQ), "
            "porosity, and Vickers microhardness across nozzle gas pressure, gas temperature, "
            "standoff distance, and traverse speed. Includes a nozzle-wear drift and a "
            "powder-feeder pressure excursion."
        ),
        primary_ctq="Coating_Thickness_um",
        ctq_unit="µm",
        lsl=460.0,
        usl=540.0,
        target=500.0,
        process_factors=[
            "Gas_Pressure_MPa",
            "Gas_Temp_C",
            "Standoff_mm",
            "Traverse_Speed_mm_s",
        ],
        factor_units={
            "Gas_Pressure_MPa": "MPa",
            "Gas_Temp_C": "°C",
            "Standoff_mm": "mm",
            "Traverse_Speed_mm_s": "mm/s",
        },
        factor_bounds={
            "Gas_Pressure_MPa": (2.6, 3.4),
            "Gas_Temp_C": (480.0, 560.0),
            "Standoff_mm": (18.0, 32.0),
            "Traverse_Speed_mm_s": (180.0, 260.0),
        },
        secondary_ctqs=["Porosity_pct", "Hardness_HV"],
        ground_truth_events=[
            "Runs 95–102: Powder feeder pressure drop (-0.42 MPa) causing low coating thickness.",
            "Runs 145–180: Gradual nozzle throat erosion shifting standoff/velocity and drifting thickness upward.",
            "Runs 162–164: Gas heater controller spike (+55 °C) causing simultaneous multivariate excursion.",
        ],
    ),
    "semiconductor_etch": ProcessDatasetSpec(
        key="semiconductor_etch",
        name="Semiconductor Plasma Etch (SiO2 Wafer Fab)",
        domain="Semiconductor Process Engineering",
        description=(
            "300 mm wafer plasma etch lots tracking etch rate (CTQ), within-wafer non-uniformity, "
            "and particle defect density across RF power, chamber pressure, CF4/O2 gas ratio, "
            "and chuck temperature. Includes RF matching drift and chamber leak excursions."
        ),
        primary_ctq="Etch_Rate_nm_min",
        ctq_unit="nm/min",
        lsl=185.0,
        usl=215.0,
        target=200.0,
        process_factors=[
            "RF_Power_W",
            "Chamber_Pressure_mTorr",
            "Gas_Ratio_CF4_O2",
            "Chuck_Temp_C",
        ],
        factor_units={
            "RF_Power_W": "W",
            "Chamber_Pressure_mTorr": "mTorr",
            "Gas_Ratio_CF4_O2": "ratio",
            "Chuck_Temp_C": "°C",
        },
        factor_bounds={
            "RF_Power_W": (550.0, 650.0),
            "Chamber_Pressure_mTorr": (35.0, 55.0),
            "Gas_Ratio_CF4_O2": (2.5, 4.5),
            "Chuck_Temp_C": (40.0, 60.0),
        },
        secondary_ctqs=["Non_Uniformity_pct", "Defect_Density_cm2"],
        ground_truth_events=[
            "Lots 72–76: Chamber pressure valve stick (+11 mTorr) depressing etch rate.",
            "Lots 130–175: RF matching network drift causing gradual upward etch-rate trend.",
            "Lots 152–154: Chuck chiller fault (+14 °C) triggering multivariate anomaly.",
        ],
    ),
    "lpbf_metal_am": ProcessDatasetSpec(
        key="lpbf_metal_am",
        name="LPBF Metal Additive Manufacturing (Inconel 718)",
        domain="Metal Additive & Precision Manufacturing",
        description=(
            "Laser powder bed fusion builds tracking critical feature dimensional error (CTQ), "
            "relative density, and surface roughness (Ra) across laser power, scan speed, "
            "hatch spacing, and preheat temperature. Includes recoater-blade wear and laser "
            "optics fouling."
        ),
        primary_ctq="Dimensional_Error_um",
        ctq_unit="µm",
        lsl=-40.0,
        usl=40.0,
        target=0.0,
        process_factors=[
            "Laser_Power_W",
            "Scan_Speed_mm_s",
            "Hatch_Spacing_um",
            "Preheat_Temp_C",
        ],
        factor_units={
            "Laser_Power_W": "W",
            "Scan_Speed_mm_s": "mm/s",
            "Hatch_Spacing_um": "µm",
            "Preheat_Temp_C": "°C",
        },
        factor_bounds={
            "Laser_Power_W": (240.0, 320.0),
            "Scan_Speed_mm_s": (800.0, 1100.0),
            "Hatch_Spacing_um": (90.0, 120.0),
            "Preheat_Temp_C": (140.0, 200.0),
        },
        secondary_ctqs=["Relative_Density_pct", "Roughness_Ra_um"],
        ground_truth_events=[
            "Builds 82–87: Protective window condensation reducing effective laser power (-38 W).",
            "Builds 140–180: Galvo thermal drift shifting dimensional error positive.",
            "Builds 160–162: Powder moisture spike causing density drop and high Ra.",
        ],
    ),
}


def generate_production_data(
    process_key: str = "cold_spray",
    n_runs: int = 180,
    seed: int = 42,
    inject_anomalies: bool = True,
) -> Tuple[pd.DataFrame, ProcessDatasetSpec]:
    """Generate time-ordered production SPC + multivariate sensor data."""
    if process_key not in PROCESS_SPECS:
        raise ValueError(f"Unknown process_key '{process_key}'. Valid: {list(PROCESS_SPECS)}")

    spec = PROCESS_SPECS[process_key]
    rng = np.random.default_rng(seed)
    run_ids = np.arange(1, n_runs + 1)
    timestamps = pd.date_range("2026-09-01 08:00", periods=n_runs, freq="2h")
    shifts = np.where((run_ids - 1) % 3 == 0, "Shift A", np.where((run_ids - 1) % 3 == 1, "Shift B", "Shift C"))
    subgroups = ((run_ids - 1) // 5) + 1

    if process_key == "cold_spray":
        pressure = rng.normal(3.0, 0.06, n_runs)
        temp = rng.normal(520.0, 6.0, n_runs)
        standoff = rng.normal(25.0, 0.7, n_runs)
        speed = rng.normal(220.0, 4.5, n_runs)

        if inject_anomalies:
            # Event 1: Powder feeder / pressure drop
            idx1 = (run_ids >= 95) & (run_ids <= 102)
            pressure[idx1] -= 0.42
            # Event 2: Gradual nozzle wear drift
            idx2 = run_ids >= 145
            drift = np.linspace(0.0, 1.0, idx2.sum())
            standoff[idx2] -= 2.2 * drift
            pressure[idx2] += 0.14 * drift
            # Event 3: Gas heater spike
            idx3 = (run_ids >= 162) & (run_ids <= 164)
            temp[idx3] += 55.0

        # Physics-inspired response equations
        thickness = (
            500.0
            + 52.0 * (pressure - 3.0)
            + 0.35 * (temp - 520.0)
            - 5.5 * (standoff - 25.0)
            - 0.75 * (speed - 220.0)
            - 18.0 * (pressure - 3.0) ** 2
            + rng.normal(0.0, 6.2, n_runs)
        )
        porosity = np.clip(
            1.65
            - 1.3 * (pressure - 3.0)
            - 0.008 * (temp - 520.0)
            + 0.09 * (standoff - 25.0)
            + rng.normal(0.0, 0.18, n_runs),
            0.2,
            8.0,
        )
        hardness = (
            285.0
            + 28.0 * (pressure - 3.0)
            + 0.15 * (temp - 520.0)
            - 1.8 * (standoff - 25.0)
            + rng.normal(0.0, 4.5, n_runs)
        )

        df = pd.DataFrame(
            {
                "Run_ID": run_ids,
                "Timestamp": timestamps,
                "Subgroup": subgroups,
                "Shift": shifts,
                "Gas_Pressure_MPa": np.round(pressure, 3),
                "Gas_Temp_C": np.round(temp, 1),
                "Standoff_mm": np.round(standoff, 2),
                "Traverse_Speed_mm_s": np.round(speed, 1),
                "Coating_Thickness_um": np.round(thickness, 2),
                "Porosity_pct": np.round(porosity, 3),
                "Hardness_HV": np.round(hardness, 1),
            }
        )

    elif process_key == "semiconductor_etch":
        rf_power = rng.normal(600.0, 7.0, n_runs)
        pressure = rng.normal(45.0, 1.2, n_runs)
        gas_ratio = rng.normal(3.5, 0.12, n_runs)
        chuck_temp = rng.normal(50.0, 1.1, n_runs)

        if inject_anomalies:
            idx1 = (run_ids >= 72) & (run_ids <= 76)
            pressure[idx1] += 11.0
            idx2 = run_ids >= 130
            drift = np.linspace(0.0, 1.0, idx2.sum())
            rf_power[idx2] += 26.0 * drift
            idx3 = (run_ids >= 152) & (run_ids <= 154)
            chuck_temp[idx3] += 14.0

        etch_rate = (
            200.0
            + 0.34 * (rf_power - 600.0)
            - 1.15 * (pressure - 45.0)
            + 6.5 * (gas_ratio - 3.5)
            + 0.45 * (chuck_temp - 50.0)
            + rng.normal(0.0, 2.4, n_runs)
        )
        non_unif = np.clip(
            2.1
            + 0.045 * np.abs(pressure - 45.0)
            + 0.015 * np.abs(rf_power - 600.0)
            + 0.08 * np.abs(chuck_temp - 50.0)
            + rng.normal(0.0, 0.15, n_runs),
            0.5,
            9.0,
        )
        defects = np.clip(
            0.12
            + 0.015 * np.maximum(0.0, pressure - 46.0)
            + 0.012 * np.maximum(0.0, chuck_temp - 52.0)
            + rng.normal(0.0, 0.02, n_runs),
            0.02,
            2.5,
        )

        df = pd.DataFrame(
            {
                "Run_ID": run_ids,
                "Timestamp": timestamps,
                "Subgroup": subgroups,
                "Shift": shifts,
                "RF_Power_W": np.round(rf_power, 1),
                "Chamber_Pressure_mTorr": np.round(pressure, 2),
                "Gas_Ratio_CF4_O2": np.round(gas_ratio, 3),
                "Chuck_Temp_C": np.round(chuck_temp, 2),
                "Etch_Rate_nm_min": np.round(etch_rate, 2),
                "Non_Uniformity_pct": np.round(non_unif, 3),
                "Defect_Density_cm2": np.round(defects, 3),
            }
        )

    else:  # lpbf_metal_am
        power = rng.normal(280.0, 4.5, n_runs)
        scan_speed = rng.normal(950.0, 16.0, n_runs)
        hatch = rng.normal(105.0, 2.0, n_runs)
        preheat = rng.normal(170.0, 3.5, n_runs)

        if inject_anomalies:
            idx1 = (run_ids >= 82) & (run_ids <= 87)
            power[idx1] -= 38.0
            idx2 = run_ids >= 140
            drift = np.linspace(0.0, 1.0, idx2.sum())
            power[idx2] += 18.0 * drift
            preheat[idx2] += 12.0 * drift
            idx3 = (run_ids >= 160) & (run_ids <= 162)
            hatch[idx3] += 14.0

        dim_error = (
            0.0
            + 0.62 * (power - 280.0)
            - 0.14 * (scan_speed - 950.0)
            - 0.85 * (hatch - 105.0)
            + 0.35 * (preheat - 170.0)
            + rng.normal(0.0, 6.0, n_runs)
        )
        rel_density = np.clip(
            99.45
            + 0.012 * (power - 280.0)
            - 0.003 * (scan_speed - 950.0)
            - 0.035 * np.maximum(0.0, hatch - 105.0)
            - 0.0005 * (power - 280.0) ** 2
            + rng.normal(0.0, 0.11, n_runs),
            95.0,
            99.95,
        )
        roughness = np.clip(
            7.2
            - 0.025 * (power - 280.0)
            + 0.008 * (scan_speed - 950.0)
            + 0.09 * (hatch - 105.0)
            + rng.normal(0.0, 0.45, n_runs),
            3.5,
            18.0,
        )

        df = pd.DataFrame(
            {
                "Run_ID": run_ids,
                "Timestamp": timestamps,
                "Subgroup": subgroups,
                "Shift": shifts,
                "Laser_Power_W": np.round(power, 1),
                "Scan_Speed_mm_s": np.round(scan_speed, 1),
                "Hatch_Spacing_um": np.round(hatch, 2),
                "Preheat_Temp_C": np.round(preheat, 1),
                "Dimensional_Error_um": np.round(dim_error, 2),
                "Relative_Density_pct": np.round(rel_density, 3),
                "Roughness_Ra_um": np.round(roughness, 2),
            }
        )

    return df, spec


def generate_doe_ccd_data(
    process_key: str = "cold_spray",
    seed: int = 101,
) -> Tuple[pd.DataFrame, ProcessDatasetSpec]:
    """Generate a realistic Central Composite Design (CCD) dataset for RSM.

    Uses a 3-factor face-centered CCD (8 factorial + 6 axial + 6 center points = 20 runs)
    with true quadratic curvature and two-factor interaction so RSM & optimization
    produce authentic engineering surfaces.
    """
    if process_key not in PROCESS_SPECS:
        raise ValueError(f"Unknown process_key '{process_key}'.")

    spec = PROCESS_SPECS[process_key]
    rng = np.random.default_rng(seed)
    factors = spec.process_factors[:3]

    # Build face-centered CCD coded matrix [-1, 0, +1]
    factorial = np.array(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    )
    axial = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.zeros((6, 3))
    coded = np.vstack([factorial, axial, center])
    n = len(coded)

    # Convert coded -> natural engineering units
    data: Dict[str, np.ndarray] = {"Std_Order": np.arange(1, n + 1)}
    for j, f in enumerate(factors):
        lo, hi = spec.factor_bounds[f]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        data[f] = np.round(mid + coded[:, j] * half, 3)

    x1, x2, x3 = coded[:, 0], coded[:, 1], coded[:, 2]

    if process_key == "cold_spray":
        # Coating thickness with realistic optimum near high pressure, moderate temp, low standoff
        y = (
            504.0
            + 28.0 * x1
            + 14.5 * x2
            - 22.0 * x3
            - 14.0 * x1**2
            - 9.5 * x2**2
            - 7.0 * x3**2
            + 8.5 * x1 * x2
            - 6.0 * x1 * x3
            + rng.normal(0.0, 3.2, n)
        )
    elif process_key == "semiconductor_etch":
        y = (
            201.5
            + 12.0 * x1
            - 9.5 * x2
            + 6.2 * x3
            - 5.8 * x1**2
            - 4.5 * x2**2
            - 3.2 * x3**2
            - 4.2 * x1 * x2
            + 2.8 * x1 * x3
            + rng.normal(0.0, 1.3, n)
        )
    else:
        y = (
            -2.0
            + 18.0 * x1
            - 13.5 * x2
            - 9.0 * x3
            + 8.5 * x1**2
            + 6.0 * x2**2
            - 4.0 * x3**2
            - 7.5 * x1 * x2
            + 3.5 * x2 * x3
            + rng.normal(0.0, 2.4, n)
        )

    data[spec.primary_ctq] = np.round(y, 2)
    return pd.DataFrame(data), spec
