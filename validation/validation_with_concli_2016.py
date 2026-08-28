"""Validation setup based on Concli (2016).

Reference:
    F. Concli, "Pressure distribution in small hydrodynamic journal bearings
    considering cavitation: a numerical approach based on the open-source CFD
    code OpenFOAM", Lubrication Science 28(6), 329-347, 2016.

The paper studies a small precision-gearbox journal bearing using CFD cavitation
models and compares them with Sommerfeld / half-Sommerfeld style calculations.
Most numerical results are published as plots rather than tables. This script
therefore reproduces the published bearing setup with ROSS FluidFlow and checks
the qualitative trends reported in the paper:

* maximum pressure increases with rotational speed,
* maximum pressure increases with eccentricity ratio,
* hydraulic force increases with eccentricity ratio,
* the pressure peak shifts along the convergent gap as operating conditions vary.

The CSV outputs can be compared with digitized versions of Figures 6-8 and 13
if numerical chart data becomes available later.
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent


def bootstrap_bearing_modules() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))

    ross_pkg = types.ModuleType("ross")
    ross_pkg.__path__ = [str(PROJECT_ROOT / "ross")]
    sys.modules.setdefault("ross", ross_pkg)

    bearings_pkg = types.ModuleType("ross.bearings")
    bearings_pkg.__path__ = [str(PROJECT_ROOT / "ross" / "bearings")]
    sys.modules.setdefault("ross.bearings", bearings_pkg)

    graphics_stub = types.ModuleType("ross.bearings.fluid_flow_graphics")
    graphics_stub.plot_pressure_surface = lambda *args, **kwargs: None
    sys.modules.setdefault("ross.bearings.fluid_flow_graphics", graphics_stub)


bootstrap_bearing_modules()

from ross.bearings import fluid_flow as flow
from ross.bearings.fluid_flow_coefficients import calculate_oil_film_force


@dataclass(frozen=True)
class Concli2016Bearing:
    journal_radius: float = 2e-3
    bearing_radius: float = 2.005e-3
    length: float = 13e-3
    density: float = 1060.0
    kinematic_viscosity_40c: float = 220e-6
    kinematic_viscosity_100c: float = 40e-6
    saturation_pressure: float = 20e3

    @property
    def radial_clearance(self) -> float:
        return self.bearing_radius - self.journal_radius

    @property
    def journal_diameter(self) -> float:
        return 2 * self.journal_radius

    @property
    def l_over_d(self) -> float:
        return self.length / self.journal_diameter

    def dynamic_viscosity(self, temperature: str) -> float:
        if temperature == "40C":
            return self.density * self.kinematic_viscosity_40c
        if temperature == "100C":
            return self.density * self.kinematic_viscosity_100c
        raise ValueError(f"Unsupported temperature label: {temperature}")


def make_fluid_flow(
    bearing: Concli2016Bearing,
    omega: float,
    viscosity: float,
    eccentricity_ratio: float,
) -> flow.FluidFlow:
    return flow.FluidFlow(
        nz=16,
        ntheta=91,
        length=bearing.length,
        omega=omega,
        p_in=0.0,
        p_out=0.0,
        radius_rotor=bearing.journal_radius,
        radius_stator=bearing.bearing_radius,
        viscosity=viscosity,
        density=bearing.density,
        eccentricity=eccentricity_ratio * bearing.radial_clearance,
        attitude_angle=0.0,
        immediately_calculate_pressure_matrix_numerically=True,
    )


def positive_pressure_profile(ff: flow.FluidFlow) -> tuple[np.ndarray, np.ndarray]:
    """Return mid-plane positive-pressure profile using the paper-like theta axis."""
    mid_upper = ff.nz // 2
    mid_lower = mid_upper - 1
    pressure = 0.5 * (ff.p_mat_numerical[mid_lower] + ff.p_mat_numerical[mid_upper])
    gamma = ff.gama[0] % (2 * np.pi)
    positive = pressure > 1e-9
    # FluidFlow starts the positive half film at pi/2 for attitude_angle=0.
    theta = gamma[positive] - np.pi / 2
    order = np.argsort(theta)
    return theta[order], pressure[positive][order]


def evaluate_case(
    bearing: Concli2016Bearing,
    omega: float,
    eccentricity_ratio: float,
    temperature: str,
) -> dict[str, float | str]:
    viscosity = bearing.dynamic_viscosity(temperature)
    ff = make_fluid_flow(bearing, omega, viscosity, eccentricity_ratio)
    radial, tangential, fx, fy = calculate_oil_film_force(ff, force_type="numerical")
    theta, pressure = positive_pressure_profile(ff)
    peak_index = int(np.argmax(pressure))
    force_angle = float(np.degrees(np.arctan2(tangential, radial)))

    return {
        "temperature": temperature,
        "omega_rad_s": omega,
        "eccentricity_ratio": eccentricity_ratio,
        "dynamic_viscosity_Pa_s": viscosity,
        "radial_force_N": radial,
        "tangential_force_N": tangential,
        "fx_N": fx,
        "fy_N": fy,
        "resultant_force_N": float(np.hypot(fx, fy)),
        "force_angle_from_radial_deg": force_angle,
        "max_pressure_Pa": float(np.max(pressure)),
        "peak_theta_deg": float(np.degrees(theta[peak_index])),
        "positive_arc_start_deg": float(np.degrees(theta[0])),
        "positive_arc_end_deg": float(np.degrees(theta[-1])),
        "positive_arc_width_deg": float(np.degrees(theta[-1] - theta[0])),
    }


def sweep_cases(
    bearing: Concli2016Bearing,
    temperature: str,
) -> list[dict[str, float | str]]:
    rows = []
    for omega in [328.0, 1500.0, 3000.0]:
        for eps in [0.2, 0.4, 0.6, 0.8]:
            rows.append(evaluate_case(bearing, omega, eps, temperature))
    return rows


def figure_13_cases(
    bearing: Concli2016Bearing,
    temperature: str,
) -> list[dict[str, float | str]]:
    """Cases named in the Figure 13 caption."""
    cases = [
        ("speed_effect_a", 328.0, 0.6),
        ("speed_effect_b", 1500.0, 0.6),
        ("speed_effect_c", 3000.0, 0.6),
        ("eccentricity_effect_d", 328.0, 0.2),
        ("eccentricity_effect_e", 328.0, 0.4),
        ("eccentricity_effect_f", 328.0, 0.6),
    ]
    rows = []
    for case_id, omega, eps in cases:
        row = evaluate_case(bearing, omega, eps, temperature)
        row["figure_13_case"] = case_id
        rows.append(row)
    return rows


def approximate_article_force_references() -> list[dict[str, float | str]]:
    """Approximate force values read from Concli 2016 Table V / Fig. 6f.

    The article prints force data for omega = 328 rad/s. The transcription of
    Table V is imperfect, but the following values are readable as the
    half-Sommerfeld and cavitation-model resultants. They are intended as
    graphical/table-read reference values, not exact machine-readable data.
    """
    rows = []
    # Approximate resultant force F [N] for omega = 328 rad/s.
    # The two cavitation models are close; values follow Table V rows.
    data = {
        0.2: {
            "half_sommerfeld_force_N": 2378.0,
            "schnerr_sauer_force_N": 3542.0,
            "kunz_force_N": 3498.0,
            "half_sommerfeld_angle_deg": 97.0,
            "schnerr_sauer_angle_deg": 96.0,
            "kunz_angle_deg": 97.0,
        },
        0.4: {
            "half_sommerfeld_force_N": 4921.0,
            "schnerr_sauer_force_N": 7041.0,
            "kunz_force_N": 9485.0,
            "half_sommerfeld_angle_deg": 105.0,
            "schnerr_sauer_angle_deg": 103.0,
            "kunz_angle_deg": np.nan,
        },
        0.6: {
            "half_sommerfeld_force_N": 8170.0,
            "schnerr_sauer_force_N": 10658.0,
            "kunz_force_N": 14767.0,
            "half_sommerfeld_angle_deg": 115.0,
            "schnerr_sauer_angle_deg": 111.0,
            "kunz_angle_deg": np.nan,
        },
        0.8: {
            "half_sommerfeld_force_N": 14820.0,
            "schnerr_sauer_force_N": 15128.0,
            "kunz_force_N": 14902.0,
            "half_sommerfeld_angle_deg": 130.0,
            "schnerr_sauer_angle_deg": 122.0,
            "kunz_angle_deg": 123.0,
        },
    }
    for eps, values in data.items():
        row: dict[str, float | str] = {
            "source": "Concli 2016 Table V / Fig. 6f approximate read",
            "omega_rad_s": 328.0,
            "eccentricity_ratio": eps,
        }
        row.update(values)
        rows.append(row)
    return rows


def compare_with_article_references(
    sweep_rows: list[dict[str, float | str]],
    references: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    comparisons = []
    for reference in references:
        omega = reference["omega_rad_s"]
        eps = reference["eccentricity_ratio"]
        match = next(
            row
            for row in sweep_rows
            if row["omega_rad_s"] == omega and row["eccentricity_ratio"] == eps
        )
        fluidflow_force = float(match["resultant_force_N"])
        half_force = float(reference["half_sommerfeld_force_N"])
        ss_force = float(reference["schnerr_sauer_force_N"])
        kunz_force = float(reference["kunz_force_N"])
        comparisons.append(
            {
                "omega_rad_s": omega,
                "eccentricity_ratio": eps,
                "fluidflow_force_N": fluidflow_force,
                "article_half_sommerfeld_force_N": half_force,
                "article_schnerr_sauer_force_N": ss_force,
                "article_kunz_force_N": kunz_force,
                "fluidflow_vs_article_half_rel_error": (
                    fluidflow_force - half_force
                )
                / half_force,
                "fluidflow_vs_article_schnerr_sauer_rel_error": (
                    fluidflow_force - ss_force
                )
                / ss_force,
                "fluidflow_vs_article_kunz_rel_error": (
                    fluidflow_force - kunz_force
                )
                / kunz_force,
                "fluidflow_force_angle_deg": match["force_angle_from_radial_deg"],
                "article_half_sommerfeld_angle_deg": reference[
                    "half_sommerfeld_angle_deg"
                ],
                "article_schnerr_sauer_angle_deg": reference[
                    "schnerr_sauer_angle_deg"
                ],
            }
        )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_monotonic_trends(rows: list[dict[str, float | str]]) -> list[str]:
    messages = []

    for omega in sorted({row["omega_rad_s"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["omega_rad_s"] == omega],
            key=lambda row: row["eccentricity_ratio"],
        )
        forces = np.array([row["resultant_force_N"] for row in subset], dtype=float)
        pressures = np.array([row["max_pressure_Pa"] for row in subset], dtype=float)
        force_ok = bool(np.all(np.diff(forces) > 0))
        pressure_ok = bool(np.all(np.diff(pressures) > 0))
        messages.append(
            f"force increases with eccentricity at omega={omega:g}: "
            f"{'PASS' if force_ok else 'FAIL'}"
        )
        messages.append(
            f"max pressure increases with eccentricity at omega={omega:g}: "
            f"{'PASS' if pressure_ok else 'FAIL'}"
        )

    for eps in sorted({row["eccentricity_ratio"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["eccentricity_ratio"] == eps],
            key=lambda row: row["omega_rad_s"],
        )
        forces = np.array([row["resultant_force_N"] for row in subset], dtype=float)
        pressures = np.array([row["max_pressure_Pa"] for row in subset], dtype=float)
        force_ok = bool(np.all(np.diff(forces) > 0))
        pressure_ok = bool(np.all(np.diff(pressures) > 0))
        messages.append(
            f"force increases with speed at epsilon={eps:g}: "
            f"{'PASS' if force_ok else 'FAIL'}"
        )
        messages.append(
            f"max pressure increases with speed at epsilon={eps:g}: "
            f"{'PASS' if pressure_ok else 'FAIL'}"
        )

    return messages


def print_report(rows: list[dict[str, float | str]]) -> None:
    print("\nConcli 2016 FluidFlow sweep")
    print(
        f"{'omega':>8} {'eps':>7} {'F_res[N]':>12} {'p_max[Pa]':>12} "
        f"{'theta_pk':>10} {'arc_w':>9} {'force_ang':>10}"
    )
    for row in rows:
        print(
            f"{row['omega_rad_s']:8.0f} {row['eccentricity_ratio']:7.3f} "
            f"{row['resultant_force_N']:12.4f} {row['max_pressure_Pa']:12.4e} "
            f"{row['peak_theta_deg']:10.2f} {row['positive_arc_width_deg']:9.2f} "
            f"{row['force_angle_from_radial_deg']:10.2f}"
        )


def print_article_comparison(rows: list[dict[str, float | str]]) -> None:
    print("\nApproximate comparison with Concli 2016 Table V / Fig. 6f")
    print(
        f"{'eps':>7} {'FF F[N]':>10} {'HS F[N]':>10} {'SS F[N]':>10} "
        f"{'Kunz F[N]':>11} {'FF/HS err':>10} {'FF/SS err':>10}"
    )
    for row in rows:
        print(
            f"{row['eccentricity_ratio']:7.3f} "
            f"{row['fluidflow_force_N']:10.1f} "
            f"{row['article_half_sommerfeld_force_N']:10.1f} "
            f"{row['article_schnerr_sauer_force_N']:10.1f} "
            f"{row['article_kunz_force_N']:11.1f} "
            f"{row['fluidflow_vs_article_half_rel_error']:10.1%} "
            f"{row['fluidflow_vs_article_schnerr_sauer_rel_error']:10.1%}"
        )


def make_plots(
    rows: list[dict[str, float | str]],
    temperature: str,
) -> None:
    if plt is None:
        print("Skipping plots: matplotlib is not installed.")
        return

    plt.figure(figsize=(7, 4))
    for omega in sorted({row["omega_rad_s"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["omega_rad_s"] == omega],
            key=lambda row: row["eccentricity_ratio"],
        )
        plt.plot(
            [row["eccentricity_ratio"] for row in subset],
            [row["resultant_force_N"] for row in subset],
            marker="o",
            label=f"{omega:g} rad/s",
        )
    plt.xlabel("eccentricity ratio [-]")
    plt.ylabel("resultant hydraulic force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"concli_2016_force_sweep_{temperature}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for omega in sorted({row["omega_rad_s"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["omega_rad_s"] == omega],
            key=lambda row: row["eccentricity_ratio"],
        )
        plt.plot(
            [row["eccentricity_ratio"] for row in subset],
            [row["max_pressure_Pa"] for row in subset],
            marker="o",
            label=f"{omega:g} rad/s",
        )
    plt.xlabel("eccentricity ratio [-]")
    plt.ylabel("maximum pressure [Pa]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"concli_2016_pressure_sweep_{temperature}.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        choices=["40C", "100C"],
        default="100C",
        help="use the lubricant viscosity tabulated by Concli at 40 C or 100 C",
    )
    args = parser.parse_args()

    bearing = Concli2016Bearing()
    sweep_rows = sweep_cases(bearing, args.temperature)
    figure_rows = figure_13_cases(bearing, args.temperature)
    article_references = approximate_article_force_references()
    article_comparison = compare_with_article_references(sweep_rows, article_references)

    write_csv(OUTPUT_DIR / f"concli_2016_sweep_{args.temperature}.csv", sweep_rows)
    write_csv(
        OUTPUT_DIR / f"concli_2016_figure_13_cases_{args.temperature}.csv",
        figure_rows,
    )
    write_csv(
        OUTPUT_DIR / "concli_2016_article_force_references.csv",
        article_references,
    )
    write_csv(
        OUTPUT_DIR / f"concli_2016_article_force_comparison_{args.temperature}.csv",
        article_comparison,
    )

    print_report(sweep_rows)
    print_article_comparison(article_comparison)
    for message in check_monotonic_trends(sweep_rows):
        print(message)

    make_plots(sweep_rows, args.temperature)

    print("\nBearing setup from Concli 2016 Table II/III:")
    print(f"  R_journal = {bearing.journal_radius:.6g} m")
    print(f"  R_bearing = {bearing.bearing_radius:.6g} m")
    print(f"  L = {bearing.length:.6g} m")
    print(f"  c = {bearing.radial_clearance:.6g} m")
    print(f"  L/D = {bearing.l_over_d:.3f}")
    print(f"  rho = {bearing.density:.1f} kg/m^3")
    print(f"  dynamic viscosity = {bearing.dynamic_viscosity(args.temperature):.6g} Pa*s")
    print("\nSaved:")
    print(f"  {OUTPUT_DIR / f'concli_2016_sweep_{args.temperature}.csv'}")
    print(f"  {OUTPUT_DIR / f'concli_2016_figure_13_cases_{args.temperature}.csv'}")
    print(f"  {OUTPUT_DIR / 'concli_2016_article_force_references.csv'}")
    print(f"  {OUTPUT_DIR / f'concli_2016_article_force_comparison_{args.temperature}.csv'}")


if __name__ == "__main__":
    main()
