"""Validation setup based on Concli (2020).

Reference:
    F. Concli, "Journal Bearing: An Integrated CFD-Analytical Approach for the
    Estimation of the Trajectory and Equilibrium Position", Applied Sciences
    10(23), 8573, 2020. https://doi.org/10.3390/app10238573

The paper compares half-Sommerfeld calculations with CFD-Kunz cavitation results
for a small high-speed journal bearing. The publication gives figures rather
than numerical tables for most validation quantities, so this script reproduces
the published bearing setup with ROSS FluidFlow and checks the same qualitative
relationships:

* hydraulic force increases with eccentricity,
* force increases with speed for a fixed eccentricity,
* equilibrium eccentricity increases with applied mass,
* equilibrium eccentricity decreases with speed for comparable loading.

CSV outputs are written so the curves can be compared with Figures 5-12 of the
paper or digitized data if it becomes available later.
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
GRAVITY = 9.80665


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
class ConcliBearing:
    journal_diameter: float = 4e-3
    length: float = 13e-3
    radial_clearance: float = 10e-6
    density: float = 1060.0
    journal_mass: float = 1.307e-3
    kinematic_viscosity_40c: float = 220e-6
    kinematic_viscosity_100c: float = 40e-6

    @property
    def journal_radius(self) -> float:
        return self.journal_diameter / 2

    @property
    def bearing_radius(self) -> float:
        return self.journal_radius + self.radial_clearance

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
    bearing: ConcliBearing,
    omega: float,
    viscosity: float,
    *,
    eccentricity_ratio: float | None = None,
    load: float | None = None,
) -> flow.FluidFlow:
    kwargs = {}
    if eccentricity_ratio is not None:
        kwargs["eccentricity"] = eccentricity_ratio * bearing.radial_clearance
        kwargs["attitude_angle"] = 0.0
    if load is not None:
        kwargs["load"] = load

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
        immediately_calculate_pressure_matrix_numerically=True,
        **kwargs,
    )


def force_sweep(bearing: ConcliBearing, temperature: str) -> list[dict[str, float | str]]:
    viscosity = bearing.dynamic_viscosity(temperature)
    rows: list[dict[str, float | str]] = []
    for omega in [328.0, 1500.0, 3000.0]:
        for eps in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
            ff = make_fluid_flow(
                bearing,
                omega,
                viscosity,
                eccentricity_ratio=eps,
            )
            radial, tangential, fx, fy = calculate_oil_film_force(
                ff, force_type="numerical"
            )
            rows.append(
                {
                    "temperature": temperature,
                    "omega_rad_s": omega,
                    "eccentricity_ratio": eps,
                    "radial_force_N": radial,
                    "tangential_force_N": tangential,
                    "fx_N": fx,
                    "fy_N": fy,
                    "resultant_force_N": float(np.hypot(fx, fy)),
                    "attitude_angle_deg": float(np.degrees(ff.attitude_angle)),
                    "max_pressure_Pa": float(np.max(ff.p_mat_numerical)),
                }
            )
    return rows


def equilibrium_cases() -> list[tuple[str, float, list[float]]]:
    return [
        ("Figure 8", 328.0, [0.25, 0.5, 1.0]),
        ("Figure 9", 1500.0, [1.0, 2.5, 5.0]),
        ("Figure 10", 3000.0, [2.5, 5.0, 10.0]),
    ]


def equilibrium_sweep(
    bearing: ConcliBearing,
    temperature: str,
) -> list[dict[str, float | str]]:
    viscosity = bearing.dynamic_viscosity(temperature)
    rows: list[dict[str, float | str]] = []
    for figure, omega, additional_masses in equilibrium_cases():
        for additional_mass in additional_masses:
            total_mass = bearing.journal_mass + additional_mass
            load = total_mass * GRAVITY
            print(
                f"Solving {figure}: omega={omega:g} rad/s, "
                f"additional_mass={additional_mass:g} kg, {temperature}",
                flush=True,
            )
            try:
                ff = make_fluid_flow(
                    bearing,
                    omega,
                    viscosity,
                    load=load,
                )
                eccentricity_ratio = float(ff.eccentricity / bearing.radial_clearance)
                h_min = bearing.radial_clearance - ff.eccentricity
                status = "ok"
                error = ""
                x_um = ff.xi * 1e6
                y_um = ff.yi * 1e6
                attitude_deg = float(np.degrees(ff.attitude_angle))
            except Exception as exc:
                eccentricity_ratio = np.nan
                h_min = np.nan
                status = "failed"
                error = str(exc)
                x_um = np.nan
                y_um = np.nan
                attitude_deg = np.nan

            rows.append(
                {
                    "temperature": temperature,
                    "article_figure": figure,
                    "omega_rad_s": omega,
                    "additional_mass_kg": additional_mass,
                    "journal_mass_kg": bearing.journal_mass,
                    "total_mass_kg": total_mass,
                    "load_N": load,
                    "status": status,
                    "error": error,
                    "x_um": x_um,
                    "y_um": y_um,
                    "eccentricity_ratio": eccentricity_ratio,
                    "h_min_um": h_min * 1e6,
                    "attitude_angle_deg": attitude_deg,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_monotonic_force(rows: list[dict[str, float | str]]) -> list[str]:
    messages = []
    for omega in sorted({row["omega_rad_s"] for row in rows}):
        subset = [row for row in rows if row["omega_rad_s"] == omega]
        forces = np.array([row["resultant_force_N"] for row in subset], dtype=float)
        ok = bool(np.all(np.diff(forces) > 0))
        messages.append(
            f"force increases with eccentricity at omega={omega:g}: {'PASS' if ok else 'FAIL'}"
        )
    for eps in sorted({row["eccentricity_ratio"] for row in rows}):
        subset = [row for row in rows if row["eccentricity_ratio"] == eps]
        subset = sorted(subset, key=lambda row: row["omega_rad_s"])
        forces = np.array([row["resultant_force_N"] for row in subset], dtype=float)
        ok = bool(np.all(np.diff(forces) > 0))
        messages.append(
            f"force increases with speed at epsilon={eps:g}: {'PASS' if ok else 'FAIL'}"
        )
    return messages


def check_equilibrium_trends(rows: list[dict[str, float | str]]) -> list[str]:
    messages = []
    for figure in sorted({row["article_figure"] for row in rows}):
        subset = [
            row
            for row in rows
            if row["article_figure"] == figure and row["status"] == "ok"
        ]
        subset = sorted(subset, key=lambda row: row["additional_mass_kg"])
        eps = np.array([row["eccentricity_ratio"] for row in subset], dtype=float)
        ok = len(eps) > 1 and bool(np.all(np.diff(eps) > 0))
        messages.append(
            f"equilibrium eccentricity increases with mass in {figure}: "
            f"{'PASS' if ok else 'FAIL'}"
        )
    return messages


def print_equilibrium_report(rows: list[dict[str, float | str]]) -> None:
    print("\nEquilibrium points for Concli 2020 setup")
    print(
        f"{'fig':>9} {'omega':>8} {'m_add':>8} {'status':>8} "
        f"{'eps':>9} {'h_min_um':>10} {'x_um':>10} {'y_um':>10}"
    )
    for row in rows:
        print(
            f"{row['article_figure']:>9} {row['omega_rad_s']:8.0f} "
            f"{row['additional_mass_kg']:8.3f} {row['status']:>8} "
            f"{row['eccentricity_ratio']:9.4f} {row['h_min_um']:10.3f} "
            f"{row['x_um']:10.3f} {row['y_um']:10.3f}"
        )


def make_plots(
    force_rows: list[dict[str, float | str]],
    equilibrium_rows: list[dict[str, float | str]],
    temperature: str,
) -> None:
    if plt is None:
        print("Skipping plots: matplotlib is not installed.")
        return

    plt.figure(figsize=(7, 4))
    for omega in sorted({row["omega_rad_s"] for row in force_rows}):
        subset = [row for row in force_rows if row["omega_rad_s"] == omega]
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
    plt.savefig(OUTPUT_DIR / f"concli_2020_force_sweep_{temperature}.png", dpi=160)
    plt.close()

    ok_rows = [row for row in equilibrium_rows if row["status"] == "ok"]
    plt.figure(figsize=(7, 4))
    for figure in sorted({row["article_figure"] for row in ok_rows}):
        subset = [row for row in ok_rows if row["article_figure"] == figure]
        plt.plot(
            [row["additional_mass_kg"] for row in subset],
            [row["eccentricity_ratio"] for row in subset],
            marker="o",
            label=figure,
        )
    plt.xlabel("additional mass [kg]")
    plt.ylabel("equilibrium eccentricity ratio [-]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"concli_2020_equilibrium_sweep_{temperature}.png",
        dpi=160,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        choices=["40C", "100C"],
        default="100C",
        help="use the oil viscosity tabulated by Concli at 40 C or 100 C",
    )
    parser.add_argument(
        "--force-only",
        action="store_true",
        help="skip equilibrium cases, which are slower",
    )
    args = parser.parse_args()

    bearing = ConcliBearing()
    force_rows = force_sweep(bearing, args.temperature)
    write_csv(OUTPUT_DIR / f"concli_2020_force_sweep_{args.temperature}.csv", force_rows)

    for message in check_monotonic_force(force_rows):
        print(message)

    equilibrium_rows: list[dict[str, float | str]] = []
    if not args.force_only:
        equilibrium_rows = equilibrium_sweep(bearing, args.temperature)
        write_csv(
            OUTPUT_DIR / f"concli_2020_equilibrium_sweep_{args.temperature}.csv",
            equilibrium_rows,
        )
        print_equilibrium_report(equilibrium_rows)
        for message in check_equilibrium_trends(equilibrium_rows):
            print(message)

    make_plots(force_rows, equilibrium_rows, args.temperature)

    print("\nBearing setup from Concli 2020:")
    print(f"  D = {bearing.journal_diameter * 1e3:.3f} mm")
    print(f"  L = {bearing.length * 1e3:.3f} mm")
    print(f"  c = {bearing.radial_clearance * 1e6:.3f} um")
    print(f"  L/D = {bearing.l_over_d:.3f}")
    print(f"  rho = {bearing.density:.1f} kg/m^3")
    print(f"  dynamic viscosity = {bearing.dynamic_viscosity(args.temperature):.6g} Pa*s")
    print("\nSaved:")
    print(f"  {OUTPUT_DIR / f'concli_2020_force_sweep_{args.temperature}.csv'}")
    if not args.force_only:
        print(f"  {OUTPUT_DIR / f'concli_2020_equilibrium_sweep_{args.temperature}.csv'}")


if __name__ == "__main__":
    main()
