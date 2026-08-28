"""Validate ROSS FluidFlow against Chasalevris & Sfyris (2013).

Reference:
    A. Chasalevris, D. Sfyris,
    "Evaluation of the finite journal bearing characteristics, using the exact
    analytical solution of the Reynolds equation", Tribology International 57
    (2013), 216-234.

The paper provides an Appendix A finite-difference model. This script uses that
Appendix A model as a numerical reference and compares it with ROSS FluidFlow for
the same plain finite journal-bearing assumptions:

* Newtonian, laminar, isoviscous, isothermal lubricant,
* aligned rigid journal and bearing,
* Gumbel / half-Sommerfeld pressure boundary condition,
* finite bearing with selected L/D values.
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent


def bootstrap_bearing_modules() -> None:
    """Import only the ROSS bearing modules needed by FluidFlow."""
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
class BearingCase:
    radius: float = 0.1
    clearance: float = 1e-4
    viscosity: float = 0.01
    density: float = 860.0
    omega: float = 100.0
    l_over_d: float = 1.0

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def length(self) -> float:
        return self.l_over_d * self.diameter

    @property
    def stator_radius(self) -> float:
        return self.radius + self.clearance


def load_from_chasalevris_sommerfeld(case: BearingCase, sommerfeld: float) -> float:
    """Invert the paper's Sommerfeld definition.

    The paper defines:
        S = 1/(2*pi) * (R/cr)^2 * (mu * Lb * (2R) * Omega / W)
    """
    return (
        (1 / (2 * np.pi))
        * (case.radius / case.clearance) ** 2
        * case.viscosity
        * case.length
        * case.diameter
        * case.omega
        / sommerfeld
    )


@lru_cache(maxsize=None)
def chasalevris_fdm_pressure(
    l_over_d: float,
    eccentricity_ratio: float,
    radius: float,
    clearance: float,
    viscosity: float,
    omega: float,
    nx_intervals: int = 15,
    ny_intervals: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appendix A FDM pressure field.

    Returns
    -------
    x : ndarray
        Interior axial coordinates.
    theta : ndarray
        Interior circumferential coordinates in the positive half-film.
    pressure : ndarray
        Pressure field with shape (nx_intervals - 1, ny_intervals - 1).
    """
    length = l_over_d * 2 * radius
    eccentricity = eccentricity_ratio * clearance
    dx = length / nx_intervals
    dtheta = np.pi / ny_intervals
    n_x = nx_intervals - 1
    n_t = ny_intervals - 1
    n_total = n_x * n_t

    def idx(i: int, j: int) -> int:
        return (i - 1) * n_t + (j - 1)

    matrix = np.zeros((n_total, n_total))
    rhs = np.zeros(n_total)

    for i in range(1, nx_intervals):
        for j in range(1, ny_intervals):
            row = idx(i, j)
            theta = j * dtheta
            h = clearance + eccentricity * np.cos(theta)

            axial = h**3 / (6 * viscosity * dx**2)
            circum = h**3 / (6 * viscosity * radius**2 * dtheta**2)
            convective = (
                -eccentricity
                * h**2
                * np.sin(theta)
                / (2 * viscosity)
                / (radius**2 * 2 * dtheta)
            )

            matrix[row, row] += -2 * axial - 2 * circum
            if i + 1 < nx_intervals:
                matrix[row, idx(i + 1, j)] += axial
            if i - 1 > 0:
                matrix[row, idx(i - 1, j)] += axial
            if j + 1 < ny_intervals:
                matrix[row, idx(i, j + 1)] += circum + convective
            if j - 1 > 0:
                matrix[row, idx(i, j - 1)] += circum - convective

            rhs[row] = -omega * eccentricity * np.sin(theta)

    pressure = solve(matrix, rhs).reshape((n_x, n_t))
    x = np.array([-length / 2 + i * dx for i in range(1, nx_intervals)])
    theta = np.array([j * dtheta for j in range(1, ny_intervals)])
    return x, theta, pressure


def chasalevris_fdm_force(case: BearingCase, eccentricity_ratio: float) -> tuple[float, float, float]:
    x, theta, pressure = chasalevris_fdm_pressure(
        case.l_over_d,
        eccentricity_ratio,
        case.radius,
        case.clearance,
        case.viscosity,
        case.omega,
    )
    dx = case.length / 15
    dtheta = np.pi / 45
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radial = np.sum(pressure * cos_t[None, :] * case.radius * dtheta * dx)
    tangential = np.sum(pressure * sin_t[None, :] * case.radius * dtheta * dx)
    resultant = float(np.hypot(radial, tangential))
    return float(radial), float(tangential), resultant


def chasalevris_equilibrium_epsilon(case: BearingCase, sommerfeld: float) -> float:
    target_load = load_from_chasalevris_sommerfeld(case, sommerfeld)

    def residual(eps: float) -> float:
        return chasalevris_fdm_force(case, eps)[2] - target_load

    return float(brentq(residual, 1e-5, 0.999, maxiter=80))


def fluid_flow_fixed_eccentricity(case: BearingCase, eccentricity_ratio: float) -> flow.FluidFlow:
    return flow.FluidFlow(
        nz=16,
        ntheta=91,
        length=case.length,
        omega=case.omega,
        p_in=0.0,
        p_out=0.0,
        radius_rotor=case.radius,
        radius_stator=case.stator_radius,
        viscosity=case.viscosity,
        density=case.density,
        eccentricity=eccentricity_ratio * case.clearance,
        attitude_angle=0.0,
        immediately_calculate_pressure_matrix_numerically=True,
    )


def fluid_flow_equilibrium(case: BearingCase, sommerfeld: float) -> flow.FluidFlow:
    load = load_from_chasalevris_sommerfeld(case, sommerfeld)
    return flow.FluidFlow(
        nz=16,
        ntheta=91,
        length=case.length,
        omega=case.omega,
        p_in=0.0,
        p_out=0.0,
        radius_rotor=case.radius,
        radius_stator=case.stator_radius,
        viscosity=case.viscosity,
        density=case.density,
        load=load,
        immediately_calculate_pressure_matrix_numerically=True,
    )


def central_axial_profile(values: np.ndarray) -> np.ndarray:
    mid_upper = values.shape[0] // 2
    mid_lower = mid_upper - 1
    return 0.5 * (values[mid_lower] + values[mid_upper])


def pressure_profile_validation(case: BearingCase) -> list[dict[str, float]]:
    rows = []
    for eps in [0.3, 0.5, 0.7, 0.9]:
        _, theta_ref, pressure_ref = chasalevris_fdm_pressure(
            case.l_over_d,
            eps,
            case.radius,
            case.clearance,
            case.viscosity,
            case.omega,
        )
        ref_profile = central_axial_profile(pressure_ref)

        ff = fluid_flow_fixed_eccentricity(case, eps)
        ross_profile_full = central_axial_profile(ff.p_mat_numerical)
        gamma = ff.gama[0] % (2 * np.pi)
        positive = ross_profile_full > 1e-9
        theta_ross = gamma[positive] - np.pi / 2
        pressure_ross = ross_profile_full[positive]
        order = np.argsort(theta_ross)
        theta_ross = theta_ross[order]
        pressure_ross = pressure_ross[order]
        interp_ross = np.interp(theta_ref, theta_ross, pressure_ross)

        peak_ref = float(np.max(ref_profile))
        peak_ross = float(np.max(interp_ross))
        rms = float(np.sqrt(np.mean((interp_ross - ref_profile) ** 2)))
        rows.append(
            {
                "l_over_d": case.l_over_d,
                "eccentricity_ratio": eps,
                "reference_peak_pressure_Pa": peak_ref,
                "fluidflow_peak_pressure_Pa": peak_ross,
                "peak_relative_error": (peak_ross - peak_ref) / peak_ref,
                "normalized_rms_error": rms / peak_ref,
                "reference_peak_theta_deg": float(np.degrees(theta_ref[np.argmax(ref_profile)])),
                "fluidflow_peak_theta_deg": float(np.degrees(theta_ref[np.argmax(interp_ross)])),
            }
        )

        if plt is not None:
            plt.figure(figsize=(7, 4))
            plt.plot(np.degrees(theta_ref), ref_profile, label="Appendix A FDM")
            plt.plot(np.degrees(theta_ref), interp_ross, "--", label="ROSS FluidFlow")
            plt.xlabel("theta [deg]")
            plt.ylabel("pressure [Pa]")
            plt.title(f"Pressure profile, L/D={case.l_over_d:g}, epsilon={eps:g}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"chasalevris_pressure_LD_{case.l_over_d:g}_eps_{eps:g}.png",
                dpi=160,
            )
            plt.close()

    return rows


def equilibrium_validation(l_over_d_values: list[float], sommerfeld_values: list[float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for l_over_d in l_over_d_values:
        case = BearingCase(l_over_d=l_over_d)
        for sommerfeld in sommerfeld_values:
            print(
                f"Solving equilibrium point: L/D={l_over_d:g}, S={sommerfeld:g}",
                flush=True,
            )
            load = load_from_chasalevris_sommerfeld(case, sommerfeld)
            ref_eps = chasalevris_equilibrium_epsilon(case, sommerfeld)
            try:
                ff = fluid_flow_equilibrium(case, sommerfeld)
                ross_eps = float(ff.eccentricity / case.clearance)
                status = "ok"
                error = ""
            except Exception as exc:
                ross_eps = np.nan
                status = "failed"
                error = str(exc)

            rows.append(
                {
                    "status": status,
                    "error": error,
                    "l_over_d": l_over_d,
                    "sommerfeld": sommerfeld,
                    "load_N": load,
                    "reference_eccentricity_ratio": ref_eps,
                    "fluidflow_eccentricity_ratio": ross_eps,
                    "absolute_error": ross_eps - ref_eps,
                    "relative_error": (ross_eps - ref_eps) / ref_eps,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_pressure_report(rows: list[dict[str, float]]) -> None:
    print("\nPressure profile validation against Appendix A FDM, Fig. 2 setup")
    print(
        f"{'eps':>8} {'p_ref_max':>14} {'p_ross_max':>14} "
        f"{'peak_err':>10} {'rms_norm':>10} {'theta_ref':>10} {'theta_ross':>11}"
    )
    for row in rows:
        print(
            f"{row['eccentricity_ratio']:8.3f} "
            f"{row['reference_peak_pressure_Pa']:14.4e} "
            f"{row['fluidflow_peak_pressure_Pa']:14.4e} "
            f"{row['peak_relative_error']:10.3%} "
            f"{row['normalized_rms_error']:10.3%} "
            f"{row['reference_peak_theta_deg']:10.2f} "
            f"{row['fluidflow_peak_theta_deg']:11.2f}"
        )


def print_equilibrium_report(rows: list[dict[str, float | str]]) -> None:
    print("\nEquilibrium validation: eccentricity ratio vs Sommerfeld number")
    print(
        f"{'L/D':>6} {'S':>8} {'status':>8} {'eps_ref':>10} "
        f"{'eps_ross':>10} {'abs_err':>10} {'rel_err':>10}"
    )
    for row in rows:
        print(
            f"{row['l_over_d']:6.2f} {row['sommerfeld']:8.3g} "
            f"{row['status']:>8} {row['reference_eccentricity_ratio']:10.4f} "
            f"{row['fluidflow_eccentricity_ratio']:10.4f} "
            f"{row['absolute_error']:10.4f} {row['relative_error']:10.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-equilibrium",
        action="store_true",
        help="only validate pressure profiles; equilibrium points are slower",
    )
    parser.add_argument(
        "--full-equilibrium",
        action="store_true",
        help="validate L/D = 0.5, 1 and 4; default validates only L/D = 1",
    )
    args = parser.parse_args()

    pressure_rows = pressure_profile_validation(BearingCase(l_over_d=1.0))
    write_csv(OUTPUT_DIR / "chasalevris_pressure_profile_validation.csv", pressure_rows)
    print_pressure_report(pressure_rows)

    if not args.skip_equilibrium:
        l_over_d_values = [0.5, 1.0, 4.0] if args.full_equilibrium else [1.0]
        equilibrium_rows = equilibrium_validation(
            l_over_d_values=l_over_d_values,
            sommerfeld_values=[0.01, 0.1, 1.0, 10.0],
        )
        write_csv(OUTPUT_DIR / "chasalevris_equilibrium_validation.csv", equilibrium_rows)
        print_equilibrium_report(equilibrium_rows)

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'chasalevris_pressure_profile_validation.csv'}")
    if not args.skip_equilibrium:
        print(f"  {OUTPUT_DIR / 'chasalevris_equilibrium_validation.csv'}")
    if plt is None:
        print("  pressure plots skipped: matplotlib is not installed")


if __name__ == "__main__":
    main()
