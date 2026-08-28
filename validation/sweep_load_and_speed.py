"""Sweep load and speed for the test bench journal bearing.

The script uses the same bearing parameters as
``test_bench_early_2026_modification.py`` and checks whether the static
equilibrium behaves physically:

* increasing speed should generally reduce eccentricity,
* increasing load should generally increase eccentricity,
* lightly loaded/high-speed cases should tend toward a larger attitude angle.
"""

from __future__ import annotations

import csv
import argparse
import sys
import types
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap_bearing_modules() -> None:
    """Load bearing modules without importing the full ROSS package.

    The full package import initializes plotting and rotor modules. This validation
    script only needs ``ross.bearings`` and should also run in lean environments
    where optional plotting dependencies are not installed.
    """
    ross_pkg = types.ModuleType("ross")
    ross_pkg.__path__ = [str(PROJECT_ROOT / "ross")]
    sys.modules.setdefault("ross", ross_pkg)

    bearings_pkg = types.ModuleType("ross.bearings")
    bearings_pkg.__path__ = [str(PROJECT_ROOT / "ross" / "bearings")]
    sys.modules.setdefault("ross.bearings", bearings_pkg)

    graphics_stub = types.ModuleType("ross.bearings.fluid_flow_graphics")

    def plot_pressure_surface(*args, **kwargs):
        raise RuntimeError("Plotting is not available in this validation script.")

    graphics_stub.plot_pressure_surface = plot_pressure_surface
    sys.modules.setdefault("ross.bearings.fluid_flow_graphics", graphics_stub)


bootstrap_bearing_modules()

from ross.bearings import fluid_flow as flow
from ross.bearings.fluid_flow_coefficients import (
    calculate_oil_film_force,
)
from ross.bearings.fluid_flow_geometry import (
    modified_sommerfeld_number,
    reynolds_number,
    sommerfeld_number,
)


OUTPUT_DIR = Path(__file__).resolve().parent


# Parameters copied from test_bench_early_2026_modification.py.
D_SHAFT = 0.02
RADIUS_SHAFT = D_SHAFT / 2
RADIAL_CLEARANCE = 0.0001
RADIUS_JOURNAL_BEARING = RADIUS_SHAFT + RADIAL_CLEARANCE
NZ = 20
NTHETA = 129
LENGTH_BRG = 0.08
P_IN = 4954.0
P_OUT = 0.0
VISCOSITY = 0.89e-3
DENSITY = 997.0
BASE_LOAD = 35.35
BASE_RPM = 2800.0


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * 2 * np.pi / 60


def failed_result(load: float, rpm: float, error: Exception) -> dict[str, float | str]:
    omega = rpm_to_rad_s(rpm)
    return {
        "status": "failed",
        "error": str(error),
        "rpm": rpm,
        "omega_rad_s": omega,
        "load_N": load,
        "x0_um": np.nan,
        "y0_um": np.nan,
        "eccentricity_um": np.nan,
        "eccentricity_ratio": np.nan,
        "h_min_um": np.nan,
        "attitude_from_load_axis_deg": np.nan,
        "angle_from_x_deg": np.nan,
        "fx_N": np.nan,
        "fy_N": np.nan,
        "fy_minus_load_N": np.nan,
        "modified_sommerfeld": np.nan,
        "sommerfeld": np.nan,
        "reynolds": np.nan,
    }


def solve_equilibrium(load: float, rpm: float) -> dict[str, float | str]:
    omega = rpm_to_rad_s(rpm)

    try:
        bearing = flow.FluidFlow(
            nz=NZ,
            ntheta=NTHETA,
            length=LENGTH_BRG,
            omega=omega,
            p_in=P_IN,
            p_out=P_OUT,
            radius_rotor=RADIUS_SHAFT,
            radius_stator=RADIUS_JOURNAL_BEARING,
            viscosity=VISCOSITY,
            density=DENSITY,
            load=load,
            immediately_calculate_pressure_matrix_numerically=False,
        )

        # FluidFlow already calls find_equilibrium_position() when load is given.
        bearing.calculate_pressure_matrix_numerical()
        _, _, fx, fy = calculate_oil_film_force(bearing, force_type="numerical")
    except Exception as error:
        return failed_result(load, rpm, error)

    eccentricity = float(np.hypot(bearing.xi, bearing.yi))
    eccentricity_ratio = eccentricity / RADIAL_CLEARANCE
    h_min = RADIAL_CLEARANCE - eccentricity
    attitude_from_load_axis = float(np.degrees(np.arccos(abs(bearing.yi / eccentricity))))
    angle_from_x = float(np.degrees(np.arctan2(bearing.yi, bearing.xi)))
    mod_s = modified_sommerfeld_number(
        RADIUS_JOURNAL_BEARING,
        omega,
        VISCOSITY,
        LENGTH_BRG,
        load,
        RADIAL_CLEARANCE,
    )

    return {
        "status": "ok",
        "error": "",
        "rpm": rpm,
        "omega_rad_s": omega,
        "load_N": load,
        "x0_um": bearing.xi * 1e6,
        "y0_um": bearing.yi * 1e6,
        "eccentricity_um": eccentricity * 1e6,
        "eccentricity_ratio": eccentricity_ratio,
        "h_min_um": h_min * 1e6,
        "attitude_from_load_axis_deg": attitude_from_load_axis,
        "angle_from_x_deg": angle_from_x,
        "fx_N": fx,
        "fy_N": fy,
        "fy_minus_load_N": fy - load,
        "modified_sommerfeld": mod_s,
        "sommerfeld": sommerfeld_number(mod_s, RADIUS_JOURNAL_BEARING, LENGTH_BRG),
        "reynolds": reynolds_number(
            DENSITY,
            omega * RADIUS_SHAFT,
            RADIAL_CLEARANCE,
            VISCOSITY,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def successful_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    return [row for row in rows if row["status"] == "ok"]


def plot_sweep(
    path: Path,
    rows: list[dict[str, float | str]],
    x_key: str,
    x_label: str,
) -> None:
    if plt is None:
        print(f"Skipping {path.name}: matplotlib is not installed.")
        return

    rows = successful_rows(rows)
    if not rows:
        print(f"Skipping {path.name}: no successful points.")
        return

    x = np.array([row[x_key] for row in rows])
    eccentricity_ratio = np.array([row["eccentricity_ratio"] for row in rows])
    h_min_um = np.array([row["h_min_um"] for row in rows])
    attitude = np.array([row["attitude_from_load_axis_deg"] for row in rows])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(x, eccentricity_ratio, marker="o")
    axes[0].set_ylabel("epsilon [-]")
    axes[0].grid(True)

    axes[1].plot(x, h_min_um, marker="o")
    axes[1].set_ylabel("h_min [um]")
    axes[1].grid(True)

    axes[2].plot(x, attitude, marker="o")
    axes[2].set_ylabel("attitude [deg]")
    axes[2].set_xlabel(x_label)
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(title: str, rows: list[dict[str, float | str]], x_key: str) -> None:
    print(f"\n{title}")
    print(
        f"{x_key:>10} {'load_N':>10} {'x0_um':>12} {'y0_um':>12} "
        f"{'eps':>10} {'h_min_um':>12} {'att_deg':>10} {'Fy-W':>12}"
    )
    for row in rows:
        if row["status"] != "ok":
            print(
                f"{row[x_key]:10.3f} {row['load_N']:10.3f} "
                f"{'FAILED':>58} {row['error']}"
            )
            continue

        print(
            f"{row[x_key]:10.3f} {row['load_N']:10.3f} "
            f"{row['x0_um']:12.3f} {row['y0_um']:12.3f} "
            f"{row['eccentricity_ratio']:10.4f} {row['h_min_um']:12.3f} "
            f"{row['attitude_from_load_axis_deg']:10.3f} "
            f"{row['fy_minus_load_N']:12.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate static journal-bearing equilibrium trends."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run a denser sweep; this may take several minutes",
    )
    args = parser.parse_args()

    if args.full:
        speed_rpms = np.array(
            [500, 750, 1000, 1500, 2000, 2500, 2800, 3000, 3500, 4000]
        )
        load_values = np.array([5, 10, 15, 20, 25, 30, BASE_LOAD, 40, 50, 60])
    else:
        speed_rpms = np.array([1000, 2000, BASE_RPM, 3500, 4000])
        load_values = np.array([10, 20, BASE_LOAD, 50])

    speed_rows = []
    for rpm in speed_rpms:
        print(f"Solving speed sweep point: rpm={rpm:.1f}, load={BASE_LOAD:.3f} N", flush=True)
        speed_rows.append(solve_equilibrium(BASE_LOAD, float(rpm)))

    load_rows = []
    for load in load_values:
        print(f"Solving load sweep point: rpm={BASE_RPM:.1f}, load={load:.3f} N", flush=True)
        load_rows.append(solve_equilibrium(float(load), BASE_RPM))

    write_csv(OUTPUT_DIR / "sweep_speed_results.csv", speed_rows)
    write_csv(OUTPUT_DIR / "sweep_load_results.csv", load_rows)
    plot_sweep(OUTPUT_DIR / "sweep_speed_results.png", speed_rows, "rpm", "speed [rpm]")
    plot_sweep(OUTPUT_DIR / "sweep_load_results.png", load_rows, "load_N", "load [N]")

    print_summary("Speed sweep at constant load", speed_rows, "rpm")
    print_summary("Load sweep at constant speed", load_rows, "load_N")
    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'sweep_speed_results.csv'}")
    print(f"  {OUTPUT_DIR / 'sweep_load_results.csv'}")
    if plt is not None:
        print(f"  {OUTPUT_DIR / 'sweep_speed_results.png'}")
        print(f"  {OUTPUT_DIR / 'sweep_load_results.png'}")


if __name__ == "__main__":
    main()
