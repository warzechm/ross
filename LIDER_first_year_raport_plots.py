# Author: Mariusz Warzecha
# SPDX-License-Identifier: GPL-3.0-or-later
# Date: 31.08.2026
# Description: Script for creation of plots for yearly LIDER report, influence of geometry and viscosity on vibration amplitude;
# this script is based on early 2026 test bench configuration, which was used for comparison with MaDyn; nevertheless it includes
# grooved version of bearing


import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import ross as rs
from ross.bearings import fluid_flow as flow
from fluid_flow_grooves import (
    FluidFlowGrooves,
    find_equilibrium_position_grooved,
)
from ross.bearings.fluid_flow_coefficients import calculate_stiffness_and_damping_coefficients
from ross.bearings.fluid_flow_coefficients import find_equilibrium_position
from ross.bearings.fluid_flow_geometry import (calculate_attitude_angle,
                                               modified_sommerfeld_number,
                                               sommerfeld_number)
from ross.bearings.fluid_flow_graphics import plot_pressure_surface
from matplotlib.patches import Circle
from ross.probe import Probe

# -----------------------
# Plotly renderer
# -----------------------
pio.renderers.default = "browser"  # or "notebook" in Jupyter

d = 0.02  # shaft diameter
# JOURNAL BEARING CONSTANTS
radius_shaft = d/2
radial_clearance = 0.000035  #0.0001/2
radius_journal_bearing = radius_shaft + radial_clearance  # 0.1 mm clearance
nz = 40
ntheta = 257            # MUST be odd 129
length_brg = 0.08
p_in = 0.0  # Pa 4954.0 (rho * g * h) h = 0.505 m (tested but seemed not to work well with model)
p_out = 0.0
rho = 997.0
node_journal_bearing = 44

# element of coupling (disk)
node_disk = 54
m_D = 0.8  # kg, coupling (disk) mass, will be changed to ensure proper loading of journal bearing
r_D = 0.047 # m, assumed average radius of coupling to model it as disk
rho_D = 7200  # kg/m^3, density of cast iron, material used for coupling
b_D = m_D / (np.pi * r_D**2 * rho_D)  # width of the disk calculated to fit the mass
g = 9.81  #  m/s^2, gravitational acceleration
F_D = m_D * g  # force due to coupling weight
I_p = 0.5 * m_D * r_D**2  # moment of inertia along shaft axis
I_d = 1/12 * m_D * (3*r_D**2 + b_D**2)  # moment of inertia along axes perpendicular to shaft axis
coupling = rs.DiskElement(n=node_disk, m=m_D, Id=I_d, Ip=I_p, tag="coupling")
print("Report - created disk parameters:")
print("Disk node: ", node_disk)
print("Mass [kg] and moment of inertia [kg*m^2] (I_p, I_d): ", m_D, I_p, I_d)

# calculation of journal bearing load (for reference see document Schemat_symulacji.svg)
F_w = 24.724449  # shaft weight divided by 10 to get the journal working
journal_bearing_load = (F_D * (1.022 - 0.038) + F_w * (1.022/2 - 0.038)) / (0.833 - 0.038)
print("Calculated journal bearing load in [N]: ", journal_bearing_load)


def detremine_vib_amp_at_journal_bearing(journal_bearing : rs.BearingElement, coupling : rs.DiskElement, rotation_speed : float):
    steel = rs.materials.steel  # definition of material

    # shaft divided into 19 mm segments
    L_elem = 0.019
    N_elem = 54
    shaft = [rs.ShaftElement(L=L_elem, idl=0, odl=d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True)
             for _ in range(N_elem)]
    node_ball_bearing = 2
    unbalance_node = 54

    # rolling bearing
    k_roll = 5e7  # N/m, average value, stiffness depends on exact bearing type, clearance and preload
    c_roll = 200  # N*s/m, this value may be to big
    ball_bearing = rs.BearingElement(n=node_ball_bearing, kxx=k_roll, kyy=k_roll, cxx=c_roll, cyy=c_roll, tag="ball bearing")

    # rotor creation
    rotor = rs.Rotor(shaft_elements=shaft,
                     disk_elements=[coupling],
                     bearing_elements=[ball_bearing, journal_bearing])
    #fig_rotor= rotor.plot_rotor()
    #fig_rotor.write_html("rotor_geometry.html", auto_open=True)

    # unbalance response
    # Unbalance magnitude is m*e in kg*m.
    # Conversion: (g*mm) -> (kg*m) multiply by 1e-6
    # Example: 500 g*mm = 5e-4 kg*m
    unb_magnitude = 3e-4   # [kg*m]  <-- adjust
    unb_phase = 0.0        # [rad]

    # run at a single frequency = omega (rad/s)
    freq = np.array([rotation_speed], dtype=float)

    results = rotor.run_unbalance_response(
        [unbalance_node],              # node
        [unb_magnitude],         # unbalance_magnitude
        [unb_phase],             # unbalance_phase
        freq                     # frequency
    )

    #plot = results.plot_deflected_shape_3d(rotation_speed)
    #plot.show()
    # Complex displacement response array: (ndof, nfreq)
    U = results.forced_resp

    # DOF mapping per node: 0=x, 1=y, 2=theta_x, 3=theta_y
    dofx = rotor.number_dof * node_journal_bearing + 0
    dofy = rotor.number_dof * node_journal_bearing + 1

    X = U[dofx, 0]  # complex amplitude in x, its absolute value decodes sinusoid amplitude
    Y = U[dofy, 0]  # complex amplitude in y
    speed_range = np.linspace(0, 4000 * 2 * np.pi / 60, 81)  # rad/s
    #campbell = rotor.run_campbell(speed_range)
    
    #fig = campbell.plot()
    #fig.show()
    return (abs(X), abs(Y))


def create_plain_fluid_flow_journal(viscosity : float, rotation_speed : float, print_matrix : bool):
    fluid_film = flow.FluidFlow(nz=nz,
                                ntheta=ntheta,
                                length=length_brg,
                                omega=rotation_speed,
                                p_in=p_in,
                                p_out=p_out,
                                radius_rotor=radius_shaft,
                                radius_stator=radius_journal_bearing,
                                viscosity=viscosity,
                                density=rho,
                                load=journal_bearing_load)
    equilibrium_position = find_equilibrium_position(fluid_film, True)
    K_journal_bearing, C_journal_bearing = calculate_stiffness_and_damping_coefficients(fluid_film)
    if print_matrix:
        print(K_journal_bearing)
        print(C_journal_bearing)
        fig = plot_pressure_surface(fluid_film)
        fig.show()

    journal_bearing = rs.BearingElement(n=node_journal_bearing,
                                        kxx=K_journal_bearing[0],
                                        kxy=K_journal_bearing[1],
                                        kyx=K_journal_bearing[2],
                                        kyy=K_journal_bearing[3],
                                        cxx=C_journal_bearing[0],
                                        cxy=C_journal_bearing[1],
                                        cyx=C_journal_bearing[2],
                                        cyy=C_journal_bearing[3],
                                        tag="journal bearing")
    return journal_bearing


def create_grooved_fluid_flow_journal(
    viscosity : float,
    rotation_speed : float,
    grooves : tuple,
    groove_depth : float,
    angle : float,
    print_matrix : bool,
    equilibrium_guess=None,
):
    fluid_film_grooves = FluidFlowGrooves(nz=nz,
                                          ntheta=ntheta,
                                          length=length_brg,
                                          omega=rotation_speed,
                                          p_in=p_in,
                                          p_out=p_out,
                                          grooves=grooves,
                                          groove_depth=groove_depth,
                                          shape_geometry="grooves",
                                          radius_rotor=radius_shaft,
                                          radius_stator=radius_journal_bearing,
                                          viscosity=viscosity,
                                          density=rho,
                                          load=journal_bearing_load,
                                          groove_rotation=angle)
    equilibrium_result = find_equilibrium_position_grooved(
        fluid_film_grooves,
        initial_guess=equilibrium_guess,
        print_result=True,
    )
    K_journal_bearing, C_journal_bearing = calculate_stiffness_and_damping_coefficients(fluid_film_grooves)
    if print_matrix:
        print(K_journal_bearing)
        print(C_journal_bearing)
        fig = plot_pressure_surface(fluid_film_grooves)
        fig.show()

    journal_bearing = rs.BearingElement(n=node_journal_bearing,
                                        kxx=K_journal_bearing[0],
                                        kxy=K_journal_bearing[1],
                                        kyx=K_journal_bearing[2],
                                        kyy=K_journal_bearing[3],
                                        cxx=C_journal_bearing[0],
                                        cxy=C_journal_bearing[1],
                                        cyx=C_journal_bearing[2],
                                        cyy=C_journal_bearing[3],
                                        tag="journal bearing")
    return journal_bearing, equilibrium_result.x


def calculate_amplitudes_over_velocity_range_for_plain_bearing():
    visc = 0.89e-3
    rpm_range = np.arange(600, 4010, 10)
    amplitudes = np.zeros((rpm_range.size, 3))
    for i, j in enumerate(rpm_range):
        print("Working on rotation speed: ", j)
        rotation_speed = j * 2*np.pi/60  # [rad/s]
        journal_bearing = create_plain_fluid_flow_journal(visc, rotation_speed, False)
        (A_x, A_y) = detremine_vib_amp_at_journal_bearing(journal_bearing, coupling, rotation_speed)
        amplitudes[i, 0] = j
        amplitudes[i, 1] = A_x 
        amplitudes[i, 2] = A_y
        np.savetxt("amplitudes_plain_bearing_test.txt", amplitudes, header="rpm,amplitude_x,amplitude_y", delimiter=",")


def calculate_amplitudes_over_velocity_range_for_grooved_bearing(grooves : tuple, groove_depth : float):
    visc = 0.89e-3
    rpm_range = np.arange(600, 4010, 10)
    amplitudes = np.zeros((rpm_range.size, 3))
    equilibrium_guess = None
    first_converged_speed = None
    for i, j in enumerate(rpm_range):
        print("Working on rotation speed: ", j)
        rotation_speed = j * 2*np.pi/60  # [rad/s]
        try:
            journal_bearing, equilibrium_guess = create_grooved_fluid_flow_journal(
                visc,
                rotation_speed,
                grooves,
                groove_depth,
                0*np.pi/180,
                False,
                equilibrium_guess,
            )
        except RuntimeError as error:
            print(f"No equilibrium found at {j} rpm: {error}")
            amplitudes[i, 0] = j
            amplitudes[i, 1:] = np.nan
            equilibrium_guess = None
            np.savetxt(
                "amplitudes_grooved_bearing_0_deg.txt",
                amplitudes,
                header="rpm,amplitude_x,amplitude_y",
                delimiter=",",
            )
            continue

        if first_converged_speed is None:
            first_converged_speed = j
            print(f"First converged equilibrium found at {j} rpm.")

        (A_x, A_y) = detremine_vib_amp_at_journal_bearing(journal_bearing, coupling, rotation_speed)
        amplitudes[i, 0] = j
        amplitudes[i, 1] = A_x 
        amplitudes[i, 2] = A_y
        np.savetxt("amplitudes_grooved_bearing_6_deg.txt", amplitudes, header="rpm,amplitude_x,amplitude_y", delimiter=",")


grooves = ((351.4, 8.6), (36.4, 53.6), (81.4, 98.6), (126.4, 143.6),
           (171.4, 188.6), (216.4, 233.6), (261.4, 278.6), (306.4, 323.6))

#calculate_amplitudes_over_velocity_range_for_plain_bearing()
calculate_amplitudes_over_velocity_range_for_grooved_bearing(grooves, 0.00018)  # groove depth is fake, to get 0 Pa inside groove, but make variation of h^3 values in matrix smaller
#journal_bearing = create_plain_fluid_flow_journal(0.89, 1200*2*np.pi/60, False)
#(A_x, A_y) = detremine_vib_amp_at_journal_bearing(journal_bearing, coupling, 1200*2*np.pi/60)
amp_plain = np.loadtxt("amplitudes_plain_bearing.txt", delimiter=",", skiprows=1)
amp_grooved = np.loadtxt("amplitudes_grooved_bearing_0_deg.txt", delimiter=",", skiprows=1)
amp_grooved_6 = np.loadtxt("amplitudes_grooved_bearing_6_deg.txt", delimiter=",", skiprows=1)
fig, ax = plt.subplots()
ax.plot(amp_plain[:,0], amp_plain[:, 2], label="plain")
ax.plot(amp_grooved[:,0], amp_grooved[:, 2], label="grooved_0")
ax.plot(amp_grooved_6[:,0], amp_grooved_6[:, 2], label="grooved_6")
ax.legend()
plt.show()






"""
# Save results to .txt file
results = np.column_stack((t, x_t, y_t))
np.savetxt('simulation_results_2800_rpm_3_e-4_kg*m.txt', results)

speed_range = np.linspace(0, 3000*2*np.pi/60, 300)  # rad/s (if your model uses rad/s)
results2 = rotor.run_unbalance_response(
    [unbalance_node],              # node
    [unb_magnitude],         # unbalance_magnitude
    [unb_phase],             # unbalance_phase
    speed_range                     # frequency
)
probe = Probe(node=unbalance_node, angle="major", tag="Probe @ node 3 (major)")
fig = results2.plot(probe=[probe], amplitude_units="m", phase_units="deg")
fig.show()
#mod_sommerfeld_number = modified_sommerfeld_number(radius_journal_bearing, omega, visc, length_brg, load, radial_clearance)
#sommer_number = sommerfeld_number(mod_sommerfeld_number, radius_journal_bearing, length_brg)
#print("Modified Sommerfeld number: ", mod_sommerfeld_number, "\n")
#print("Sommerfeld number", sommer_number, "\n")


    # Reconstruct orbit in time: x(t)=Re(X*e^(jωt)), y(t)=Re(Y*e^(jωt))
    T = 2*np.pi / rotation_speed
    n_cycles = 20
    n_per_rev = 400

    t = np.linspace(0, n_cycles*T, n_cycles*n_per_rev, endpoint=False)
    ejwt = np.exp(1j * rotation_speed * t)

    x_t = np.real(X * ejwt)
    y_t = np.real(Y * ejwt)

    #Plot orbit (ellipse)
    fig, ax = plt.subplots()
    ax.plot(t, x_t)
    #ax.plot(x_t + 6.079055233197487e-05 ,y_t - 6.236860175127931e-05 )
    #ax.plot(t, x_t)
    # --- add circle ---
    #r_circle = 0.0001  # example: 50 micrometers
    #circle = Circle((0.0, 0.0), r_circle, fill=False, linewidth=2)  # center at (0,0)
    #ax.add_patch(circle)
    # ------------------
    #ax.set_xlabel("x displacement [m]")
    #ax.set_ylabel("y displacement [m]")
    #ax.set_title(f"Journal node orbit at {rpm} rpm (unbalance excitation)")
    #ax.grid(True)
    #ax.set_aspect("equal", adjustable="box")  # prevents visual distortion
    plt.show()

"""
