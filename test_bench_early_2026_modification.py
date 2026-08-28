""" This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Written by: Mariusz Warzecha

    Date: 10.04.2026

    Description: This model should represent test bench modified by Mateusz in January 2026"""


import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import ross as rs
from ross.bearings import fluid_flow as flow
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
steel = rs.materials.steel  # definition of material

# shaft divided into three segments
"""
L0, L1, L2 = 0.038, 0.795, 0.189  # distances to split the shaft and obtain nodes where bearings are
shaft = [rs.ShaftElement(L=L0, idl=0, odl=d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True),
         rs.ShaftElement(L=L1, idl=0, odl=d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True),
         rs.ShaftElement(L=L2, idl=0, odl=d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True)]
node_ball_bearing = 1
node_journal_bearing = 2
node_disk = 3
unbalance_node = 3
"""

# shaft divided into 19 mm segments
L_elem = 0.019
N_elem = 54
shaft = [rs.ShaftElement(L=L_elem, idl=0, odl=d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True)
         for _ in range(N_elem)]
node_ball_bearing = 2
node_journal_bearing = 44
node_disk = 54
unbalance_node = 54


# rolling bearing
k_roll = 5e7  # N/m, average value, stiffness depends on exact bearing type, clearance and preload
c_roll = 200  # N*s/m
ball_bearing = rs.BearingElement(n=node_ball_bearing, kxx=k_roll, kyy=k_roll, cxx=c_roll, cyy=c_roll, tag="ball bearing")

# journal bearing
radius_shaft = d/2
radial_clearance = 0.0001
radius_journal_bearing = radius_shaft + radial_clearance  # 0.1 mm clearance
nz = 20
ntheta = 129            # MUST be odd
length_brg = 0.08
p_in = 4954.0  # Pa (rho * g * h) h = 0.505 m
p_out = 0.0
visc = 0.89e-3
rho = 997.0
load = 35.35  # N
rpm = 2800.0
omega = rpm * 2*np.pi/60  # [rad/s]
mod_sommerfeld_number = modified_sommerfeld_number(radius_journal_bearing, omega, visc, length_brg, load, radial_clearance)
sommer_number = sommerfeld_number(mod_sommerfeld_number, radius_journal_bearing, length_brg)
print("Modified Sommerfeld number: ", mod_sommerfeld_number, "\n")
print("Sommerfeld number", sommer_number, "\n")
fluid_film = flow.FluidFlow(nz=nz,
                            ntheta=ntheta,
                            length=length_brg,
                            omega=omega,
                            p_in=p_in,
                            p_out=p_out,
                            radius_rotor=radius_shaft,
                            radius_stator=radius_journal_bearing,
                            viscosity=visc,
                            density=rho,
                            load=load)
equilibrium_position = find_equilibrium_position(fluid_film, True)
K_journal_bearing, C_journal_bearing = calculate_stiffness_and_damping_coefficients(fluid_film)
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

# element of coupling (disk)
coupling = rs.DiskElement(n=node_disk, m=1.7, Id=0.001401842, Ip=0.00165659, tag="coupling")

# rotor creation
rotor = rs.Rotor(shaft_elements=shaft,
                 disk_elements=[coupling],
                 bearing_elements=[ball_bearing, journal_bearing])

print(rotor.nodes_pos)
fig_rotor= rotor.plot_rotor()
fig_rotor.write_html("rotor_geometry.html", auto_open=True)

# unbalance response
# Unbalance magnitude is m*e in kg*m.
# Conversion: (g*mm) -> (kg*m) multiply by 1e-6
# Example: 500 g*mm = 5e-4 kg*m
unb_magnitude = 3e-4   # [kg*m]  <-- adjust
unb_phase = 0.0        # [rad]

# run at a single frequency = omega (rad/s)
freq = np.array([omega], dtype=float)

results = rotor.run_unbalance_response(
    [unbalance_node],              # node
    [unb_magnitude],         # unbalance_magnitude
    [unb_phase],             # unbalance_phase
    freq                     # frequency
)

plot = results.plot_deflected_shape_3d(omega)
plot.show()
# Complex displacement response array: (ndof, nfreq)
U = results.forced_resp

# DOF mapping per node: 0=x, 1=y, 2=theta_x, 3=theta_y
dofx = rotor.number_dof * node_journal_bearing + 0
dofy = rotor.number_dof * node_journal_bearing + 1

X = U[dofx, 0]  # complex amplitude in x
Y = U[dofy, 0]  # complex amplitude in y


# Reconstruct orbit in time: x(t)=Re(X*e^(jωt)), y(t)=Re(Y*e^(jωt))
T = 2*np.pi / omega
n_cycles = 20
n_per_rev = 400

t = np.linspace(0, n_cycles*T, n_cycles*n_per_rev, endpoint=False)
ejwt = np.exp(1j * omega * t)

x_t = np.real(X * ejwt)
y_t = np.real(Y * ejwt)

#Plot orbit (ellipse)
fig, ax = plt.subplots()
ax.plot(x_t + 6.079055233197487e-05 ,y_t - 6.236860175127931e-05 )
#ax.plot(t, x_t)
# --- add circle ---
r_circle = 0.0001  # example: 50 micrometers
circle = Circle((0.0, 0.0), r_circle, fill=False, linewidth=2)  # center at (0,0)
ax.add_patch(circle)
# ------------------
ax.set_xlabel("x displacement [m]")
ax.set_ylabel("y displacement [m]")
ax.set_title(f"Journal node orbit at {rpm} rpm (unbalance excitation)")
ax.grid(True)
ax.set_aspect("equal", adjustable="box")  # prevents visual distortion
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
"""
