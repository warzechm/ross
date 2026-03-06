"""
Complete flexible rotor model (FEM shaft) with:
- rolling element bearings at both ends (linear K,C)
- hydrodynamic journal bearing at midspan (FluidFlow -> linearized K,C)
- housing compliance collapsed into an equivalent bearing to ground (series combination)
- synchronous ORBIT from unbalance response at 1500 rpm
- rotor visualization via rotor.plot_rotor() saved to HTML

Author: Mariusz Warzecha
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import ross as rs

# -----------------------
# Plotly renderer
# -----------------------
pio.renderers.default = "browser"  # or "notebook" in Jupyter

# =======================
# 1) Test rig / rotor data
# =======================
L_total = 1.0          # [m] shaft length
d = 0.020              # [m] shaft diameter (constant)
steel = rs.materials.steel

# FEM discretization of the shaft
N = 20                 # increase to 40-60 for refinement
L = L_total / N

shaft = [
    rs.ShaftElement(
        L=L,
        idl=0.0, idr=0.0,
        odl=d,  odr=d,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    for _ in range(N)
]

n_left = 0
n_right = N
n_mid = N // 2  # journal bearing location

# =======================
# 2) Rolling bearings at ends (linear K,C)
# =======================
k_roll = 1e7   # [N/m]  (replace with identified values if available)
c_roll = 200   # [N*s/m]

brg_left = rs.BearingElement(
    n=n_left, kxx=k_roll, kyy=k_roll, cxx=c_roll, cyy=c_roll, tag="rolling_left"
)
brg_right = rs.BearingElement(
    n=n_right, kxx=k_roll, kyy=k_roll, cxx=c_roll, cyy=c_roll, tag="rolling_right"
)

# =======================
# 3) Journal bearing (FluidFlow -> K,C)
# =======================
from ross.bearings import fluid_flow as flow
from ross.bearings.fluid_flow_coefficients import calculate_stiffness_and_damping_coefficients
from ross.bearings.fluid_flow_geometry import calculate_attitude_angle

radius_rotor = 0.0099
radius_stator = 0.0100

nz = 8
ntheta = 129            # MUST be odd
length_brg = 0.08
p_in = 0.0
p_out = 0.0
visc = 0.89e-3
rho = 997.0

rpm = 1500.0
omega = rpm * 2*np.pi/60  # [rad/s]

radial_clearance = radius_stator - radius_rotor
eps_ratio = 0.97
eccentricity = eps_ratio * radial_clearance
attitude_angle = calculate_attitude_angle(eps_ratio)

ff = flow.FluidFlow(
    nz, ntheta, length_brg, omega, p_in, p_out,
    radius_rotor, radius_stator,
    visc, rho,
    eccentricity=eccentricity,
    attitude_angle=attitude_angle,
)

K4, C4 = calculate_stiffness_and_damping_coefficients(ff)
kxx, kxy, kyx, kyy = K4
cxx, cxy, cyx, cyy = C4

# =======================
# 4) Housing compliance collapsed into equivalent bearing to ground (series stiffness)
# =======================
K_film = np.array([[kxx, kxy],
                   [kyx, kyy]], dtype=float)

C_film = np.array([[cxx, cxy],
                   [cyx, cyy]], dtype=float)

k_housing = 77000.0  # 1 mm -> 77 N
K_h = np.array([[k_housing, 0.0],
                [0.0, k_housing]], dtype=float)

# Series combination: K_eq = (K_film^-1 + K_h^-1)^-1
K_eq = np.linalg.inv(np.linalg.inv(K_film) + np.linalg.inv(K_h))

# Practical approximation: keep film damping
C_eq = C_film

journal_to_ground = rs.BearingElement(
    n=n_mid,
    kxx=K_eq[0, 0], kxy=K_eq[0, 1],
    kyx=K_eq[1, 0], kyy=K_eq[1, 1],
    cxx=C_eq[0, 0], cxy=C_eq[0, 1],
    cyx=C_eq[1, 0], cyy=C_eq[1, 1],
    tag="journal_eq_to_ground",
)

# =======================
# 5) Build rotor
# =======================
rotor = rs.Rotor(
    shaft_elements=shaft,
    disk_elements=[],
    bearing_elements=[brg_left, brg_right, journal_to_ground],
)

# =======================
# 6) Visualize rotor geometry (always works: write HTML)
# =======================
fig_rotor = rotor.plot_rotor(nodes=2)
fig_rotor.write_html("rotor_geometry.html", auto_open=True)

# =======================
# 7) UNBALANCE response (frequency-domain) at 1500 rpm
# =======================
# Unbalance magnitude is m*e in kg*m.
# Conversion: (g*mm) -> (kg*m) multiply by 1e-6
# Example: 500 g*mm = 5e-4 kg*m
unb_node = n_mid
unb_magnitude = 5e-4   # [kg*m]  <-- adjust
unb_phase = 0.0        # [rad]

# run at a single frequency = omega (rad/s)
freq = np.array([omega], dtype=float)

# Many versions of ROSS name args as below; keep positional to be robust:
results = rotor.run_unbalance_response(
    [unb_node],              # node
    [unb_magnitude],         # unbalance_magnitude
    [unb_phase],             # unbalance_phase
    freq                     # frequency
)

# Complex displacement response array: (ndof, nfreq)
U = results.forced_resp

# DOF mapping per node: 0=x, 1=y, 2=theta_x, 3=theta_y
dofx = rotor.number_dof * n_mid + 0
dofy = rotor.number_dof * n_mid + 1

X = U[dofx, 0]  # complex amplitude in x
Y = U[dofy, 0]  # complex amplitude in y

# =======================
# 8) Reconstruct orbit in time: x(t)=Re(X*e^(jωt)), y(t)=Re(Y*e^(jωt))
# =======================
T = 2*np.pi / omega
n_cycles = 20
n_per_rev = 400

t = np.linspace(0, n_cycles*T, n_cycles*n_per_rev, endpoint=False)
ejwt = np.exp(1j * omega * t)

x_t = np.real(X * ejwt)
y_t = np.real(Y * ejwt)

# =======================
# 9) Plot orbit (ellipse)
# =======================
plt.figure()
plt.plot(x_t, y_t)
plt.xlabel("x displacement [m]")
plt.ylabel("y displacement [m]")
plt.title("Journal node orbit at 1500 rpm (unbalance excitation)")
plt.grid(True)
plt.axis("equal")  # IMPORTANT: prevents visual distortion
plt.show()

# Optional: time histories
plt.figure()
plt.plot(t, x_t, label="x(t)")
plt.plot(t, y_t, label="y(t)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Journal node displacement vs time (reconstructed from complex response)")
plt.grid(True)
plt.legend()
plt.show()
