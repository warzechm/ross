# This will be my first approach to modelling a simple shaft supported by two sliding bearings.
# I want to test the capabilities of the ROSS library.

import ross as rs
from ross.materials import steel
import numpy as np
import math
import plotly.graph_objects as go
from ross.bearings.fluid_flow import FluidFlow
from ross.bearings.fluid_flow_graphics import (plot_pressure_surface, plot_pressure_theta,
    plot_eccentricity, plot_pressure_theta_cylindrical)
from ross.bearings.fluid_flow_coefficients import (
    calculate_oil_film_force,
    calculate_stiffness_and_damping_coefficients)
from ross.bearings import fluid_flow as flow
from ross.bearings.fluid_flow_geometry import (
    sommerfeld_number,
    modified_sommerfeld_number,
)
from ross.bearings.fluid_flow_geometry import internal_radius_function
import matplotlib.pyplot as mpt
from scipy.integrate import ode
from ross.bearings.fluid_flow_geometry import move_rotor_center, move_rotor_center_abs

# Make sure the default renderer is set to 'notebook' for inline plots in Jupyter
import plotly.io as pio

pio.renderers.default = "browser"

# GLOBAL VARIABLES
force_x_cached = 0
force_y_cached = 0
last_force_t = 0
dt_force = 0.0001
K = [0, 0, 0, 0]
C = [0, 0, 0, 0]
last_eccentricity = 0

def shaft_with_two_sliding_bearings():
    """Check possibility to time simulate the orbit in the bearing.
    """
    steel = rs.Material.load_material('Steel')
    shaft_half = rs.ShaftElement(L=0.5, material=steel, idl=0, odl=0.02)
    shaft_elements = [shaft_half, shaft_half]
    bearing = rs.BearingFluidFlow(
        n=0,
        nz=30,
        ntheta=20,
        length=0.03,
        omega=np.linspace(1, 5000, 5),
        p_in=0,
        p_out=0,
        radius_rotor=0.0199,
        radius_stator=0.02,
        visc=0.1,
        rho=860.0,
        load=525,
    )
    bearing_copy = rs.BearingElement(
        n=2,
        frequency=bearing.frequency,
        kxx=bearing.kxx,
        kxy=bearing.kxy,
        kyx=bearing.kyx,
        kyy=bearing.kyy,
        cxx=bearing.cxx,
        cxy=bearing.cxy,
        cyx=bearing.cyx,
        cyy=bearing.cyy,
    )

    bearing_elements = [bearing, bearing_copy]

    rotor = rs.Rotor(
        shaft_elements=shaft_elements,
        bearing_elements=bearing_elements
    )

    speed = 600.0
    time_samples = 1001
    node = 3
    t = np.linspace(0, 0.5, time_samples)
    
    F = np.zeros((time_samples, rotor.ndof))
    
    # component on direction
    F[:, 4 * node + 0] = 100
    # component on direction y
    F[:, 4 * node + 1] = 100
    
    response = rotor.run_time_response(speed, F, t)
    fig = response.plot_2d(node=0)
    fig.show()
    fig = rotor.plot_rotor()
    fig.show()


def pressure_distribution_plot():
    nz = 8
    ntheta = 128 # 64
    length = 0.03
    omega = 157.1 # 100.*2*np.pi/60
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.04
    radius_stator = 0.05
    viscosity = 0.1
    density = 860.
    eccentricity = 0.005
    attitude_angle = 0
    load = 525 # 100
    # remove attitude_angle and eccentricity; replace it by load
    bearing_pressure_distribution = FluidFlow(nz, ntheta, length,
                                              omega, p_in, p_out, radius_rotor,
                                              radius_stator, viscosity, density,
                                              attitude_angle=attitude_angle,
                                              eccentricity=eccentricity)
    fig = plot_pressure_surface(bearing_pressure_distribution)
    fig.show()
    fig = plot_pressure_theta(bearing_pressure_distribution, z=int(nz/2))
    fig.show()
    fig = plot_eccentricity(bearing_pressure_distribution)
    fig.show()
    fig = plot_pressure_theta_cylindrical(bearing_pressure_distribution, z=int(nz/2))
    fig.show()
    radial_force, tangential_force, force_x, force_y = calculate_oil_film_force(bearing_pressure_distribution)
    print("N=", radial_force)
    print("T=", tangential_force)
    print("fx=", force_x)
    print("fy=", force_y)

def hydrodynamic_journal_bearing_example_9():
    # Instantiating a Pressure Matrix
    nz = 8
    ntheta = 128
    length = 0.03
    omega = 157.1
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.0
    my_fluid_flow = flow.FluidFlow(
        nz,
        ntheta,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        load=load,
    )
    # Getting the eccentricity
    print(my_fluid_flow.eccentricity)
    # Calculating the modified sommerfeld number and the sommerfeld number
    
    modified_s = modified_sommerfeld_number(
	my_fluid_flow.radius_stator,
	my_fluid_flow.omega,
	my_fluid_flow.viscosity,
	my_fluid_flow.length,
	my_fluid_flow.load,
	my_fluid_flow.radial_clearance,
    )
    print(sommerfeld_number(modified_s, my_fluid_flow.radius_stator, my_fluid_flow.length))

    # Plotting the eccentricity
    fig = plot_eccentricity(my_fluid_flow, scale_factor=0.5)
    fig.show()

    # Getting the stiffness and damping matrices
    K, C = calculate_stiffness_and_damping_coefficients(my_fluid_flow)
    print(f"Kxx, Kxy, Kyx, Kyy = {K}")
    print(f"Cxx, Cxy, Cyx, Cyy = {C}")

    # Calculating pressure matrix
    print(my_fluid_flow.calculate_pressure_matrix_numerical()[int(nz / 2)])
    # Plotting pressure along theta in a chosen z
    fig = plot_pressure_theta(my_fluid_flow, z=int(nz / 2))
    fig.show()


def test_internal_radius_function():
    gamma = np.arange(0, 2*np.pi, 0.1)
    radius_rotor = 0.01
    smallest_radius = radius_rotor
    smallest_radius_gamma = 0
    attitude_angle = np.pi/2
    eccentricity = 0.001
    for i in gamma:
        (radius, xri, yri) = internal_radius_function(i, attitude_angle, radius_rotor, eccentricity)
        if radius < smallest_radius:
            smallest_radius = radius
            smallest_radius_gamma = i
        print(f"For angle gamma = {i}, the radius = {radius}.")
        if (np.pi / 2 + attitude_angle) < i < (3 * np.pi / 2 + attitude_angle):
            alpha = np.absolute(3 * np.pi / 2 - i + attitude_angle)
        else:
            alpha = i + np.pi / 2 - attitude_angle
        print(f"For angle gamma = {i}, the angle alfa is equal to {alpha}. \n")
    print(f"The smallest radius is equal {smallest_radius} and it occurs for gamma \
    {smallest_radius_gamma}")

def shift_grooves_from_teta_to_gamma(grooves : tuple, fluid_flow : FluidFlow) -> tuple:
    """ The teta angle is measured counterclockwise starting from -X axis.
    The fluid flow is calculated using gamma angle, therefore the grooves positions have to be adjusted.

    Parameters
    ----------
    grooves : tuple of tuples
        Angle coordinates of all grooves given in teta angle
    fluid_flow : FluidFlow
        FluidFlow object to get the sift between teta and gamma.

    Returns
    -------
    tuple
        Angle coordinates of all grooves given in gamma angle.
    
    Notes
    -----
    None.

    Examples
    --------
    To be added if necessary. 
    """
    shift_value = (np.pi / 2) + fluid_flow.attitude_angle
    return tuple((math.radians(start) + shift_value, math.radians(end) + shift_value) for start, end in grooves)


def check_if_angle_inside_groove(grooves : tuple, angle : float) -> bool:
    """ The teta angle is measured counterclockwise starting from -X axis.
    The fluid flow is calculated using gamma angle, therefore the grooves positions have to be adjusted.

    Parameters
    ----------
    grooves : tuple of tuples
        Angle coordinates of all grooves given in gamma angle.
    angle : float
        Value of gamma angle

    Returns
    -------
    bool
        True if angle is inside the groove.
    
    Notes
    -----
    None.

    Examples
    --------
    To be added if necessary. 
    """
    for start, end in grooves:
        if start <= end:
            if start <= angle <= end:
                return True
        else:
            if angle >= start or angle <= end:
                return True
    return False

def set_zero_pressure_in_grooves(grooves : tuple, fluid_flow : FluidFlow) -> FluidFlow:
    """ Set 0 pressure in grooves and return new FluidFlow object

    Parameters
    ----------
    grooves : tuple of tuples
        Angle coordinates of all grooves given in teta angle.
    fluid_flow : FluidFlow
        FluidFlow object storing pressure data

    Returns
    -------
    FluidFlow
        New object with change pressure profile.
    
    Notes
    -----
    None.

    Examples
    --------
    To be added if necessary. 
    """
    grooves_in_gamma = shift_grooves_from_teta_to_gamma(grooves, fluid_flow)
    #print(grooves_in_gamma)
    for i in range(0, fluid_flow.ntheta):
        if check_if_angle_inside_groove(grooves_in_gamma, fluid_flow.gama[0,i]):
            #print(f"Current angle is equal {fluid_flow.gama[0,i]} and the check results in it beeing in groove.")
            for j in range(0, fluid_flow.nz):
                fluid_flow.p_mat_numerical[j, i] = 0
    return fluid_flow


def calculate_stiffness_and_damping_coefficients_for_flow_with_grooves(grooves : tuple, fluid_flow_object : FluidFlow) -> FluidFlow:
    """This function calculates the bearing stiffness and damping matrices numerically.
    Before calculation it applies the effect of grooves given as input tuple.
    
    Parameters
    ----------
    grooves : tuple
        Tuple containing tuples of two numbers. Each tuple describes one groove (its angular beginning and end)
    fluid_flow_object: FluidFlow
        Object describing fluid flow.
    
    Returns
    -------
    Two lists of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy.
        And another list of length four including damping floats in this order: cxx, cxy, cyx, cyy.
        And

    Notes
    -----
    To be done.
    
    Examples
    --------
    To be done.
    """
    N = 6
    t = np.linspace(0, 2 * np.pi / fluid_flow_object.omegap, N)
    fluid_flow_object.xp = fluid_flow_object.radial_clearance * 0.0001
    fluid_flow_object.yp = fluid_flow_object.radial_clearance * 0.0001
    dx = np.zeros(N)
    dy = np.zeros(N)
    xdot = np.zeros(N)
    ydot = np.zeros(N)
    radial_force = np.zeros(N)
    tangential_force = np.zeros(N)
    force_xx = np.zeros(N)
    force_yx = np.zeros(N)
    force_xy = np.zeros(N)
    force_yy = np.zeros(N)
    X1 = np.zeros([N, 3])
    X2 = np.zeros([N, 3])
    F1 = np.zeros(N)
    F2 = np.zeros(N)
    F3 = np.zeros(N)
    F4 = np.zeros(N)

    for i in range(N):
        fluid_flow_object.t = t[i]

        delta_x = fluid_flow_object.xp * np.sin(
            fluid_flow_object.omegap * fluid_flow_object.t
        )
        move_rotor_center(fluid_flow_object, delta_x, 0)
        dx[i] = delta_x
        xdot[i] = (
            fluid_flow_object.omegap
            * fluid_flow_object.xp
            * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        )
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical(direction="x")
        fluid_flow_object = set_zero_pressure_in_grooves(grooves, fluid_flow_object)
        [
            radial_force[i],
            tangential_force[i],
            force_xx[i],
            force_yx[i],
        ] = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
        delta_y = fluid_flow_object.yp * np.sin(
            fluid_flow_object.omegap * fluid_flow_object.t
        )
        move_rotor_center(fluid_flow_object, -delta_x, 0)
        move_rotor_center(fluid_flow_object, 0, delta_y)
        dy[i] = delta_y
        ydot[i] = (
            fluid_flow_object.omegap
            * fluid_flow_object.yp
            * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        )
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical(direction="y")
        fluid_flow_object = set_zero_pressure_in_grooves(grooves, fluid_flow_object)
        [
            radial_force[i],
            tangential_force[i],
            force_xy[i],
            force_yy[i],
        ] = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
        move_rotor_center(fluid_flow_object, 0, -delta_y)
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical()

        X1[i] = [1, dx[i], xdot[i]]
        X2[i] = [1, dy[i], ydot[i]]
        F1[i] = -force_xx[i]
        F2[i] = -force_xy[i]
        F3[i] = -force_yx[i]
        F4[i] = -force_yy[i]

    P1 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X1), X1)), np.transpose(X1)), F1
    )
    P2 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), F2
    )
    P3 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X1), X1)), np.transpose(X1)), F3
    )
    P4 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), F4
    )

    K = [P1[1], P2[1], P3[1], P4[1]]
    C = [P1[2], P2[2], P3[2], P4[2]]

    return K, C


def test_pressure_in_grooves():
    # 8 grooves, each 17.2 deg in angular width 0, 45, 90, 135, 180, 225, 270, 315
    grooves = ((351.4, 8.6), (36.4, 53.6), (81.4, 98.6), (126.4, 143.6),
               (171.4, 188.6), (216.4, 233.6), (261.4, 278.6), (306.4, 323.6))
    nz = 8
    ntheta = 128
    length = 0.08
    omega = 157.1
    p_in = 0.0
    p_out = 0.0
    eccentricity = (0.05-0.0499)/4
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.0
    attitude_angle = np.pi*2
    my_fluid_flow = flow.FluidFlow(
        nz,
        ntheta,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=eccentricity,
        attitude_angle = attitude_angle,
        load=load,
    )
    radial_force, tangential_force, force_x, force_y = calculate_oil_film_force(my_fluid_flow)
    print("Forces before groove influence: \n")
    print("N=", radial_force)
    print("T=", tangential_force)
    print("fx=", force_x)
    print("fy=", force_y)
    # Getting the stiffness and damping matrices
    K, C = calculate_stiffness_and_damping_coefficients(my_fluid_flow)
    print(f"Stiffness before groove influence Kxx, Kxy, Kyx, Kyy = {K}")
    print(f"Damping before groove influence Cxx, Cxy, Cyx, Cyy = {C}")
    
    new_flow = set_zero_pressure_in_grooves(grooves, my_fluid_flow);
    radial_force, tangential_force, force_x, force_y = calculate_oil_film_force(new_flow)
    print("Forces after groove influence: \n")
    print("N=", radial_force)
    print("T=", tangential_force)
    print("fx=", force_x)
    print("fy=", force_y)
    # Getting the stiffness and damping matrices
    K, C = calculate_stiffness_and_damping_coefficients_for_flow_with_grooves(grooves, my_fluid_flow)
    print(f"Stiffness after groove influence Kxx, Kxy, Kyx, Kyy = {K}")
    print(f"Damping after groove influence Cxx, Cxy, Cyx, Cyy = {C}")

    fig = plot_pressure_surface(my_fluid_flow)
    fig.show()
    fig = plot_pressure_theta(my_fluid_flow, z=int(nz / 2))
    fig.show()
    fig = plot_eccentricity(my_fluid_flow);
    fig.show()


def calculate_attitude_angle_from_shaft_position(x : float, y : float) -> float:
    """ The attitude angle is defined by line connecting centre of shaft and sleeve. Because the
    sleeve centre is used as the centre of global coordinate system (always (0, 0)) the function
    takes position of the shaft centre and based on it calculates the angle, which is measured from
    negative Y axis.

    Parameters
    ----------
    x : float
        X coordinate of the shaft centre
    y : float
        Y coordinate of the shaft centre

    Returns
    -------
    float
        Attitude angle given in radians.

    Notes
    -----

    Examples
    --------
    To be added if neccessary.
    """
    if x == 0 and y == 0:
        return 0
    if x == 0 and y < 0:
        return 0
    if x == 0 and y > 0:
        return np.pi
    tangens_of_attitude_agle = y / x
    if x > 0 and y < 0:
        return -math.atan(tangens_of_attitude_agle)
    if x > 0 and y > 0:
        return math.atan(tangens_of_attitude_agle) + math.pi/2
    if x < 0 and y > 0:
        return math.atan(tangens_of_attitude_agle) + 3/2*math.pi
    if x < 0 and y < 0:
        return -math.atan(tangens_of_attitude_agle) + 2*math.pi
    

def rhs(t : float, x : np.array) -> np.array:
    """ Function defines system of differential equations to be solved by ODE integrator.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Examples
    --------
    To be added if neccessary
    """
    global force_x_cached
    global force_y_cached
    global last_force_t
    global dt_force
    load = -77  # this parameter may cause problems, as it is included in motion equation
    m = 2.47  # [kg], D = 0.02 m, length = 1 m, ro = 7850 kg/m^3
    if True: #(t - last_force_t) >= dt_force:
        omega = 20 * 2 * math.pi   # 20 Hz
        grooves = ((355, 5), (85, 95), (175, 185), (265, 275))
        nz = 8
        ntheta = 128
        length = 0.08
        p_in = 0.0
        p_out = 0.0
        print(f"{x[0]}, {x[2]}")
        eccentricity = math.sqrt(x[0]**2 + x[2]**2)
        print(eccentricity)
        radius_rotor = 0.0499
        radius_stator = 0.05
        visc = 0.89e-3
        rho = 997.0
        attitude_angle = calculate_attitude_angle_from_shaft_position(x[0], x[2])
        my_fluid_flow = flow.FluidFlow(
            nz,
            ntheta,
            length,
            omega,
            p_in,
            p_out,
            radius_rotor,
            radius_stator,
            visc,
            rho,
            eccentricity=eccentricity,
            attitude_angle = attitude_angle,
            load=load,
        )
        flow_with_grooves = set_zero_pressure_in_grooves(grooves, my_fluid_flow);
        print("Evaluate oil film parameters\n")
        radial_force, tangential_force, force_x, force_y = calculate_oil_film_force(flow_with_grooves)
        force_x_cached = force_x
        force_y_cached = force_y
        last_force_t = t
        print(f"Updated forces [{force_x}, {force_y}] at time {t}\n")
    x0_dot = x[1]
    x1_dot = (force_x_cached + load) / m
    x2_dot = x[3]
    x3_dot = force_y_cached / m
    return np.array([x0_dot, x1_dot, x2_dot, x3_dot])


def rhs2(t : float, x : np.array) -> np.array:
    """ Function defines system of differential equations to be solved by ODE integrator.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Examples
    --------
    To be added if neccessary
    """
    global K
    global C
    global last_force_t
    global dt_force
    global last_eccentricity
    load = -10  # this parameter may cause problems, as it is included in motion equation, was 77
    m = 2.47  # [kg], D = 0.02 m, length = 1 m, ro = 7850 kg/m^3
    if (t - last_force_t) >= dt_force:
        eccentricity = math.sqrt(x[0]**2 + x[2]**2)
        radius_rotor = 0.0499
        radius_stator = 0.05
        if eccentricity > 0.95*(radius_stator - radius_rotor):
            print("Contact between shaft and bearing. Use material data for stiffness")
            K = [-68.7e6 for x in K]
        else:
            omega = 20 * 2 * math.pi   # 20 Hz
            grooves = ((351.4, 8.6), (36.4, 53.6), (81.4, 98.6), (126.4, 143.6),
                       (171.4, 188.6), (216.4, 233.6), (261.4, 278.6), (306.4, 323.6))
            nz = 8
            ntheta = 128
            length = 0.08
            p_in = 0.0
            p_out = 0.0
            visc = 0.89e-3
            rho = 997.0
            attitude_angle = calculate_attitude_angle_from_shaft_position(x[0], x[2])
            my_fluid_flow = flow.FluidFlow(
                nz,
                ntheta,
                length,
                omega,
                p_in,
                p_out,
                radius_rotor,
                radius_stator,
                visc,
                rho,
                eccentricity=eccentricity,
                attitude_angle = attitude_angle)
            K, C = calculate_stiffness_and_damping_coefficients_for_flow_with_grooves(grooves, my_fluid_flow)
        last_force_t = t
        last_eccentricity = eccentricity
        print("Evaluate oil film parameters\n")
        print(f"Updated stiffness and damping at time {t}. K_xx = {K[0]}, C_xx = {C[0]}\n")
        
    x0_dot = x[1]
    x1_dot = (x[0]*K[0] + x[2]*K[1] - x[1]*C[0] - x[3]*C[1] + load) / m
    x2_dot = x[3]
    x3_dot = (x[0]*K[2] + x[2]*K[3] - x[1]*C[2] - x[3]*C[3]) / m
    return np.array([x0_dot, x1_dot, x2_dot, x3_dot])


def solve_ODE(func : callable, init_cond : np.array,
              time_steps : np.array, problem_size : int)  -> np.array:
    """ Takes system of ODE's and solves it for given initial conditions and time steps.
    Main computational cost.

    Parameters
    ----------
    func : callable
        Function which implements the system of differential equations.
    init_cond : np.array
        Array containing initial conditions of the system. Size of the array is determined by problem
        size. Both have to correspond to each other.
    time_steps : np.array
        Array containing time steps at which the system state will be obtained (it is mainly important
        from user perspecitve, as it determines the precision of obtained results. The integration
        allgorithm may subdived the steps, but only internally.
    problem_size : int
        Number determining problem size (or in other words unknowns)
    
    Returns
    -------
    np.array
        Array containing values of uknowns at each time step defined by function input
        array time_steps.
    
    Notes
    -----

    Examples
    --------
    To be added if neccessary.
    """
    global last_eccentricity
    # generate matrix for results
    x_all = np.zeros((time_steps.shape[0], problem_size))
    eq = ode(func)
    
    eq.set_integrator('dopri5', nsteps=10000)
    eq.set_initial_value(init_cond, 0)

    for i, t in enumerate(time_steps):
        if not eq.successful():
            print(f"Integrator fail at time {eq.t}")
            break
        x = eq.integrate(t)
        for j in range(problem_size):
            x_all[i][j] = x[j]
        print(f"Current time step is equal to {t} and last eccentricity {last_eccentricity}")

    return x_all


if __name__ == "__main__":
#    hydrodynamic_journal_bearing_example_9()
    # pressure_distribution_plot()
    # test_internal_radius_function()
#    test_pressure_in_grooves()
    init_cond = np.array([0, 0, 0, 0])
    time_discretisation = dt_force  #  [s]
    simulation_time = 1  #  [s]
    t = np.arange(time_discretisation, simulation_time, time_discretisation) 
    results = solve_ODE(rhs, init_cond, t, 4)
    fig = mpt.figure()
    axes = mpt.subplot()
    axes.plot(results[:, 0], results[:, 2])
    mpt.show()
    
