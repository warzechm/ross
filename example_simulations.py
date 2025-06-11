import ross as rs
from ross.materials import steel
import numpy as np
import plotly.graph_objects as go

# Make sure the default renderer is set to 'notebook' for inline plots in Jupyter
import plotly.io as pio

pio.renderers.default = "browser"

def example_3():
    # This code is based on Example 3 - Isotropic Bearings, asymmetrical rotor example from ROSS
    # documentation accessed at: https://ross.readthedocs.io/en/latest/user_guide/example_3.html.
    # Classic Instantiation of the rotor
    shaft_elements = []
    bearing_seal_elements = []
    disk_elements = []

    for i in range(6):
        shaft_elements.append(rs.ShaftElement(L=0.25, material=steel, n=i, idl=0, odl=0.05))

    disk_elements.append(
        rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28)
    )

    disk_elements.append(
        rs.DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35)
    )
    bearing_seal_elements.append(rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, cyy=0))
    bearing_seal_elements.append(rs.BearingElement(n=6, kxx=1e6, kyy=1e6, cxx=0, cyy=0))

    rotor591c = rs.Rotor(
        shaft_elements=shaft_elements,
        bearing_elements=bearing_seal_elements,
        disk_elements=disk_elements,
    )

    rotor591c.plot_rotor()

    # From_section class method instantiation.
    bearing_seal_elements = []
    disk_elements = []
    shaft_length_data = 3 * [0.5]
    i_d = 3 * [0]
    o_d = 3 * [0.05]

    disk_elements.append(
        rs.DiskElement.from_geometry(n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28)
    )

    disk_elements.append(
        rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.35)
    )
    bearing_seal_elements.append(rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, cyy=0))
    bearing_seal_elements.append(rs.BearingElement(n=3, kxx=1e6, kyy=1e6, cxx=0, cyy=0))

    rotor591fs = rs.Rotor.from_section(
        brg_seal_data=bearing_seal_elements,
        disk_data=disk_elements,
        leng_data=shaft_length_data,
        idl_data=i_d,
        odl_data=o_d,
        material_data=steel,
    )
    rotor591fs.plot_rotor()

    # Obtaining results (wn is in rad/s)
    fig = rotor591c.run_campbell(np.linspace(0, 4000 * np.pi / 30, 50), frequencies=7).plot(
        frequency_units="rad/s"
    )
    fig.show()

    print("Normal Instantiation =", rotor591c.run_modal(speed=2000 * np.pi / 30).wn)
    print("\n")
    print("From Section Instantiation =", rotor591fs.run_modal(speed=2000 * np.pi / 30).wn)

    # Obtaining modal results for w=4000RPM (wn is in rad/s)
    speed = 4000 * np.pi / 30
    modal591c = rotor591c.run_modal(speed, num_modes=14)

    print("Normal Instantiation =", modal591c.wn)

def example_7():
    # This code is based on Example 7 - Hydrodynamic Bearings from the ROSS
    # documentation accessed at: https://ross.readthedocs.io/en/latest/user_guide/example_3.html.
    Q_ = rs.Q_
    # Classic Instantiation of the rotor
    shaft_elements = []
    disk_elements = []
    steel = rs.materials.steel
    for i in range(6):
        shaft_elements.append(rs.ShaftElement(L=0.25, material=steel, n=i, idl=0, odl=0.05))

    disk_elements.append(
        rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28)
    )

    disk_elements.append(
        rs.DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35)
    )

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

    # copy bearing to decrease compute time and set node
    bearing_copy = rs.BearingElement(
        n=3,
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

    bearing_seal_elements = [bearing, bearing_copy]

    rotor = rs.Rotor(
        shaft_elements=shaft_elements,
        bearing_elements=bearing_seal_elements,
        disk_elements=disk_elements,
    )

    rotor.plot_rotor()

    campbell = rotor.run_campbell(speed_range=Q_(list(range(0, 4500, 50)), "RPM"))

    # Obtaining results for w=4000RPM

    modal = rotor.run_modal(4000 * np.pi / 30)

    print("Normal Instantiation =", modal.wn / (2 * np.pi))

    # The input units must be according to your unit standard system
    campbell = rotor.run_campbell(np.linspace(0, 4000 * np.pi / 30, 50))
    # Plotting frequency in RPM
    campbell.plot(frequency_units="RPM")

    for mode in range(6):
        fig = modal.plot_mode_3d(mode, frequency_units="Hz")
        fig.show()

    for mode in range(6):
        fig = modal.plot_orbit(mode, nodes=[2, 4])
        fig.show()

if __name__ == "__main__":
    example_7()
