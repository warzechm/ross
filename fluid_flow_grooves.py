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

    Date: 24.04.2026

    Description: Reimplementation of fluid flow modelling in journal bearing with grooves.
                 In this approach journal bearing geometry will be changed before the Reynolds equation is solved."""

from ross.bearings import fluid_flow
import numpy as np
from ross.bearings.fluid_flow_geometry import (external_radius_function, internal_radius_function)
from ross.bearings.fluid_flow_graphics import plot_shape, plot_eccentricity, plot_pressure_surface, plot_pressure_theta

class FluidFlowGrooves(fluid_flow.FluidFlow):
    """ Calculate the pressure matrix for a journal bearing with grooves.

    Parameters
    ----------
    grooves : tuple(tuple)
              A tuple containing angle pairs defining the groves. The positions are given using a coordinate around
              the journal circumference, in degrees. The coordinate system: X to the right, Y to the top. Angle measurement
              starts from positive X axis, counterclockwise.
    groove_depth : float
              Float number describing the groove depth with reference to the journal bearing cylindrical surface.
    shape_geometry: str, optional this parameter is extended 
        Determines the type of bearing geometry.
        'cylindrical': cylindrical bearing; 'eliptical': eliptical bearing;
        'wear': journal bearing wear.
        The default is 'cylindrical'.
        FluiFlowGrooves adds 'grooves' as valid shape_geometry
    Returns
    -------
    An object with fluid flow parameters for journal bearing with grooves. """

    def __init__(self,
                 nz,
                 ntheta,
                 length,
                 omega,
                 p_in,
                 p_out,
                 radius_rotor,
                 radius_stator,
                 viscosity,
                 density,
                 grooves,
                 groove_depth,
                 attitude_angle=None,
                 eccentricity=None,
                 load=None,
                 omegap=None,
                 immediately_calculate_pressure_matrix_numerically=False,
                 bearing_type=None,
                 shape_geometry="cylindrical",
                 preload=0.4,
                 displacement=0,
                 max_depth=None):
        self.grooves = grooves
        self.groove_depth = groove_depth
        super().__init__(nz,
                         ntheta,
                         length,
                         omega,
                         p_in,
                         p_out,
                         radius_rotor,
                         radius_stator,
                         viscosity,
                         density,
                         attitude_angle,
                         eccentricity,
                         load,
                         omegap,
                         immediately_calculate_pressure_matrix_numerically,
                         bearing_type,
                         shape_geometry,
                         preload,
                         displacement,
                         max_depth)
        
    def rotate_grooves(self, angle):
        """Rotate the groove positions by a given angle. Operates on grooves stored in class.

        Parameters
        ----------
        angle : float
            A float number giving the rotation angle in radians (0, 2*pi)

        Returns
        Tuple(tuple)
            Tuple of tuples with positions of rotated grooves given in radians.
        """
        new_groove_positions = []
        for (a, b) in self.grooves:
            a_new = a * np.pi / 180 + angle
          #  if a_new > 2 * np.pi:
          #      a_new -= 2 * np.pi
            b_new = b * np.pi / 180 + angle
          #  if b_new > 2 * np.pi:
          #      b_new -= 2 * np.pi
            new_groove_positions.append((a_new, b_new))
        return tuple(new_groove_positions)

    def geometry_description(self):
        """Overloaded function to get the grooves inside the journal bearing. It calculates internal
        and external radii.
        """
        if self.shape_geometry == "cylindrical":
            start = (np.pi / 2) + self.attitude_angle
        else:
            start = 0
        grooves = self.rotate_grooves(0)

        for i in range(0, self.nz):
            zno = i * self.dz
            self.z_list[i] = zno
            for j in range(0, self.ntheta):
                # fmt: off
                self.gama[i, j] = j * self.dtheta + start
                [radius_external, self.xre[i, j], self.yre[i, j]] = \
                    external_radius_function(self.gama[i, j], self.radius_stator, self.radius_rotor,
                                             shape=self.shape_geometry, preload=self.preload,
                                             displacement=self.displacement, max_depth=self.max_depth,
                                             grooves=grooves, groove_depth=self.groove_depth, print_info=False)
                [radius_internal, self.xri[i, j], self.yri[i, j]] = \
                    internal_radius_function(self.gama[i, j], self.attitude_angle, self.radius_rotor,
                                             self.eccentricity)
                self.re[i, j] = radius_external
                self.ri[i, j] = radius_internal

if __name__ == "__main__":
    # journal bearing
    d = 0.02
    radius_shaft = d/2
    radial_clearance = 0.0001
    radius_journal_bearing = radius_shaft + radial_clearance  # 0.1 mm clearance
    nz = 16
    ntheta = 519            # MUST be odd
    length_brg = 0.08
    p_in = 4954  # Pa (rho * g * h) h = 0.505 m
    p_out = 0.0
    visc = 0.89e-3*10
    rho = 997.0
    load = 35.35*0.1  # N
    rpm = 2800.0
    omega = rpm * 2*np.pi/60  # [rad/s]
    grooves = ((351.4, 8.6), (36.4, 53.6), (81.4, 98.6), (126.4, 143.6),
               (171.4, 188.6), (216.4, 233.6), (261.4, 278.6), (306.4, 323.6))
    fluid_film = FluidFlowGrooves(nz=nz,
                                  ntheta=ntheta,
                                  length=length_brg,
                                  omega=omega,
                                  p_in=p_in,
                                  p_out=p_out,
                                  grooves=grooves,
                                  groove_depth=5,
                                  shape_geometry="grooves",
                                  radius_rotor=radius_shaft,
                                  radius_stator=radius_journal_bearing,
                                  viscosity=visc,
                                  density=rho,
                                  load=load)
    fluid_film.calculate_pressure_matrix_numerical()
    fig = plot_eccentricity(fluid_film)
    fig.show()
    fig = plot_pressure_theta(fluid_film,z=7)
    fig.show()
    fig = plot_pressure_surface(fluid_film)
    fig.show()
