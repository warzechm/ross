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

class FluidFlowGrooves(fluid_flow.FluidFlow):
    """ Calculate the pressure matrix for a journal bearing with grooves.

    Parameters
    ----------
    grooves : tuple(tuple)
              A tuple containing angle pairs defining the groves. The positions are given using a coordinate around
              the journal circumference, in degrees. The coordinate system: X to the right, Y to the top. Angle measurement
              starts from positive X axis, counterclockwise.

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
    
