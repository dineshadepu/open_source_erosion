"""Simulation of solid-fluid mixture flow using moving particle methods
Shuai Zhang

TODO: 1. Fix the dam such that the bottom layer is y - spacing/2.
TODO: 2. Implement a simple 2d variant of rigid body collision.
"""

from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme

from pysph.tools.geometry import get_2d_block, get_2d_tank


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def create_circle(diameter=1, spacing=0.05, center=None):
    radius = diameter/2.
    xtmp, ytmp = get_2d_block(spacing, diameter+spacing, diameter+spacing)
    x = []
    y = []
    for i in range(len(xtmp)):
        dist = xtmp[i]**2. + ytmp[i]**2.
        if dist < radius**2:
            x.append(xtmp[i])
            y.append(ytmp[i])

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def hydrostatic_tank_2d(fluid_length, fluid_height, tank_height, tank_layers,
                        fluid_spacing, tank_spacing):
    xt, yt = get_2d_tank(dx=tank_spacing,
                         length=fluid_length + 2. * tank_spacing,
                         height=tank_height,
                         num_layers=tank_layers)
    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length,
                          height=fluid_height,
                          center=[-1.5, 1])

    xf += (np.min(xt) - np.min(xf))
    yf -= (np.min(yf) - np.min(yt))

    # now adjust inside the tank
    xf += tank_spacing * (tank_layers)
    yf += tank_spacing * (tank_layers)

    return xf, yf, xt, yt


class ZhangStackOfCylinders(Application):
    def initialize(self):
        self.dim = 2
        spacing = 1.

        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = spacing * 1e-3
        self.dam_layers = 2
        self.dam_rho = 2000.

        self.cylinder_radius = 1. / 2. * 1e-2
        self.cylinder_diameter = 1. * 1e-2
        self.cylinder_spacing = spacing * 1e-3
        self.cylinder_rho = 2700

        # simulation properties
        self.hdx = 0.5
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.cylinder_spacing

        # solver data
        self.tf = 0.1
        self.dt = 5e-5

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def create_particles(self):
        # get bodyid for each cylinder
        xc, yc, body_id = self.create_cylinders_stack_1()
        dem_id = body_id
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = 1.0 * self.cylinder_spacing
        rad_s = self.cylinder_spacing / 2.
        cylinders = get_particle_array(name='cylinders',
                                       x=xc,
                                       y=yc,
                                       h=h,
                                       m=m,
                                       rho=self.cylinder_rho,
                                       rad_s=rad_s,
                                       constants={
                                           'E': 69 * 1e9,
                                           'poisson_ratio': 0.3,
                                       })
        cylinders.add_property('dem_id', type='int', data=dem_id)
        cylinders.add_property('body_id', type='int', data=body_id)
        cylinders.add_constant('max_tng_contacts_limit', 10)

        self.scheme.setup_properties([cylinders])

        self.scheme.scheme.set_linear_velocity(
            cylinders, np.array([0.5, 0.0, 0., -0.5, 0.0, 0.]))

        return [cylinders]

    def create_scheme(self):
        rfc = RigidBody3DScheme(rigid_bodies=['cylinders'],
                                boundaries=None,
                                dim=2,
                                gy=0.)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def create_cylinders_stack_1(self):
        # create a row of six cylinders
        x = np.array([])
        y = np.array([])
        x_tmp1, y_tmp1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])

        x = np.concatenate((x, x_tmp1))
        y = np.concatenate((y, y_tmp1))

        x_tmp2, y_tmp2 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])
        x_tmp2 += 2. * self.cylinder_diameter

        x = np.concatenate((x, x_tmp2))
        y = np.concatenate((y, y_tmp2))

        body_id = np.array([], dtype=int)
        for i in range(2):
            b_id = np.ones(len(x_tmp1), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    # app.create_particles()
    # app.geometry()
    app.run()
