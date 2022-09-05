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

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from rigid_body_3d import RigidBody3DScheme
# from rigid_body_common import setup_damping_coefficient
from pysph.tools.geometry import get_2d_block, get_2d_tank


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
        spacing = 3.

        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = spacing * 1e-3
        self.dam_layers = 1
        self.dam_rho = 2000.

        self.body_length = 5. * 1e-2
        self.body_height = 5. * 1e-2
        self.body_spacing = self.dam_spacing
        self.body_rho = 2700

        # simulation properties
        self.hdx = 1.0
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.body_spacing

        # solver data
        self.tf = 0.5
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6

    def create_particles(self):
        # get body_id for each cylinder
        # create 3 bodies
        x1, y1 = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height,
                              center=[-1.5, 1])
        x2, y2 = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height,
                              center=[-1.5, 1])
        x2[:] += self.body_length/2.
        y2[:] -= self.body_height/2. + self.body_height
        x3, y3 = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height,
                              center=[-1.5, 1])
        x3[:] += 2. * self.body_length/2.
        y3[:] -= self.body_height/2. + 2.5 * self.body_height

        xb = np.concatenate((x1, x2, x3))
        yb = np.concatenate((y1, y2, y3))

        body_id_1 = np.ones(len(x1)) * 0
        body_id_2 = np.ones(len(x2)) * 1
        body_id_3 = np.ones(len(x3)) * 2
        body_id = np.concatenate((body_id_1, body_id_2, body_id_3))
        dem_id = body_id
        m = self.body_rho * self.body_spacing**2
        h = self.hdx * self.body_spacing
        rad_s = self.body_spacing / 2.
        bodies = get_particle_array(name='bodies',
                                    x=xb,
                                    y=yb,
                                    rho=self.body_rho,
                                    h=h,
                                    m=m,
                                    rad_s=rad_s,
                                    E=69 * 1e9,
                                    nu=0.3,
                                    constants={
                                        'spacing0': self.body_spacing,
                                    })
        bodies.add_property('dem_id', type='int', data=dem_id)
        bodies.add_property('body_id', type='int', data=body_id)
        bodies.add_constant('total_no_bodies', 4)

        # create dam with normals
        _xf, _yf, xd, yd = hydrostatic_tank_2d(
            self.dam_length, self.dam_height, self.dam_height, self.dam_layers,
            self.body_spacing, self.body_spacing)
        xd += min(bodies.x) - min(xd) - self.dam_spacing * self.dam_layers

        dam = get_particle_array(x=xd,
                                 y=yd,
                                 rho=self.body_rho,
                                 h=h,
                                 m=m,
                                 rad_s=self.dam_spacing / 2.,
                                 name="dam",
                                 E=30*1e8,
                                 nu=0.3)
        dam.y[:] += min(bodies.y) - min(dam.y) - self.body_height / 8.
        dam.x[:] -= self.body_length
        dam.add_property('dem_id', type='int', data=max(body_id) + 1)

        self.scheme.setup_properties([bodies, dam])

        # compute the boundary particles of the bodies
        bodies.add_property('contact_force_is_boundary')
        # is_boundary = self.get_boundary_particles(max(bodies.body_id)+1)
        bodies.contact_force_is_boundary[:] = bodies.is_boundary[:]
        # bodies.is_boundary[:] = is_boundary[:]

        dam.add_property('is_boundary', type='int')
        # set all particles to boundary if the boundary layers are only one
        dam.is_boundary[:] = 1
        dam.add_property('contact_force_is_boundary')
        dam.contact_force_is_boundary[:] = dam.is_boundary[:]

        # remove particles which are not used in computation
        indices = []
        for i in range(len(dam.x)):
            if dam.is_boundary[i] == 0:
                indices.append(i)

        dam.remove_particles(indices)

        # print(bodies.total_mass)

        return [bodies, dam]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['bodies'],
                                 boundaries=['dam'],
                                 gx=0.,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=2,
                                 fric_coeff=0.45)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    # app.create_particles()
    # app.geometry()
    app.run()
    # app.post_process(app.info_filename)
