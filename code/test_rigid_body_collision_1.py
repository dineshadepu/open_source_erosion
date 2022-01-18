from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate


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


def get_contact_force_is_boundary(x, y, spacing):
    """To understand this snipped please comment few lines and check the
    viewer to see the boundary particles
    """
    max_y = max(y)
    # bottom solid boundary
    indices_1 = (y > max_y - 0.5 * spacing)

    contact_force_is_boundary = np.ones_like(x) * 0.
    contact_force_is_boundary[indices_1] = 1.

    return contact_force_is_boundary


class RigidBodyNormallyImpactingWall(Application):
    def initialize(self):
        self.dim = 2
        spacing = 1e-2

        self.wall_length = 0.2
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 2
        self.wall_rho = 2000.

        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        # simulation properties
        self.hdx = 1.5
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.rigid_body_spacing

        # solver data
        self.tf = 0.2
        self.dt = 5e-5

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def create_particles(self):
        # get bodyid for each rigid_body
        xc, yc, body_id = self.create_rigid_body()
        # xc, yc, _zs = rotate(xc, yc, np.zeros(len(xc)), axis=np.array([0., 0., 1.]), angle=-10.)

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xc,
                                        y=yc,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        constants={
                                           'E': 69 * 1e9,
                                           'poisson_ratio': 0.3,
                                           'spacing0': self.rigid_body_spacing
                                        })
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('max_tng_contacts_limit', 10)

        # Create wall particles
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.wall_length,
                            height=self.wall_layers * self.wall_spacing)

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.

        contact_force_is_boundary = get_contact_force_is_boundary(x, y, self.rigid_body_spacing)

        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.rigid_body_rho,
                                  rad_s=rad_s,
                                  contact_force_is_boundary=contact_force_is_boundary,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })

        # move rigid body to up
        rigid_body.y[:] += max(wall.y) - min(rigid_body.y) + self.rigid_body_spacing * 3.
        rigid_body.x[:] += self.rigid_body_spacing * 0.5

        self.scheme.setup_properties([rigid_body, wall])

        # self.scheme.scheme.set_linear_velocity(
        #     rigid_body, np.array([0.0, -0.5, 0.]))

        return [rigid_body, wall]

    def create_scheme(self):
        rfc = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                boundaries=['wall'],
                                dim=2,
                                gy=self.gy)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def create_rigid_body(self):
        x = np.array([])
        y = np.array([])

        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id


if __name__ == '__main__':
    app = RigidBodyNormallyImpactingWall()
    # app.create_particles()
    # app.geometry()
    app.run()
