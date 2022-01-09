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

        self.wall_height = 20 * 1e-2
        self.wall_spacing = spacing * 1e-3
        self.wall_layers = 2
        self.wall_time = 0.5
        self.wall_rho = 2700

        # simulation properties
        self.hdx = 0.5
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.cylinder_spacing

        # solver data
        self.tf = 0.5 + self.wall_time
        self.dt = 5e-5

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def create_particles(self):
        # get bodyid for each cylinder
        xc, yc, body_id = self.create_cylinders_stack_1()
        dem_id = body_id
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_radius
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

        # create dam with normals
        _xf, _yf, xd, yd = hydrostatic_tank_2d(
            self.dam_length, self.dam_height, self.dam_height, self.dam_layers,
            self.dam_spacing, self.dam_spacing)
        xd += min(cylinders.x) - min(xd) - self.dam_spacing * self.dam_layers

        dam = get_particle_array(x=xd,
                                 y=yd,
                                 rad_s=self.dam_spacing / 2.,
                                 name="dam",
                                 constants={
                                     'E': 30 * 1e8,
                                     'poisson_ratio': 0.3,
                                 })
        dam.add_property('dem_id', type='int', data=max(body_id) + 1)

        # create wall with normals
        xw, yw = get_2d_block(
            self.wall_spacing,
            self.wall_spacing,
            self.wall_height / 4.
        )
        xw += max(cylinders.x) - min(xw) + self.wall_spacing
        yw += min(dam.y) - min(yw)
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  rad_s=self.wall_spacing / 2.,
                                  name="wall",
                                  constants={
                                      'E': 30 * 1e8,
                                      'poisson_ratio': 0.3,
                                  })
        wall.add_property('dem_id', type='int', data=max(body_id) + 2)

        self.scheme.setup_properties([cylinders, dam, wall])

        print(cylinders.inertia_tensor_body_frame[0:9])
        print(cylinders.total_mass)
        print(cylinders.xcm[0:2])

        # please run this function to know how
        # geometry looks like
        # from matplotlib import pyplot as plt
        # plt.scatter(cylinders.x, cylinders.y)
        # plt.scatter(dam.x, dam.y)
        # plt.scatter(wall.x, wall.y)
        # plt.axes().set_aspect('equal', 'datalim')
        # print("done")
        # plt.show()
        return [cylinders, dam, wall]

    def create_scheme(self):
        rfc = RigidBody3DScheme(rigid_bodies=['cylinders'],
                                boundaries=['dam', 'wall'],
                                dim=2,
                                gy=self.gy)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def create_cylinders_stack_1(self):
        # create a row of six cylinders
        x_six_1 = np.array([])
        y_six_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])
        for i in range(6):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter +
                                  self.cylinder_spacing/4.)
            x_six_1 = np.concatenate((x_six_1, x_tmp))
            y_six_1 = np.concatenate((y_six_1, y_tmp1))

        # create a row of five cylinders
        x_five_1 = np.array([])
        y_five_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                2. * self.cylinder_radius, self.cylinder_radius +
                self.cylinder_spacing + 2. * self.cylinder_spacing
            ])

        for i in range(5):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter +
                                  self.cylinder_spacing / 2.)
            x_five_1 = np.concatenate((x_five_1, x_tmp))
            y_five_1 = np.concatenate((y_five_1, y_tmp1))

        y_five_1 = y_five_1 + .78 * self.cylinder_diameter
        x_five_1 = x_five_1 - self.cylinder_spacing/2.

        # Create the third row from bottom, six cylinders
        x_six_2 = np.array(x_six_1, copy=True)
        y_six_2 = np.array(y_six_1, copy=True)
        y_six_2 += np.max(y_five_1) - np.min(y_six_1) + self.cylinder_spacing

        # Create the fourth row from bottom, five cylinders
        x_five_2 = np.array(x_five_1, copy=True)
        y_five_2 = np.array(y_five_1, copy=True)
        y_five_2 += np.max(y_six_2) - np.min(y_five_2) + self.cylinder_spacing

        # Create the third row from bottom, six cylinders
        x_six_3 = np.array(x_six_2, copy=True)
        y_six_3 = np.array(y_six_2, copy=True)
        y_six_3 += np.max(y_five_2) - np.min(y_six_3) + self.cylinder_spacing

        # Create the fourth row from bottom, five cylinders
        x_five_3 = np.array(x_five_2, copy=True)
        y_five_3 = np.array(y_five_2, copy=True)
        y_five_3 += np.max(y_six_3) - np.min(y_five_2) + self.cylinder_spacing

        x = np.concatenate((x_six_1, x_five_1, x_six_2, x_five_2,
                            x_six_3, x_five_3))
        y = np.concatenate((y_six_1, y_five_1, y_six_2, y_five_2,
                            y_six_3, y_five_3))

        # create body_id
        no_particles_one_cylinder = len(x_tmp)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def create_cylinders_stack(self):
        # create a row of six cylinders
        x_six_1 = np.array([])
        y_six_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])
        for i in range(6):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter +
                                  self.cylinder_spacing/4.)
            x_six_1 = np.concatenate((x_six_1, x_tmp))
            y_six_1 = np.concatenate((y_six_1, y_tmp1))

        # create a row of five cylinders
        x_five_1 = np.array([])
        y_five_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                2. * self.cylinder_radius, self.cylinder_radius +
                self.cylinder_spacing + 2. * self.cylinder_spacing
            ])

        for i in range(5):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter +
                                  self.cylinder_spacing / 2.)
            x_five_1 = np.concatenate((x_five_1, x_tmp))
            y_five_1 = np.concatenate((y_five_1, y_tmp1))

        y_five_1 = y_five_1 + .78 * self.cylinder_diameter
        x_five_1 = x_five_1 - self.cylinder_spacing/2.

        # Create the third row from bottom, six cylinders
        x_six_2 = np.array(x_six_1, copy=True)
        y_six_2 = np.array(y_six_1, copy=True)
        y_six_2 += np.max(y_five_1) - np.min(y_six_1) + self.cylinder_spacing

        # Create the fourth row from bottom, five cylinders
        x_five_2 = np.array(x_five_1, copy=True)
        y_five_2 = np.array(y_five_1, copy=True)
        y_five_2 += np.max(y_six_2) - np.min(y_five_2) + self.cylinder_spacing

        # Create the third row from bottom, six cylinders
        x_six_3 = np.array(x_six_2, copy=True)
        y_six_3 = np.array(y_six_2, copy=True)
        y_six_3 += np.max(y_five_2) - np.min(y_six_3) + self.cylinder_spacing

        # Create the fourth row from bottom, five cylinders
        x_five_3 = np.array(x_five_2, copy=True)
        y_five_3 = np.array(y_five_2, copy=True)
        y_five_3 += np.max(y_six_3) - np.min(y_five_2) + self.cylinder_spacing

        x = np.concatenate((x_six_1, x_five_1, x_six_2, x_five_2,
                            x_six_3, x_five_3))
        y = np.concatenate((y_six_1, y_five_1, y_six_2, y_five_2,
                            y_six_3, y_five_3))

        # create body_id
        no_particles_one_cylinder = len(x_tmp)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'wall':
                    pa.x += 0.25

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import os
        files = self.output_files
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files[::10], 'cylinders'):
            _t = sd['t']
            t.append(_t)
            # get the system center
            cm_x = 0
            cm_y = 0
            for i in range(array.nb[0]):
                cm_x += array.xcm[3 * i]
                cm_y += array.xcm[3 * i + 1]
            cm_x = cm_x / 33
            cm_y = cm_y / 33

            system_x.append(cm_x / self.dam_length)
            system_y.append(cm_y / self.dam_length)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        t = t - self.wall_time

        # load the data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        data = np.loadtxt(
            os.path.join(directory,
                         'x_com_zhang.csv'),
            delimiter=',')
        tx, xcom_zhang = data[:, 0], data[:, 1]

        plt.plot(tx, xcom_zhang, "s--", label='Experimental')
        plt.plot(t, system_x, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("x/L")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "xcom.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        data = np.loadtxt(
            os.path.join(directory,
                         'y_com_zhang.csv'),
            delimiter=',')
        ty, ycom_zhang = data[:, 0], data[:, 1]

        plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        plt.plot(t, system_y, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("y/L")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "ycom.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['cylinders']
        b.plot.actor.property.point_size = 2.
        ''')


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    # app.create_particles()
    # app.geometry()
    app.run()
    app.post_process(app.info_filename)
