"""[1] Particle-Based Numerical Simulation Study of Solid Particle
Erosion of Ductile Materials Leading to an Erosion Model,
Including the Particle Shape Effect

https://doi.org/10.3390/ma15010286


3.3.2 Controlled sliding on a flat surface

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme
from pysph.sph.equation import Equation, Group

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


def setup_properties_for_gradual_force(pa):
    pa.add_constant('normal_force_time', 0.)
    pa.add_constant('tmp_normal_force', 0.)
    pa.add_constant('delta_fn', 0.)

    pa.add_constant('tangential_force_time', 0.)
    pa.add_constant('tmp_tangential_force', 0.)
    pa.add_constant('delta_ft', 0.)

    force_idx_fn = np.zeros_like(pa.x)
    force_idx_ft = np.zeros_like(pa.x)
    pa.add_property('force_idx_fn', type='int', data=force_idx_fn)
    pa.add_property('force_idx_ft', type='int', data=force_idx_ft)


def force_index_single(body):
    max_y = np.max(body.y)

    indices_fn = np.where(max_y == body.y)[0]
    body.force_idx_fn[indices_fn] = 1

    min_x = np.min(body.x)
    indices_ft = np.where(min_x == body.x)[0]
    body.force_idx_ft[indices_ft] = 1


class ApplyForceOnRigidBody(Equation):
    def initialize(self, d_idx,
                   d_m,
                   d_fx,
                   d_fy,
                   d_fz,
                   d_normal_force_time,
                   d_tmp_normal_force,
                   d_delta_fn,
                   d_force_idx_fn,
                   d_tangential_force_time,
                   d_tmp_tangential_force,
                   d_delta_ft,
                   d_force_idx_ft,
                   dt, t):
        t_1 = d_normal_force_time[0]

        if t <= t_1:
            if d_idx == 1:
                d_tmp_normal_force[0] += d_delta_fn[0]

        if d_force_idx_fn[d_idx] == 1:
            d_fy[d_idx] += d_tmp_normal_force[0]

        t_2 = d_normal_force_time[0] + d_tangential_force_time[0]
        if t > t_1 and t <= t_2:
            if d_idx == 1:
                d_tmp_tangential_force[0] += d_delta_ft[0]

        if d_force_idx_ft[d_idx] == 1:
            d_fx[d_idx] += d_tmp_tangential_force[0]


class Mohseni2021FreeSlidingOnASlope(Application):
    def initialize(self):
        self.dim = 2
        spacing = 1e-2

        self.wall_length = 100.
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 0
        self.wall_rho = 2000.

        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        self.angle = 30.

        # simulation properties
        self.hdx = 1.5
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.rigid_body_spacing

        # solver data
        self.tf = 3.
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

        # force application parameters
        self.fn = -2. * 1e3
        self.ft = 4. * 1e3

        self.normal_force_time = 0.5
        self.tangential_force_time = 0.5

        timesteps_fn = self.normal_force_time / self.dt
        self.fn_increment = self.fn / timesteps_fn

        timesteps_ft = self.tangential_force_time / self.dt
        self.ft_increment = self.ft / timesteps_ft

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

    def create_particles(self):
        # get bodyid for each rigid_body
        xc, yc, body_id = self.create_rigid_body()

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
                                            'initial_spacing0': self.rigid_body_spacing,
                                        })
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('max_tng_contacts_limit', 10)

        setup_properties_for_gradual_force(rigid_body)
        force_index_single(rigid_body)

        rigid_body.normal_force_time[0] = self.normal_force_time
        rigid_body.tmp_normal_force[0] = 0.
        rigid_body.delta_fn[0] = 0.

        rigid_body.tangential_force_time[0] = self.tangential_force_time
        rigid_body.tmp_tangential_force[0] = 0.

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_fn == 1.)[0])
        rigid_body.delta_fn[0] = self.fn_increment / no_par

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_ft == 1.)[0])
        rigid_body.delta_ft[0] = self.ft_increment / no_par

        # Create wall particles
        length_fac = 1.
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.wall_length * length_fac,
                            height=self.wall_layers * self.wall_spacing)
        contact_force_is_boundary = get_contact_force_is_boundary(x, y, self.rigid_body_spacing)

        # x, y, _z = rotate(x, y, np.zeros(len(x)), axis=np.array([0., 0., 1.]),
        #                   angle=-(90. - self.angle))

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.

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
        # remove particles outside the circle
        indices = []
        for i in range(len(wall.x)):
            if wall.x[i] < - self.rigid_body_length:
                indices.append(i)

        wall.remove_particles(indices)

        # move rigid body to up
        # rigid_body.y[:] += max(wall.y) - min(rigid_body.y) + self.rigid_body_spacing * 1.
        # rigid_body.x[:] -= abs(max(rigid_body.x) - min(wall.x))
        # rigid_body.x[:] += 2. * self.rigid_body_length

        xc, yc, _zs = rotate(rigid_body.x, rigid_body.y, rigid_body.z, axis=np.array([0., 0., 1.]),
                             angle=-self.angle)
        x, y, _z = rotate(wall.x, wall.y, wall.z, axis=np.array([0., 0., 1.]), angle=-30.)

        rigid_body.x[:] = xc[:]
        rigid_body.y[:] = yc[:]
        radians = (90. - self.angle) * np.pi / 180.
        rigid_body.x[:] += (self.rigid_body_length / 2. + self.rigid_body_spacing) * np.cos(radians)
        rigid_body.y[:] += (self.rigid_body_length / 2. + self.rigid_body_spacing) * np.sin(radians)

        wall.x[:] = x[:]
        wall.y[:] = y[:]

        # rigid_body.y[:] += - 0.0001
        # rigid_body.x[:] += self.rigid_body_spacing * 0.5

        self.scheme.setup_properties([rigid_body, wall])

        # self.scheme.scheme.set_linear_velocity(
        #     rigid_body, np.array([0.0, -0.5, 0.]))

        return [rigid_body, wall]

    def create_scheme(self):
        rfc = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                boundaries=['wall'],
                                dim=2,
                                gy=self.gy,
                                kr=1e8)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # # Apply external force
        # force_eqs = []
        # force_eqs.append(
        #     ApplyForceOnRigidBody(dest="rigid_body", sources=None))

        # eqns.groups[-1].insert(-2, Group(force_eqs))

        return eqns

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    # def post_step(self, solver):
    #     t = solver.t
    #     # dt = solver.dt
    #     # T = self.wall_time
    #     for pa in self.particles:
    #         if pa.name == 'rigid_body':
    #             t_1 = pa.normal_force_time[0]

    #             if t <= t_1:
    #                 pa.tmp_normal_force[0] += pa.delta_fn[0]
    #                 pa.fy[np.where(pa.force_idx_fn == 1)] += pa.tmp_normal_force[0]

    #             t_2 = pa.normal_force_time[0] + pa.tangential_force_time[0]
    #             if t > t_1 and t <= t_2:
    #                 pa.tmp_tangential_force[0] += pa.delta_ft[0]
    #                 pa.fx[np.where(pa.force_idx_ft == 1)] += pa.tmp_tangential_force[0]

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        import os
        from matplotlib import pyplot as plt

        info = self.read_info(fname)
        files = self.output_files

        data = load(files[0])
        arrays = data['arrays']
        t, velocity = [], []

        for sd, rb in iter_output(files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            vel = (rb.vcm[0]**2. + rb.vcm[1]**2. + rb.vcm[2]**2.)**0.5
            velocity.append(vel)

        # analytical data
        t_analytical = np.array([0., 2.989])
        v_analytical = np.array([0., 9.628])
        if self.options.fric_coeff == 0.2:
            t_analytical = np.array([0., 2.989])
            v_analytical = np.array([0., 9.628])

        elif self.options.fric_coeff == 0.4:
            t_analytical = np.array([0., 3.1393323271673674])
            v_analytical = np.array([0., 4.745746192173761])

        elif self.options.fric_coeff == np.tan(np.pi/6):
            t_analytical = np.array([0., 3.0])
            v_analytical = np.array([0., 0.])

        elif self.options.fric_coeff > np.tan(np.pi/6):
            t_analytical = np.array([0., 3.0])
            v_analytical = np.array([0., 0.])

        plt.clf()
        plt.plot(t, velocity, "-", label='Mohsen')
        plt.plot(t_analytical, v_analytical, "--", label='Analytical')

        plt.title('Velocity')
        plt.xlabel('t')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "velocity_vs_time.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Mohseni2021FreeSlidingOnASlope()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
