import numpy as np
import os
from math import cos, sin

# SPH equations
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import get_particle_array
from pysph.tools.geometry import (remove_overlap_particles)

# from solid_mech import SetHIJForInsideParticles, GraySolidsScheme
from solid_mech import SolidsScheme
from solid_mech_common import (setup_mie_gruniesen_parameters,
                               setup_johnson_cook_parameters,
                               setup_damage_parameters,
                               get_youngs_mod_from_G_nu,
                               get_youngs_mod_from_K_G,
                               get_poisson_ratio_from_E_G)

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from pysph.sph.scheme import add_bool_argument
from pysph.solver.utils import load, get_files
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_block, rotate

import matplotlib


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


class Vyas2022DeformationValidation(Application):
    def initialize(self):
        # constants
        self.E = 200 * 1e9
        self.nu = 0.29
        self.rho0 = 7870

        self.dx = 0.5 * 1e-3 / 2.
        self.hdx = 1.0
        self.h = self.hdx * self.dx

        # target mie-Gruneisen parameters
        self.target_mie_gruneisen_gamma = 1.97
        self.target_mie_gruneisen_S = 1.4

        # target properties
        self.target_JC_A = 553.1 * 1e6
        self.target_JC_B = 600.8 * 1e6
        self.target_JC_C = 0.0134
        self.target_JC_n = 0.234
        self.target_JC_m = 1.
        self.target_JC_T_melt = 1733
        self.target_specific_heat = 875

        # setup target damage parameters
        self.target_damage_1 = -0.77
        self.target_damage_2 = 1.45
        self.target_damage_3 = -0.47
        self.target_damage_4 = 0.
        self.target_damage_5 = 1.6

        # geometry
        self.rigid_body_length = 4.780 * 1e-3
        self.rigid_body_height = 4.780 * 1e-3
        self.rigid_body_diameter = 50 * 1e-3
        self.rigid_body_rho = 2000.
        self.rigid_body_spacing = self.dx

        # geometry
        self.target_length = 0.3 * self.rigid_body_diameter
        self.target_height = 0.1 * self.rigid_body_diameter

        self.dim = 2

        # compute the timestep
        self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        # self.dt = 1e-9

        self.tf = 3 * 1e-5

        self.c0 = np.sqrt(self.E / (3 * (1. - 2 * self.nu) * self.rho0))
        self.pb = self.rho0 * self.c0**2.

        self.artificial_vis_alpha = 1.0
        self.artificial_vis_beta = 0.0

        self.seval = None

        # edac constants
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # attributes for Sun PST technique
        self.u_f = 0.059
        self.u_max = self.u_f * self.c0

        # this is manually taken by running one simulation
        self.u_max = 50
        self.mach_no = self.u_max / self.c0

        # attributes for IPST technique
        self.ipst_max_iterations = 10

        # boundary equations
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["target"], sources=["target", "target_wall"],
            boundaries=["target_wall"])

        self.boundary_equations = self.boundary_equations_1

        # print(self.boundary_equations)

    def add_user_options(self, group):
        group.add_argument("--spacing",
                           action="store",
                           type=float,
                           dest="spacing",
                           default=0.5 * 1e-3,
                           help="Spacing (default to 0.5 * 1e-3)")

        group.add_argument("--velocity",
                           action="store",
                           type=float,
                           dest="velocity",
                           default=6.264,
                           help="Velocity (default to 6.264)")

        group.add_argument("--angle",
                           action="store",
                           type=float,
                           dest="angle",
                           default=90.,
                           help="Angle (default to 90. degrees)")

    def consume_user_options(self):
        self.velocity = self.options.velocity
        self.angle = self.options.angle

        self.dx = self.options.spacing
        self.rigid_body_spacing = self.dx
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        print("timestep is ", self.dt)

    def create_rigid_body(self):
        x, y = create_circle_1(diameter=self.rigid_body_diameter,
                               spacing=self.dx)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def create_particles(self):
        x, y = get_2d_block(self.dx, self.target_length, self.target_height)
        x = x.ravel()
        y = y.ravel()

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx * dx
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0
        rad_s = self.dx / 2.

        target = get_particle_array(x=x,
                                    y=y,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    rad_s=rad_s,
                                    E=self.E,
                                    nu=self.nu,
                                    rho_ref=self.rho0,
                                    name="target",
                                    constants={
                                        'n': 4,
                                        'spacing0': self.dx,
                                        'specific_heat': self.target_specific_heat,
                                        'jc_model': 1,
                                        'damage_model': 1
                                    })
        dem_id = np.ones(len(target.x), dtype=int) * 0.
        target.add_property('dem_id', type='int', data=dem_id)
        target.add_constant('total_no_bodies', [2])

        x, y = get_2d_block(self.dx, self.target_length + 6. * self.dx,
                            self.target_height + 6. * self.dx)
        x = x.ravel()
        y = y.ravel()

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx * dx
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0

        target_wall = get_particle_array(x=x,
                                         y=y,
                                         m=m,
                                         rho=rho,
                                         h=h,
                                         E=self.E,
                                         nu=self.nu,
                                         rho_ref=self.rho0,
                                         name="target_wall",
                                         constants={
                                             'n': 4,
                                             'spacing0': self.dx,
                                         })
        target_wall.y -= 3. * self.dx
        remove_overlap_particles(target_wall, target, self.dx/2.)

        indices = []
        min_y = min(target_wall.y)
        for i in range(len(target_wall.y)):
            if target_wall.y[i] < min_y + 2. * self.dx:
                indices.append(i)
        target_wall.remove_particles(indices)

        # Create rigid body
        xc, yc, body_id = self.create_rigid_body()
        xc, yc, _zs = rotate(xc, yc, np.zeros(len(xc)), axis=np.array([0., 0., 1.]), angle=-5.)
        yc += max(target.y) - min(yc) + 1.1 * self.dx
        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = 1.0 * self.dx
        rad_s = self.dx / 2.
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
                                            'spacing0': self.dx,
                                        })
        dem_id = np.ones(len(rigid_body.x), dtype=int)
        body_id = np.zeros(len(rigid_body.x), dtype=int)
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('total_no_bodies', [2])

        self.scheme.setup_properties([target, target_wall, rigid_body])

        rigid_body.add_property('contact_force_is_boundary')
        rigid_body.contact_force_is_boundary[:] = rigid_body.is_boundary[:]

        vel = self.velocity
        angle = self.angle / 180 * np.pi
        self.scheme.scheme.set_linear_velocity(
            rigid_body, np.array([vel * cos(angle),
                                  -vel * sin(angle), 0.]))

        # self.scheme.scheme.set_angular_velocity(
        #     rigid_body, np.array([0.0, 0.0, 10.]))
        # remove particles which are not boundary
        indices = []
        for i in range(len(rigid_body.y)):
            if rigid_body.is_boundary[i] == 0:
                indices.append(i)
        rigid_body.remove_particles(indices)

        # setup Mie Grunieson of state parameters
        setup_mie_gruniesen_parameters(
            pa=target, mie_gruneisen_sigma=self.target_mie_gruneisen_gamma,
            mie_gruneisen_S=self.target_mie_gruneisen_S)

        # setup the Johnson-Cook parameters
        setup_johnson_cook_parameters(
            pa=target,
            JC_A=self.target_JC_A,
            JC_B=self.target_JC_B,
            JC_C=self.target_JC_C,
            JC_n=self.target_JC_n,
            JC_m=self.target_JC_m,
            JC_T_melt=self.target_JC_T_melt)

        # setup damage parameters
        setup_damage_parameters(
            pa=target,
            damage_1=self.target_damage_1,
            damage_2=self.target_damage_2,
            damage_3=self.target_damage_3,
            damage_4=self.target_damage_4,
            damage_5=self.target_damage_5)

        return [target, target_wall, rigid_body]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf

        output = np.array([0.0, 1.38 * 1e-3, 5.17 * 1e-3, 7.38 * 1e-3, 11.462 *
                           1e-3, 15.4 * 1e-3])

        pfreq = 300

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq,
                                     output_at_times=output)

    def create_scheme(self):
        solid = SolidsScheme(solids=['target'],
                             boundaries=['target_wall'],
                             rigid_bodies=['rigid_body'],
                             dim=2,
                             pb=self.pb,
                             u_max=self.u_max,
                             mach_no=self.mach_no,
                             ipst_max_iterations=self.ipst_max_iterations,
                             h=self.h,
                             hdx=self.hdx,
                             artificial_vis_alpha=self.artificial_vis_alpha,
                             artificial_vis_beta=self.artificial_vis_beta,
                             kr=1e12)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def _make_accel_eval(self, equations, pa_arrays):
        if self.seval is None:
            kernel = self.scheme.scheme.kernel(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays,
                                 equations=equations,
                                 dim=self.dim,
                                 kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if solver.count % 10 == 0:
            t = solver.t
            dt = solver.dt

            arrays = self.particles
            a_eval = self._make_accel_eval(self.boundary_equations, arrays)

            # When
            a_eval.evaluate(t, dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['target']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = Vyas2022DeformationValidation()

    app.run()
    # app.post_process(app.info_filename)
