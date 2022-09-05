import numpy as np
import os
from math import cos, sin
import math

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
from solid_mech import SolidsScheme, SolidMechStep
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
from pysph.tools.geometry import get_3d_block, rotate

import matplotlib


def create_surface_particles_sphere(samples=100):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    sphere_rad = 25 * 1e-3

    for i in range(samples):
        y = 1. - (i / float(samples - 1.)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1. - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    # print("points", len(points))
    x_, y_, z_ = [], [], []
    for i in range(len(points)):
        x_.append(points[i][0] * sphere_rad)
        y_.append(points[i][1] * sphere_rad)
        z_.append(points[i][2] * sphere_rad)

    return np.asarray(x_), np.asarray(y_), np.asarray(z_)


class Vyas2021ReboundKinematics(Application):
    def initialize(self):
        # constants
        self.target_E = 70 * 1e9
        self.target_nu = 0.3
        self.target_rho = 2650.

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
        self.rigid_body_diameter = 50. * 1e-3
        self.rigid_body_height = 0.2
        self.rigid_body_length = 0.2
        self.rigid_body_density = 2650.

        # geometry
        self.target_length = 2. * self.rigid_body_diameter
        self.target_height = 0.1 * self.rigid_body_diameter
        # this is z dimension
        self.target_width = 2. * self.rigid_body_diameter

        self.tf = 2.5 * 1e-4

        self.c0 = np.sqrt(self.target_E / (3 * (1. - 2 * self.target_nu) * self.target_rho))
        self.pb = self.target_rho * self.c0**2.

        self.artificial_vis_alpha = 1.0
        self.artificial_vis_beta = 0.0

        self.seval = None

        # attributes for Sun PST technique
        self.u_f = 0.059
        self.u_max = self.u_f * self.c0

        # this is manually taken by running one simulation
        self.u_max = 50
        self.mach_no = self.u_max / self.c0

        # attributes for IPST technique
        self.ipst_max_iterations = 10
        self.dim = 3

        # print(self.spacing)
        self.hdx = 1.0
        self.h = self.hdx * 25 * 1e-3 / 30

        # boundary equations
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["target"], sources=["target", "target_wall"],
            boundaries=["target_wall"])

        self.boundary_equations = self.boundary_equations_1

        # print(self.boundary_equations)

    def add_user_options(self, group):
        group.add_argument("--samples",
                           action="store",
                           type=int,
                           dest="samples",
                           default=1000,
                           help="samples (default to 3000)")

        group.add_argument("--velocity",
                           action="store",
                           type=float,
                           dest="velocity",
                           default=5.,
                           help="Velocity (default to 5.)")

        group.add_argument("--angle",
                           action="store",
                           type=float,
                           dest="angle",
                           default=10.,
                           help="Angle (default to 10. degrees)")

    def consume_user_options(self):
        self.velocity = self.options.velocity
        self.angle = self.options.angle

        # find the spacing
        self.samples = self.options.samples
        xb, yb, zb = create_surface_particles_sphere(self.options.samples)

        x_76 = xb[76]
        y_76 = yb[76]
        z_76 = zb[76]
        xb[76] = 0.
        yb[76] = 0.
        zb[76] = 0.
        d = ((xb - x_76)**2. + (yb - y_76)**2. + (zb - z_76)**2.)**0.5
        self.spacing = min(d)
        # print("spacing is", self.spacing)

        self.rigid_body_spacing = self.spacing
        # print("spacing is ")
        # print(self.spacing)
        self.hdx = 1.0
        self.h = self.hdx * self.spacing

        # edac constants
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        self.dt = 0.25 * self.h / ((self.target_E / self.target_rho)**0.5 + 2.85)
        print("timestep is ", self.dt)

    def create_particles(self):
        x, y, z = get_3d_block(self.spacing, self.target_length,
                               self.target_height,
                               self.target_width)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        dx = self.spacing
        hdx = self.hdx
        m = self.target_rho * dx**self.dim
        h = np.ones_like(x) * hdx * dx
        rho = self.target_rho
        rad_s = self.spacing / 2.

        target = get_particle_array(x=x,
                                    y=y,
                                    z=z,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    rad_s=rad_s,
                                    E=self.target_E,
                                    nu=self.target_nu,
                                    rho_ref=self.target_rho,
                                    name="target",
                                    constants={
                                        'n': 4,
                                        'spacing0': self.spacing,
                                        'specific_heat': self.target_specific_heat,
                                        'jc_model': 1,
                                        'damage_model': 1
                                    })
        dem_id = np.ones(len(target.x), dtype=int) * 0.
        target.add_property('dem_id', type='int', data=dem_id)
        target.add_constant('total_no_bodies', [2])

        x, y, z = get_3d_block(self.spacing,
                               self.target_length + 6. * self.spacing,
                               self.target_height + 6. * self.spacing,
                               self.target_width + 6. * self.spacing)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        dx = self.spacing
        hdx = self.hdx
        m = self.target_rho * dx**self.dim
        h = np.ones_like(x) * hdx * dx
        rho = self.target_rho

        target_wall = get_particle_array(x=x,
                                         y=y,
                                         z=z,
                                         m=m,
                                         rho=rho,
                                         h=h,
                                         E=self.target_E,
                                         nu=self.target_nu,
                                         rho_ref=self.target_rho,
                                         name="target_wall",
                                         constants={
                                             'n': 4,
                                             'spacing0': self.spacing,
                                         })
        target_wall.y -= 3. * self.spacing
        remove_overlap_particles(target_wall, target, self.spacing/2.)

        indices = []
        min_y = min(target_wall.y)
        for i in range(len(target_wall.y)):
            if target_wall.y[i] < min_y + 2. * self.spacing:
                indices.append(i)
        target_wall.remove_particles(indices)

        # Create rigid body
        xb, yb, zb = create_surface_particles_sphere(self.samples)
        m = self.rigid_body_density * self.spacing**self.dim
        h = 1.0 * self.spacing
        rad_s = self.spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xb,
                                        y=yb,
                                        z=zb,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_density,
                                        rad_s=rad_s,
                                        constants={
                                            'E': 69 * 1e9,
                                            'poisson_ratio': 0.3,
                                            'spacing0': self.spacing,
                                        })
        dem_id = np.ones(len(rigid_body.x), dtype=int)
        body_id = np.zeros(len(rigid_body.x), dtype=int)
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('total_no_bodies', [2])

        # =======================================================
        # adjust the positions of the bodies before computing the
        # rigid body properties
        # =======================================================
        scale = abs(min(target.y) - max(rigid_body.y)) + 2. * self.spacing / 2.
        target.y[:] -= scale
        target_wall.y[:] -= scale

        self.scheme.setup_properties([target, target_wall, rigid_body])

        rigid_body.add_property('contact_force_is_boundary')
        rigid_body.is_boundary[:] = 1.
        rigid_body.contact_force_is_boundary[:] = 1.

        vel = self.velocity
        angle = self.angle / 180 * np.pi
        self.scheme.scheme.set_linear_velocity(
            rigid_body, np.array([vel * sin(angle),
                                  -vel * cos(angle), 0.]))

        # # remove particles which are not boundary
        # indices = []
        # for i in range(len(rigid_body.y)):
        #     if rigid_body.is_boundary[i] == 0:
        #         indices.append(i)
        # rigid_body.remove_particles(indices)

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

        # set the boundary particles for this particular example as, we are only
        # creating particles on the surface
        rigid_body.is_boundary[:] = 1.

        # =======================================
        # set the total mass, moment of inertia
        # =======================================
        radius = self.rigid_body_diameter / 2.
        rigid_body.total_mass[:] = 4. / 3. * np.pi * radius**3. * self.rigid_body_density
        I = np.zeros(9)
        I[0] = 2. / 5. * rigid_body.total_mass[:] * radius**2.
        I[4] = 2. / 5. * rigid_body.total_mass[:] * radius**2.
        I[8] = 2. / 5. * rigid_body.total_mass[:] * radius**2.
        I[1] = 0.
        I[2] = 0.
        I[3] = 0.
        I[5] = 0.
        I[6] = 0.
        I[7] = 0.
        rigid_body.inertia_tensor_body_frame[:] = I[:]

        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        rigid_body.inertia_tensor_inverse_body_frame[:] = I_inv[:]

        rigid_body.inertia_tensor_global_frame[:] = I[:]
        rigid_body.inertia_tensor_inverse_global_frame[:] = I_inv[:]
        # print("xcm is ")
        # print(rigid_body.xcm)
        # =======================================
        # set the total mass, moment of inertia
        # =======================================

        return [target, target_wall, rigid_body]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf

        output = np.array([0.0, 1.38 * 1e-3, 5.17 * 1e-3, 7.38 * 1e-3, 11.462 *
                           1e-3, 15.4 * 1e-3])

        pfreq = 30

        step = dict(target=SolidMechStep())

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq,
                                     output_at_times=output,
                                     extra_steppers=step)

    def create_scheme(self):
        solid = SolidsScheme(solids=['target'],
                             boundaries=['target_wall'],
                             rigid_bodies=['rigid_body'],
                             dim=self.dim,
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

        b = particle_arrays['rigid_body']
        b.scalar = 'vmag'
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        info = self.read_info(fname)
        output_files = self.output_files

        from pysph.solver.utils import iter_output

        t = []
        rad = self.options.angle * np.pi / 180
        non_dim_theta = [np.tan(rad) / self.options.fric_coeff]
        non_dim_omega = []

        for sd, rb in iter_output(output_files[-1:-2:-1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            omega = rb.omega[2]
            tmp = 0.5 * self.rigid_body_diameter * omega / self.options.fric_coeff
            non_dim_omega.append(tmp / (5. * cos(rad)))
        print(non_dim_omega)

        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # experimental data (read from file)
        # load the data
        thornton = np.loadtxt(
            os.path.join(directory, 'vyas_2021_rebound_kinematics_Thornton_omega_vs_theta.csv'),
            delimiter=',')
        theta_exp, omega_exp = thornton[:, 0], thornton[:, 1]

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 non_dim_theta=non_dim_theta,
                 non_dim_omega=non_dim_omega,

                 theta_exp=theta_exp,
                 omega_exp=omega_exp)

        plt.clf()
        plt.scatter(non_dim_theta, non_dim_omega, label='Simulated')
        plt.plot(theta_exp, omega_exp, '^-', label='Thornton')

        plt.title('Theta_vs_Omega')
        plt.xlabel('non dimensional theta')
        plt.ylabel('non dimensional Omega')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "omega_vs_theta.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Vyas2021ReboundKinematics()

    app.run()
    app.post_process(app.info_filename)
