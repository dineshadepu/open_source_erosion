# python code/cao_xuerui_2022_spherical_particle_impact_3d.py --openmp --max-s 1 --cache-nnps --scheme=solid --detail --pfreq=500 --kr=100000000.0 --kf=1000000.0 --fric-coeff=0.1 --vel-alpha=45.0 --vel-magn=10.0 --samples=30000 --tf=5e-05
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
from pysph.tools.geometry import get_3d_block

import matplotlib


def create_surface_particles_sphere(samples=100):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    sphere_rad = 2.5 * 1e-3

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


class CaoXuerui2022SphericalParticleImpact3D(Application):
    def initialize(self):
        # constants
        self.E = 70 * 1e9
        self.nu = 0.33
        self.rho0 = 2700

        self.dx = 100 * 1e-6
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        self.rigid_body_spacing = self.dx

        # target mie-Gruneisen parameters
        self.target_mie_gruneisen_gamma = 1.97
        self.target_mie_gruneisen_S = 1.40

        # target properties
        self.target_JC_A = 335 * 1e6
        self.target_JC_B = 85 * 1e6
        self.target_JC_C = 0.012
        self.target_JC_n = 0.11
        self.target_JC_m = 1.0
        self.target_JC_T_melt = 925
        self.target_specific_heat = 885

        # setup target damage parameters
        self.target_damage_1 = -0.77
        self.target_damage_2 = 1.45
        self.target_damage_3 = -0.47
        self.target_damage_4 = 0.
        self.target_damage_5 = 1.6

        # geometry
        self.rigid_body_diameter = 5. * 1e-3
        self.rigid_body_rho = 7850.

        # geometry
        self.target_length_factor = 1.0
        self.target_length = 70 * 50. * 1e-6
        self.target_height = 15 * 50. * 1e-6
        # z-direction, out of the screen
        self.target_depth = 70 * 50. * 1e-6

        self.dim = 3

        self.tf = 4. * 1e-5

        self.c0 = np.sqrt(self.E / (3 * (1. - 2 * self.nu) * self.rho0))
        self.pb = self.rho0 * self.c0**2.

        self.artificial_vis_alpha = 1.0
        self.artificial_vis_beta = 0.0

        self.seval = None

        # attributes for Sun PST technique
        u_max = 1.  # manually coded from the simulation output

        # this is manually taken by running one simulation
        self.u_max = 5. * u_max
        self.mach_no = self.u_max / self.c0

        # attributes for IPST technique
        self.ipst_max_iterations = 10

        # boundary equations
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["target"], sources=["target"])

        self.boundary_equations = self.boundary_equations_1

        # print(self.boundary_equations)

    def add_user_options(self, group):
        # we don't use this in this example
        group.add_argument("--azimuth-theta",
                           action="store",
                           type=float,
                           dest="azimuth_theta",
                           default=-5,
                           help="Angle at which the square body is rotated")

        group.add_argument("--vel-alpha",
                           action="store",
                           type=float,
                           dest="vel_alpha",
                           default=45.,
                           help="Angle at which the velocity vector is pointed")

        group.add_argument("--vel-magn",
                           action="store",
                           type=float,
                           dest="vel_magn",
                           default=18.,
                           help="Magnitude of the impactor velocity")

        group.add_argument("--omega-magn",
                           action="store",
                           type=float,
                           dest="omega_magn",
                           default=1280.,
                           help="Magnitude of the angular velocity")

        group.add_argument("--spacing",
                           action="store",
                           type=float,
                           dest="spacing",
                           default=100,
                           help="Spacing in micrometers")

        group.add_argument("--samples",
                           action="store",
                           type=int,
                           dest="samples",
                           default=20000,
                           help="samples (default to 3000)")

        group.add_argument("--target-length-factor",
                           action="store",
                           type=float,
                           dest="target_length_factor",
                           default=1.,
                           help="Increase the target length with the given factor (default: 1.0)")

    def consume_user_options(self):
        # self.nu = self.options.nu
        self.vel_magn = self.options.vel_magn
        self.omega_magn = self.options.omega_magn
        self.vel_alpha = self.options.vel_alpha
        self.target_length_factor = self.options.target_length_factor
        # print("impact angle", self.vel_alpha)

        # ================================
        # find the spacing
        # ================================
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

        self.dx = self.spacing
        print("spacing is", self.dx*1e6)
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        self.rigid_body_spacing = self.dx

        # edac constants
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # compute the timestep
        # self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        self.dt = 3e-9
        print("timestep is ", self.dt)

    def create_particles(self):
        length = self.target_length_factor * self.target_length
        x, y, z = get_3d_block(dx=self.spacing,
                               length=length + 6. * self.dx,
                               height=self.target_height + 6. * self.dx,
                               depth=self.target_length + 6. * self.dx)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        # Move the target to the right
        x += (self.target_length_factor - 1.) / 2. * self.target_length

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx**self.dim
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0
        rad_s = self.dx / 2.

        target = get_particle_array(x=x,
                                    y=y,
                                    z=z,
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

        indices = []
        min_y = min(target.y)
        min_x = min(target.x)
        max_x = max(target.x)
        min_z = min(target.z)
        max_z = max(target.z)
        for i in range(len(target.y)):
            if target.y[i] < min_y + 3. * self.dx:
                indices.append(i)
            elif target.x[i] < min_x + 3. * self.dx:
                indices.append(i)
            elif target.x[i] > max_x - 3. * self.dx:
                indices.append(i)
            elif target.z[i] < min_z + 3. * self.dx:
                indices.append(i)
            elif target.z[i] > max_z - 3. * self.dx:
                indices.append(i)

        # add static particle data
        target.add_property('is_static')
        target.is_static[:] = 0.
        target.is_static[indices] = 1.

        # ==================================
        # Create rigid body
        # ==================================
        xc, yc, zc = create_surface_particles_sphere(self.samples)
        yc += max(target.y) - min(yc) + 1.1 * self.dx
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = 1.0 * self.dx
        rad_s = self.dx / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xc,
                                        y=yc,
                                        z=zc,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        constants={
                                            'E': 69 * 1e9,
                                            'poisson_ratio': 0.3,
                                            'spacing0': self.dx,
                                        })

        # ========================================
        # remove particles which are not boundary
        # ========================================
        min_y = min(rigid_body.y)
        indices = []
        for i in range(len(rigid_body.y)):
            if rigid_body.y[i] > min_y + 10. * 50. * 1e-6:
                indices.append(i)
        rigid_body.remove_particles(indices)

        dem_id = np.ones(len(rigid_body.x), dtype=int)
        body_id = np.zeros(len(rigid_body.x), dtype=int)
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('total_no_bodies', [2])

        self.scheme.setup_properties([target, rigid_body])

        self.vel_magn = self.options.vel_magn
        self.omega_magn = self.options.omega_magn

        angle = self.vel_alpha / 180 * np.pi
        self.scheme.scheme.set_linear_velocity(
            rigid_body, np.array([self.vel_magn * cos(angle),
                                  -self.vel_magn * sin(angle), 0.]))
        self.scheme.scheme.set_angular_velocity(
            rigid_body, np.array([0., 0., self.omega_magn]))

        # set the boundary particles for this particular example as, we are only
        # creating particles on the surface
        rigid_body.is_boundary[:] = 1.
        rigid_body.add_property('contact_force_is_boundary')
        rigid_body.contact_force_is_boundary[:] = rigid_body.is_boundary[:]

        # =======================================
        # set the total mass, moment of inertia
        # =======================================
        radius = self.rigid_body_diameter / 2.
        rigid_body.total_mass[:] = 4. / 3. * np.pi * radius**3. * self.rigid_body_rho
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
        # =======================================
        # set the total mass, moment of inertia
        # =======================================

        # =======================================
        # setup Mie Grunieson of state parameters
        # =======================================
        setup_mie_gruniesen_parameters(
            pa=target, mie_gruneisen_sigma=self.target_mie_gruneisen_gamma,
            mie_gruneisen_S=self.target_mie_gruneisen_S)

        # =======================================
        # setup the Johnson-Cook parameters
        # =======================================
        setup_johnson_cook_parameters(
            pa=target,
            JC_A=self.target_JC_A,
            JC_B=self.target_JC_B,
            JC_C=self.target_JC_C,
            JC_n=self.target_JC_n,
            JC_m=self.target_JC_m,
            JC_T_melt=self.target_JC_T_melt)

        # =======================================
        # setup damage parameters
        # =======================================
        setup_damage_parameters(
            pa=target,
            damage_1=self.target_damage_1,
            damage_2=self.target_damage_2,
            damage_3=self.target_damage_3,
            damage_4=self.target_damage_4,
            damage_5=self.target_damage_5)
        # target.yield_stress[:] = target.JC_A[0] * 4.

        return [target, rigid_body]

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
                             rigid_bodies=['rigid_body'],
                             dim=self.dim,
                             h=self.h,
                             hdx=self.hdx,
                             artificial_vis_alpha=self.artificial_vis_alpha,
                             artificial_vis_beta=self.artificial_vis_beta,
                             gy=0.,  # No change in the result
                             kr=1e12,
                             mach_no=self.mach_no,
                             u_max=self.u_max)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def _make_accel_eval(self, equations, pa_arrays):
        if self.seval is None:
            kernel = QuinticSpline(dim=self.dim)
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
        if self.options.use_ctvf:
            if solver.count % 10 == 0:
                t = solver.t
                dt = solver.dt

                arrays = self.particles
                a_eval = self._make_accel_eval(self.boundary_equations, arrays)

                # When
                a_eval.evaluate(t, dt)

    # def post_step(self, solver):
    #     for pa in self.particles:
    #         if pa.name == 'target':
    #             damaged = np.where(pa.is_damaged == 1.)[0]
    #             pa.remove_particles(damaged)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['target']
        b.scalar = 'vmag'
        ''')

    def get_boundary_particles_indices(self, x, y):
        from boundary_particles import (
            get_boundary_identification_etvf_equations,
            add_boundary_identification_properties)
        h = self.h
        pa = get_particle_array(x=x, y=y, m=1., h=h, rho=self.rho0, name="body")

        add_boundary_identification_properties(pa)
        # make sure your rho is not zero
        equations = get_boundary_identification_etvf_equations([pa.name],
                                                               [pa.name])
        # print(equations)

        sph_eval = SPHEvaluator(arrays=[pa],
                                equations=equations,
                                dim=2,
                                kernel=QuinticSpline(dim=2))

        sph_eval.evaluate(dt=0.1)
        return pa


    def post_process(self, fname):
        import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # from pysph.solver.utils import load, get_files
        # from pysph.solver.utils import load

        # info = self.read_info(fname)
        # output_files = self.output_files

        # from pysph.solver.utils import iter_output

        # t = []

        data = load(self.output_files[0])
        particle_arrays = data['arrays']
        # solver_data = data['solver_data']

        target = particle_arrays['target']
        x_0 = target.x
        y_0 = target.y
        z_0 = target.z

        data = load(self.output_files[-1])
        particle_arrays = data['arrays']
        # solver_data = data['solver_data']

        target = particle_arrays['target']

        displacement = y_0 - target.y

        disp_neg = []
        for val in displacement:
            # if val > 0.:
            disp_neg.append(val)

        deformation_index = np.where(disp_neg == max(disp_neg))[0]
        penetration_current = max(disp_neg)*1e6
        print("Max displacement is", penetration_current)

        indices = np.where(z_0 == z_0[deformation_index])
        x_deformed = target.x[indices]
        y_deformed = target.y[indices]
        z_deformed = target.z[indices]
        pa = self.get_boundary_particles_indices(x_deformed,
                                                 y_deformed)
        boundary_indices = np.where(pa.is_boundary == 1)
        x_deformed_surface = x_deformed[boundary_indices]
        y_deformed_surface = y_deformed[boundary_indices]

        indices = []
        for i in range(len(x_deformed_surface)):
            if y_deformed_surface[i] > 0.:
                # if x_deformed_surface[i] > -0.0015 and x_deformed_surface[i] < 0.0015:
                indices.append(i)

        plt.title('Particle plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x_deformed_surface[indices]*1e6, y_deformed_surface[indices]*1e6)
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.legend()
        # plt.ylim(450, 570)
        fig = os.path.join(os.path.dirname(fname), "topology.png")
        plt.savefig(fig, dpi=300)
        # plt.show()
        # # ========================
        # # x amplitude figure
        # # ========================

        # Save the data
        # Numerical data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        data_exp = np.loadtxt(
            os.path.join(directory, 'cao_xuerui_2022_spherical_particle_impact_3d_exp_penetration_depth_vs_incident_velocity.csv'),
            delimiter=',')

        data_sim = np.loadtxt(
            os.path.join(directory, 'cao_xuerui_2022_spherical_particle_impact_3d_numerical_penetration_depth_vs_incident_velocity.csv'),
            delimiter=',')

        vel_exp, penetration_exp = data_exp[:, 0], data_exp[:, 1]
        vel_sim, penetration_sim = data_sim[:, 0], data_sim[:, 1]
        self.vel_magn = self.options.vel_magn
        self.omega_magn = self.options.omega_magn
        self.vel_alpha = self.options.vel_alpha

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 vel_exp=vel_exp,
                 penetration_exp=penetration_exp,
                 vel_sim=vel_sim,
                 penetration_sim=penetration_sim,
                 vel_current=[self.vel_magn],
                 penetration_current=[penetration_current])

        # ========================
        # x amplitude figure
        # ========================
        plt.clf()
        plt.plot([self.options.vel_magn], [penetration_current], "-+", label='current')
        plt.plot(vel_exp, penetration_exp, "-^", label='exp')
        plt.plot(vel_sim, penetration_sim, "-o", label='FEM')

        plt.title('Penetration vs velocity')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Penetration depth (micrometers)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "penetration_vs_velocity.pdf")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = CaoXuerui2022SphericalParticleImpact3D()

    app.run()
    app.post_process(app.info_filename)
