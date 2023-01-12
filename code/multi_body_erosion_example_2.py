"""
python multi_body_erosion_example_2.py --openmp --kr 1e11
"""
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


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


class Dong2016CaseA1SquareParticleOnAl6061T6(Application):
    def initialize(self):
        # constants
        self.G = 26 * 1e9
        self.nu = 0.33
        self.E = get_youngs_mod_from_G_nu(self.G, self.nu)
        self.rho0 = 2800

        self.dx = 100 * 1e-6
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        self.rigid_body_spacing = self.dx

        # target mie-Gruneisen parameters
        self.target_mie_gruneisen_gamma = 2.17
        self.target_mie_gruneisen_S = 1.49

        # target properties
        self.target_JC_A = 324 * 1e6
        self.target_JC_B = 114 * 1e6
        self.target_JC_C = 0.42
        self.target_JC_n = 0.002
        self.target_JC_m = 1.34
        self.target_JC_T_melt = 925
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
        self.rigid_body_rho = 2000.

        # geometry
        self.target_length = 3. * self.rigid_body_length
        self.target_height = 1. * self.rigid_body_length

        self.dim = 2

        self.tf = 40 * 1e-6

        self.c0 = np.sqrt(self.E / (3 * (1. - 2 * self.nu) * self.rho0))
        self.pb = self.rho0 * self.c0**2.

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

        # boundary equations
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["target"], sources=["target"])

        self.boundary_equations = self.boundary_equations_1

        # print(self.boundary_equations)

    def add_user_options(self, group):
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
                           default=60,
                           help="Angle at which the velocity vector is pointed")

        group.add_argument("--spacing",
                           action="store",
                           type=float,
                           dest="spacing",
                           default=100,
                           help="Spacing in nanometers")

        add_bool_argument(group, 'remove-interior', dest='remove_interior',
                          default=False,
                          help='Remove the inside particles of rigid body')

    def consume_user_options(self):
        # self.nu = self.options.nu
        self.vel_alpha = self.options.vel_alpha
        self.azimuth_theta = self.options.azimuth_theta

        self.dx = self.options.spacing * 1e-6
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        self.rigid_body_spacing = self.dx

        # edac constants
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # compute the timestep
        self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        # self.dt = 1e-9
        print("timestep is ", self.dt)

        self.remove_interior = self.options.remove_interior

    def create_rigid_body(self):
        # create a row of six cylinders
        x, y = get_2d_block(self.dx, self.rigid_body_length,
                            self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def create_particles(self):
        x, y = get_2d_block(self.dx, self.target_length + 6. * self.dx,
                            self.target_height + 6. * self.dx)
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
        target.add_constant('total_no_bodies', [3])

        indices = []
        min_y = min(target.y)
        min_x = min(target.x)
        max_x = max(target.x)
        for i in range(len(target.y)):
            if target.y[i] < min_y + 3. * self.dx:
                indices.append(i)
            elif target.x[i] < min_x + 3. * self.dx:
                indices.append(i)
            elif target.x[i] > max_x - 3. * self.dx:
                indices.append(i)
        # add static particle data
        target.add_property('is_static')
        target.is_static[:] = 0.
        target.is_static[indices] = 1.

        # Create rigid body
        xc, yc, body_id = self.create_rigid_body()
        xc1, yc1, _zs1 = rotate(xc, yc, np.zeros(len(xc)), axis=np.array([0., 0., 1.]),
                                angle=self.azimuth_theta)
        body_id_1 = np.ones(len(xc), dtype=int) * 0
        # yc1 += max(target.y) - min(yc) + 1.1 * self.dx
        yc1 += max(target.y) - min(yc) + 10.1 * self.dx

        body_id_2 = np.ones(len(xc), dtype=int) * 1
        xc2 = xc1 - self.rigid_body_length * 0.5
        yc2 = yc1 + self.rigid_body_height * 1.1

        x = np.concatenate((xc1, xc2))
        y = np.concatenate((yc1, yc2))

        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = 1.0 * self.dx
        rad_s = self.dx / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=x,
                                        y=y,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        constants={
                                            'E': 69 * 1e9,
                                            'poisson_ratio': 0.3,
                                            'spacing0': self.dx,
                                        })
        body_id = np.concatenate((body_id_1, body_id_2))
        dem_id = body_id[:] + 1
        # reset to 1 so that the bodies don't collide among themselves
        # dem_id[:] = 1
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('total_no_bodies', [3])

        self.scheme.setup_properties([target, rigid_body])

        rigid_body.add_property('contact_force_is_boundary')
        rigid_body.contact_force_is_boundary[:] = rigid_body.is_boundary[:]

        vel = 51. * 5.

        angle = self.vel_alpha / 180 * np.pi
        self.scheme.scheme.set_linear_velocity(
            rigid_body, np.array([vel * cos(angle), -vel * sin(angle), 0.,
                                  vel * cos(angle), -vel * sin(angle), 0.]))
        # self.scheme.scheme.set_angular_velocity(
        #     rigid_body, np.array([0.0, 0.0, 10.]))
        # remove particles which are not boundary
        if self.remove_interior:
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

        return [target, rigid_body]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf

        output = np.array([0.0,
                           4.524 * 1e-6, 3.117 * 1e-5, 5.731 * 1e-5,  # for all bodies
                           9.0493 * 1e-6, 3.67 * 1e-5, 6. * 1e-5,  # for extrusion
                           1.106 * 1e-5, 3.4689 * 1e-5, 5.8318 * 1e-5  # for extrusion
        ])

        pfreq = 100

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq,
                                     output_at_times=output)

    def create_scheme(self):
        solid = SolidsScheme(solids=['target'],
                             rigid_bodies=['rigid_body'],
                             dim=2,
                             h=self.h,
                             hdx=self.hdx,
                             artificial_vis_alpha=self.artificial_vis_alpha,
                             artificial_vis_beta=self.artificial_vis_beta,
                             kr=1e12)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['target']
        b.scalar = 'is_damaged'
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        from pysph.solver.utils import iter_output

        # =======================================
        # =======================================
        # Schematic
        # =======================================
        info = self.read_info(fname)
        output_files = self.output_files
        data = load(output_files[0])
        particle_arrays = data['arrays']
        target = particle_arrays['target']
        rigid_body = particle_arrays['rigid_body']
        # solver_data = data['solver_data']

        s = 0.3
        # print(_t)
        fig, axs = plt.subplots(1, 1)
        axs.scatter(rigid_body.x, rigid_body.y, s=s, c=rigid_body.body_id)
        # axs.grid()
        axs.set_aspect('equal', 'box')
        # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

        # # im_ratio = tmp.shape[0]/tmp.shape[1]
        # x_min = min(body.x) - self.body_height
        # x_max = max(body.x) + 3. * self.body_height
        # y_min = min(body.y) - 4. * self.body_height
        # y_max = max(body.y) + 1. * self.body_height

        # filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
        #     (wall.y >= y_min) & (wall.y <= y_max))
        # wall_x = wall.x[filtr_1]
        # wall_y = wall.y[filtr_1]
        # wall_m = wall.m[filtr_1]
        # tmp = axs.scatter(wall_x, wall_y, s=s, c=wall_m)
        tmp = axs.scatter(target.x, target.y, s=s, c=target.m)
        axs.axis('off')
        axs.set_xticks([])
        axs.set_yticks([])

        # save the figure
        figname = os.path.join(os.path.dirname(fname), "schematic.pdf")
        fig.savefig(figname, dpi=300)
        # =============================================================
        # =============================================================

        # =============================================================
        # =============================================================
        # Save the plots of all bodies at three times
        info = self.read_info(fname)
        output_files = self.output_files
        output_times_all_bodies = np.array([4.524 * 1e-6, 3.117 * 1e-5, 5.731 * 1e-5])
        logfile = os.path.join(
            os.path.dirname(fname),
            'multi_body_erosion_example_3.log')
        # to_plot = get_files_at_given_times_from_log(output_files, output_times_all_bodies,
        #                                             logfile)
        to_plot = get_files_at_given_times(output_files, output_times_all_bodies)
        # print(to_plot)
        for i, f in enumerate(to_plot):
            data = load(f)
            _t = data['solver_data']['t']
            rigid_body = data['arrays']['rigid_body']
            target = data['arrays']['target']

            s = 0.8
            fig, axs = plt.subplots(1, 1, figsize=(10, 6))

            x_min = -4.067 * 1e-3
            x_max = 9.519 * 1e-3 + 4.78 * 1e-3 / 2.
            y_min = -7.5 * 1e-4
            y_max = 1.191 * 1e-2
            # Get the rigid body particle arrays with in the given limits
            filtr_1 = ((rigid_body.x >= x_min) & (rigid_body.x <= x_max)) & (
                (rigid_body.y >= y_min) & (rigid_body.y <= y_max))
            rigid_body_x = rigid_body.x[filtr_1] * 1e3
            rigid_body_y = rigid_body.y[filtr_1] * 1e3
            rigid_body_body_id = rigid_body.body_id[filtr_1]
            tmp = axs.scatter(rigid_body_x, rigid_body_y, s=s, c=rigid_body_body_id)

            # Get the target particle arrays with in the given limits
            filtr_1 = ((target.x >= x_min) & (target.x <= x_max)) & (
                (target.y >= y_min) & (target.y <= y_max))
            target_x = target.x[filtr_1] * 1e3
            target_y = target.y[filtr_1] * 1e3
            target_eps = target.eff_plastic_strain[filtr_1]

            # # Use this
            # axs.scatter(target_x, target_y, s=s, c=target_m, vmin=c_min,
            #             vmax=c_max, cmap="jet_r")
            # Delete this later

            # Set the limits for the extrapolated scattered values
            x_min = x_min * 1e3
            x_max = x_max * 1e3
            y_min = y_min * 1e3
            y_max = y_max * 1e3
            axs.scatter(target_x, target_y, s=s, c=target_eps, cmap="jet_r")
            axs.set_xlim([x_min, x_max])
            axs.set_ylim([y_min, y_max])
            axs.grid()
            axs.set_aspect('equal', 'box')
            plt.xlabel('x-dimension (10^{-3}(m))')
            plt.ylabel('y-dimension (10^{-3}(m))')
            # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

            # fig.colorbar(tmp, format='%.0e', orientation='horizontal',
            #              shrink=0.7)
            # axs.set_title(f"t = {_t:.1e} s")

            # save the figure
            figname = os.path.join(os.path.dirname(fname), "all_bodies_time" + str(i) + ".pdf")
            fig.savefig(figname, dpi=300)

        # =============================================================
        # =============================================================
        # Plot the center of mass of first body with time
        info = self.read_info(fname)
        output_files = self.output_files

        t, y_com = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            y_com.append(rb.xcm[1])

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")
        np.savez(res, t=t, y_com=y_com)

        plt.clf()
        plt.plot(t, y_com, "^-", label='Simulated')
        plt.title('y-center of mass')
        plt.xlabel('t')
        plt.ylabel('y-center of mass (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "ycom_vs_time.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Dong2016CaseA1SquareParticleOnAl6061T6()

    app.run()
    app.post_process(app.info_filename)
