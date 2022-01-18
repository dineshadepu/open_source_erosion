"""Two elastic rings colliding. Benchmark in Solid mechanics.

# Test cases

1. GTVF

python rings.py --openmp --pst gtvf --uhat --no-continuity-tvf-correction --no-shear-stress-tvf-correction --no-edac --no-surf-p-zero --artificial-vis-alpha 2. --kernel-choice 1 -d rings_gtvf_kernel_1_output

a. Doesn't work with alpha 1.

2. GRAY

python rings.py --openmp --pst gray --no-uhat --no-continuity-tvf-correction --no-shear-stress-tvf-correction --no-edac --no-surf-p-zero --artificial-vis-alpha 1. --kernel-choice 1 -d rings_gray_kernel_1_output

a. Works with 0.5 alpha (To test this change the alpha value in the above command)

b. Works with 0.1 alpha too, but a bit disturbed

c. Fails with 0.0 alpha

3. IPST

python rings.py --openmp --pst ipst --no-uhat-velgrad --uhat-cont --continuity-tvf-correction --no-shear-stress-tvf-correction --edac --no-surf-p-zero --kernel-choice 1 --artificial-vis-alpha 1 --ipst-max-iterations 10 --ipst-interval 1 -d rings_continuity_correction_shear_stess_correction_edac_true_ipst_max_iteration_10_ipst_interval_1_artificial_visc_0_1_kernel_1_output --pfreq 300 --detailed

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

from solid_mech import SetHIJForInsideParticles, SolidsScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from pysph.sph.scheme import add_bool_argument
from pysph.solver.utils import load, get_files

import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


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



class Rings(Application):
    def initialize(self):
        # constants
        self.E = 1e7
        self.nu = 0.3975
        self.rho0 = 1.2 * 1e3

        self.dx = 0.001
        self.hdx = 1.3
        self.h = self.hdx * self.dx
        self.fac = self.dx / 2.
        self.kernel_fac = 3

        # geometry
        self.ri = 0.03
        self.ro = 0.04

        self.spacing = 0.041
        self.dim = 2

        # compute the timestep
        self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        print(self.dt)

        self.tf = 0.016

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
            destinations=["ring_1"], sources=["ring_1"])
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["ring_2"], sources=["ring_2"])

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        # print(self.boundary_equations)

    def add_user_options(self, group):
        group.add_argument("--poisson-ratio",
                           action="store",
                           type=float,
                           dest="nu",
                           default=0.3975,
                           help="Poisson ratio of the ring (Defaults to 0.3975)")

    def consume_user_options(self):
        self.nu = self.options.nu

    def create_rings_geometry(self):
        import matplotlib.pyplot as plt
        x, y = np.array([]), np.array([])

        radii = np.arange(self.ri, self.ro, self.dx)
        # radii = np.arange(self.ri, self.ri+self.dx, self.dx)

        for radius in radii:
            points = np.arange(0., 2 * np.pi * radius, self.dx)
            theta = np.arange(0., 2. * np.pi, 2. * np.pi / len(points))
            xr, yr = radius * np.cos(theta), radius * np.sin(theta)
            x = np.concatenate((x, xr))
            y = np.concatenate((y, yr))

        # plt.scatter(x, y)
        # plt.show()
        return x, y

    def create_particles(self):
        spacing = self.spacing  # spacing = 2*5cm

        x, y = self.create_rings_geometry()
        x = x.ravel()
        y = y.ravel()

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx * dx
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0

        print("poisson ratio is ", self.nu)

        ring_1 = get_particle_array(x=-x - spacing,
                                    y=y,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    name="ring_1",
                                    constants={
                                        'E': self.E,
                                        'n': 4,
                                        'nu': self.nu,
                                        'spacing0': self.dx,
                                        'rho_ref': self.rho0
                                    })

        dem_id = np.zeros(len(ring_1.x), dtype=int)
        ring_1.add_property('dem_id', type='int', data=dem_id)
        ring_1.add_constant('total_no_bodies', [2])

        ring_2 = get_particle_array(x=x + spacing,
                                    y=y,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    name="ring_2",
                                    constants={
                                        'E': self.E,
                                        'n': 4,
                                        'nu': self.nu,
                                        'spacing0': self.dx,
                                        'rho_ref': self.rho0
                                    })

        body_id = np.zeros(len(ring_2.x), dtype=int)
        dem_id = np.ones(len(ring_2.x), dtype=int)
        ring_2.add_property('body_id', type='int', data=body_id)
        ring_2.add_property('dem_id', type='int', data=dem_id)
        ring_2.add_constant('total_no_bodies', [2])

        self.scheme.setup_properties([ring_1, ring_2])

        u_f = self.u_f
        ring_1.u = ring_1.cs * u_f

        self.scheme.scheme.set_linear_velocity(
            ring_2, np.array([-ring_1.u[0], 0., 0.]))

        ring_2.add_property('contact_force_is_boundary')
        ring_2.contact_force_is_boundary[:] = ring_2.is_boundary[:]

        return [ring_1, ring_2]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf

        tf = 15.5 * 1e-3
        output = np.array([0.0, 1.38 * 1e-3, 5.17 * 1e-3, 7.38 * 1e-3, 11.462 *
                           1e-3, 15.4 * 1e-3])

        pfreq = 300

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq,
                                     output_at_times=output)

    def create_scheme(self):
        solid = SolidsScheme(solids=['ring_1'],
                             rigid_bodies=['ring_2'],
                             boundaries=None,
                             dim=2,
                             pb=self.pb,
                             edac_nu=self.edac_nu,
                             u_max=self.u_max,
                             mach_no=self.mach_no,
                             ipst_max_iterations=self.ipst_max_iterations,
                             h=self.h,
                             hdx=self.hdx,
                             artificial_vis_alpha=self.artificial_vis_alpha,
                             artificial_vis_beta=self.artificial_vis_beta,
                             gy=-0.)

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
        if self.options.pst in ['sun2019', 'ipst']:
            if solver.count % 10 == 0:
                t = solver.t
                dt = solver.dt

                arrays = self.particles
                a_eval = self._make_accel_eval(self.boundary_equations, arrays)

                # When
                a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = Rings()

    app.run()
