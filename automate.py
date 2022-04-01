import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name, mdict, dprod, opts2path, filter_cases
from automan.jobs import free_cores
from pysph.solver.utils import load, get_files

import numpy as np
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


# n_core = free_cores()
# n_thread = 2 * free_cores()

n_core = 24
n_thread = 24 * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


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


class TestRigidBodyCollision1(Problem):
    def get_name(self):
        return 'test_rigid_body_collision_1'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/test_rigid_body_collision_1.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                pfreq=50,
                ), 'Case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Mohseni2021ControlledSlidingOnAFlatSurface(Problem):
    def get_name(self):
        return 'mohseni_2021_controlled_sliding_on_a_flat_surface'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_controlled_sliding_on_a_flat_surface.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rb3d',
                detail=None,
                pfreq=100,
                kr=1e5,
                fric_coeff=0.5,
                tf=1.5,
                ), 'case_1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        txsun = data[rand_case]['txsun']
        xdsun = data[rand_case]['xdsun']
        txbogaers = data[rand_case]['txbogaers']
        xdbogaers = data[rand_case]['xdbogaers']
        txidelsohn = data[rand_case]['txidelsohn']
        xdidelsohn = data[rand_case]['xdidelsohn']
        txliu = data[rand_case]['txliu']
        xdliu = data[rand_case]['xdliu']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        plt.plot(txsun, xdsun, "o", label='Sun et al 2019, MPS-DEM')
        plt.plot(txbogaers, xdbogaers, "^", label='Bogaers 2016, QN-LS')
        plt.plot(txidelsohn, xdidelsohn, "+", label='Idelsohn 20108, PFEM')
        plt.plot(txliu, xdliu, "v", label='Liu 2013, SPH')

        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            x_ctvf = data[name]['x_ctvf']

            plt.plot(t_ctvf, x_ctvf, '-', label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('x - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('x_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Mohseni2021FreeSlidingOnASlope(Problem):
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rfc',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rfc',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=0.4,
                tf=3.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rfc',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=3.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0.6': (dict(
                scheme='rfc',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=0.6,
                tf=3.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Mohseni2021FreeSlidingOnASlopeChallengingGeometry(Problem):
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope_challenging_geometry'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope_challenging_geometry.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rb3d',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rb3d',
                detail=None,
                pfreq=100,
                kr=1e8,
                fric_coeff=0.4,
                tf=1.,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        txsun = data[rand_case]['txsun']
        xdsun = data[rand_case]['xdsun']
        txbogaers = data[rand_case]['txbogaers']
        xdbogaers = data[rand_case]['xdbogaers']
        txidelsohn = data[rand_case]['txidelsohn']
        xdidelsohn = data[rand_case]['xdidelsohn']
        txliu = data[rand_case]['txliu']
        xdliu = data[rand_case]['xdliu']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        plt.plot(txsun, xdsun, "o", label='Sun et al 2019, MPS-DEM')
        plt.plot(txbogaers, xdbogaers, "^", label='Bogaers 2016, QN-LS')
        plt.plot(txidelsohn, xdidelsohn, "+", label='Idelsohn 20108, PFEM')
        plt.plot(txliu, xdliu, "v", label='Liu 2013, SPH')

        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            x_ctvf = data[name]['x_ctvf']

            plt.plot(t_ctvf, x_ctvf, '-', label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('x - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('x_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Shuangshuang2021TaylorImpactTest3D(Problem):
    def get_name(self):
        return 'shuangshuang_2021_taylor_impact_test'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/shuangshuang_2021_taylor_impact_test_3d.py' + backend

        self.opts = mdict(velocity=[181], length=[37.97 * 1e-3])

        self.cases = []
        for kw in self.opts:
            name = opts2path(kw)
            folder_name = self.input_path(name)
            self.cases.append(
                Simulation(folder_name, cmd,
                           job_info=dict(n_core=n_core,
                                         n_thread=n_thread), cache_nnps=None,
                           pst='sun2019',
                           no_solid_stress_bc=None,
                           no_solid_velocity_bc=None,
                           no_damping=None,
                           no_wall_pst=None,
                           artificial_vis_alpha=2.0,
                           pfreq=1000,
                           tf=40 * 1e-6,
                           **kw))

    def run(self):
        self.make_output_dir()
        self.plot_figures()

    def plot_figures(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])
        t_length_lsdyna = data[rand_case]['t_length_lsdyna']
        length_lsdyna = data[rand_case]['length_lsdyna']

        # ==================================
        # Plot length versus time
        # ==================================
        plt.plot(t_length_lsdyna, length_lsdyna, "-", label='LS-DYNA')
        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            length_ctvf = data[name]['length_ctvf']

            plt.plot(t_ctvf, length_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('length')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('length_vs_time.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot length versus time
        # ==================================


class Rushdie2019TaylorImpactTest3DPart1(Problem):
    def get_name(self):
        return 'rushdie_2019_taylor_impact_test_part_1'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/rushdie_2019_taylor_impact_test.py' + backend

        self.opts = mdict(velocity=[285.], length=[11.39 * 1e-3])
        self.opts += mdict(velocity=[234., 275., 302.], length=[15.19 * 1e-3])
        self.opts += mdict(velocity=[170., 215.], length=[25.29 * 1e-3])
        self.opts += mdict(velocity=[181., 183., 224., 234., 270.],
                           length=[37.97 * 1e-3])
        self.opts += mdict(velocity=[242.], length=[56.96 * 1e-3])

        self.cases = []
        for kw in self.opts:
            name = opts2path(kw)
            folder_name = self.input_path(name)
            self.cases.append(
                Simulation(folder_name, cmd,
                           job_info=dict(n_core=n_core,
                                         n_thread=n_thread), cache_nnps=None,
                           pst='sun2019',
                           solid_stress_bc=None,
                           solid_velocity_bc=None,
                           no_damping=None,
                           artificial_vis_alpha=2.0,
                           no_clamp=None,
                           wall_pst=None,
                           pfreq=1000,
                           tf=30 * 1e-6,
                           **kw))

    def run(self):
        self.make_output_dir()
        self.plot_disp(fname='homogenous')

    def plot_disp(self, fname):
        for case in self.cases:
            data = np.load(case.input_path('results.npz'))

            t = data['t_ctvf']
            amplitude_ctvf = data['amplitude_ctvf']

            label = opts2path(case.params, keys=['N'])

            plt.plot(t, amplitude_ctvf, label=label.replace('_', ' = '))

            t_fem = data['t_fem']
            amplitude_fem = data['amplitude_fem']

        # sort the fem data before plotting
        p = t_fem.argsort()
        t_fem_new = t_fem[p]
        amplitude_fem_new = amplitude_fem[p]
        plt.plot(t_fem_new, amplitude_fem_new, label='FEM')

        plt.xlabel('time')
        plt.ylabel('Y - amplitude')
        plt.legend()
        plt.savefig(self.output_path(fname))
        plt.clf()
        plt.close()


class TestSolidErosion1(Problem):
    def get_name(self):
        return 'test_solid_erosion_1'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/test_solid_erosion_1.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(

                pfreq=50,
                ), 'Case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dong2016CaseASquareParticleOnAl6061T6(Problem):
    def get_name(self):
        return 'dong_2016_4_1_case_A_1_square_particle_erosion_on_al6061_t6'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dong_2016_4_1_case_A_1_square_particle_erosion_on_al6061_t6.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                pfreq=300,
                ), 'Case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Vyas2022DeformationValidation(Problem):
    def get_name(self):
        return 'vyas_2022_3_3_deformation_validation'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/vyas_2022_3_3_deformation_validation.py' + backend

        # Base case info
        self.case_info = {
            # 'dx_0_5_velocity_6_264_angle_0': (dict(
            #     spacing=0.5 * 1e-3,
            #     pfreq=100,
            #     velocity=6.264,
            #     angle=90.,
            #     kr=1e12,
            #     ), 'dx=0.5'),

            # 'dx_0_25_velocity_6_264_angle_0': (dict(
            #     spacing=0.25 * 1e-3,
            #     velocity=6.264,
            #     angle=90.,
            #     pfreq=200,
            #     kr=1e12,
            #     ), 'dx=0.25'),

            # 'dx_0_25_velocity_60_264_angle_0': (dict(
            #     spacing=0.25 * 1e-3,
            #     velocity=60.264,
            #     angle=90.,
            #     pfreq=200,
            #     kr=1e12,
            #     ), 'dx=0.25, velocity 60'),

            'dx_15_nm_velocity_6_264_angle_0': (dict(
                no_mie_gruneisen_eos=None,
                spacing=0.15 * 1e-3 / 4.,
                velocity=6.264,
                angle=90.,
                pfreq=200,
                kr=1e12,
                ), 'dx=15 nm, velocity 6'),

            'dx_15_nm_velocity_60_264_angle_0': (dict(
                no_mie_gruneisen_eos=None,
                spacing=0.15 * 1e-3 / 4.,
                velocity=60.264,
                angle=90.,
                pfreq=200,
                kr=1e12,
                ), 'dx=15 nm, velocity 60'),

            'dx_15_nm_velocity_6_264_angle_0_mie_gruneisen': (dict(
                mie_gruneisen_eos=None,
                spacing=0.15 * 1e-3 / 4.,
                velocity=6.264,
                angle=90.,
                pfreq=200,
                kr=1e12,
                ), 'dx=15 nm, velocity 6'),

            'dx_15_nm_velocity_60_264_angle_0_mie_gruneisen': (dict(
                mie_gruneisen_eos=None,
                spacing=0.15 * 1e-3 / 4.,
                velocity=60.264,
                angle=90.,
                pfreq=200,
                kr=1e12,
                ), 'dx=15 nm, velocity 60'),

            # 'dx_0_125_velocity_6_264_angle_0': (dict(
            #     spacing=0.125 * 1e-3,
            #     velocity=6.264,
            #     angle=90.,
            #     pfreq=300,
            #     kr=1e12,
            #     ), 'dx=0.125'),

            # 'dx_0_125_velocity_60_264_angle_0': (dict(
            #     spacing=0.125 * 1e-3,
            #     velocity=60.264,
            #     angle=90.,
            #     pfreq=300,
            #     kr=1e12,
            #     ), 'dx=0.125, velocity 60'),

            # 'dx_0_0625_velocity_6_264_angle_0': (dict(
            #     spacing=0.0625 * 1e-3,
            #     velocity=6.264,
            #     angle=90.,
            #     pfreq=300,
            #     ), 'dx=0.0625'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


if __name__ == '__main__':
    PROBLEMS = [TestRigidBodyCollision1,
                Mohseni2021ControlledSlidingOnAFlatSurface,
                Mohseni2021FreeSlidingOnASlope,
                Mohseni2021FreeSlidingOnASlopeChallengingGeometry,

                Shuangshuang2021TaylorImpactTest3D,
                Rushdie2019TaylorImpactTest3DPart1,

                TestSolidErosion1,
                Vyas2022DeformationValidation]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
