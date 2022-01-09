import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name, mdict, dprod, opts2path, filter_cases
from automan.jobs import free_cores

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

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
                pfreq=300,
                kr=1e5,
                fric_coeff=0.4,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_tan_30': (dict(
                scheme='rb3d',
                detail=None,
                pfreq=300,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=3.,
                ), r'$\mu=$tan(30)'),
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


if __name__ == '__main__':
    PROBLEMS = [TestRigidBodyCollision1,
                Mohseni2021ControlledSlidingOnAFlatSurface,
                Mohseni2021FreeSlidingOnASlope]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
