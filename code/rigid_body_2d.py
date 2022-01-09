import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.wc.gtvf import GTVFIntegrator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from rigid_body_common import (set_total_mass, set_center_of_mass,
                               set_moment_of_inertia_izz,
                               set_body_frame_position_vectors, BodyForce,
                               SumUpExternalForces)

from numpy import sin, cos


class LeapFrogRigidBody2DStep(IntegratorStep):
    """
    This step has to be used when Leap frog integrator is used
    such as 'GTVFIntegrator'
    """
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                dst.xcm[i3 + j] = dst.xcm0[i3 + j] + dtb2 * dst.vcm[i3 + j]
                dst.vcm[i3 + j] = dst.vcm0[
                    i3 + j] + dtb2 * dst.force[i3 + j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega = dst.omega[i3 + 2]

            # update the orientation angle
            dst.orientation_angle[i] = dst.orientation_angle0[i] + omega * dtb2

            # update the orientation matrix in 2d
            dst.R[i9 + 0] = cos(dst.orientation_angle[i])
            dst.R[i9 + 1] = -sin(dst.orientation_angle[i])
            dst.R[i9 + 3] = sin(dst.orientation_angle[i])
            dst.R[i9 + 4] = cos(dst.orientation_angle[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega[i3 + 2] + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        # print(i9)
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                dst.xcm[i3 + j] = dst.xcm0[i3 + j] + dt * dst.vcm[i3 + j]
                dst.vcm[i3 + j] = dst.vcm0[
                    i3 + j] + dt * dst.force[i3 + j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega = dst.omega[i3 + 2]

            # update the orientation angle
            dst.orientation_angle[i] = dst.orientation_angle0[i] + omega * dt

            # update the orientation matrix in 2d
            dst.R[i9 + 0] = cos(dst.orientation_angle[i])
            dst.R[i9 + 1] = -sin(dst.orientation_angle[i])
            dst.R[i9 + 3] = sin(dst.orientation_angle[i])
            dst.R[i9 + 4] = cos(dst.orientation_angle[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega0[i3 + 2] + omega_dot * dt

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class RK2RigidBody2DStep(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.xcm0[3 * i + j] = dst.xcm[3 * i + j]
                dst.vcm0[3 * i + j] = dst.vcm[3 * i + j]

                # save the current angular momentum
                # dst.L0[j] = dst.L[j]
                dst.omega0[3 * i + j] = dst.omega[3 * i + j]

            # save the current orientation
            dst.orientation_angle0[i] = dst.orientation_angle[i]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                dst.xcm[i3 + j] = dst.xcm0[i3 + j] + dtb2 * dst.vcm[i3 + j]
                dst.vcm[i3 + j] = dst.vcm0[
                    i3 + j] + dtb2 * dst.force[i3 + j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega = dst.omega[i3 + 2]

            # update the orientation angle
            dst.orientation_angle[i] = dst.orientation_angle0[i] + omega * dtb2

            # update the orientation matrix in 2d
            dst.R[i9 + 0] = cos(dst.orientation_angle[i])
            dst.R[i9 + 1] = -sin(dst.orientation_angle[i])
            dst.R[i9 + 3] = sin(dst.orientation_angle[i])
            dst.R[i9 + 4] = cos(dst.orientation_angle[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega[i3 + 2] + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        # print(i9)
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                dst.xcm[i3 + j] = dst.xcm0[i3 + j] + dt * dst.vcm[i3 + j]
                dst.vcm[i3 + j] = dst.vcm0[
                    i3 + j] + dt * dst.force[i3 + j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega = dst.omega[i3 + 2]

            # update the orientation angle
            dst.orientation_angle[i] = dst.orientation_angle0[i] + omega * dt

            # update the orientation matrix in 2d
            dst.R[i9 + 0] = cos(dst.orientation_angle[i])
            dst.R[i9 + 1] = -sin(dst.orientation_angle[i])
            dst.R[i9 + 3] = sin(dst.orientation_angle[i])
            dst.R[i9 + 4] = cos(dst.orientation_angle[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega0[i3 + 2] + omega_dot * dt

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class GTVFRigidBody2DStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        # As per GTVF integrator in stage 1 we will update the velocities to
        # 1./2. time step
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            for j in range(3):
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + dtb2 * dst.force[i3 + j] / dst.total_mass[i]

            # move angular velocity to t + dt/2.
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega[i3 + 2] + omega_dot * dtb2

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly
        bid, i3, i9 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])
        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

    def py_stage2(self, dst, t, dt):
        # move positions to t + dt time step
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                dst.xcm[i3 + j] = dst.xcm[i3 + j] + dt * dst.vcm[i3 + j]

            # angular velocity in terms of matrix
            omega = dst.omega[i3 + 2]

            # update the orientation angle
            dst.orientation_angle[i] = dst.orientation_angle[i] + omega * dt

            # update the orientation matrix in 2d
            dst.R[i9 + 0] = cos(dst.orientation_angle[i])
            dst.R[i9 + 1] = -sin(dst.orientation_angle[i])
            dst.R[i9 + 3] = sin(dst.orientation_angle[i])
            dst.R[i9 + 4] = cos(dst.orientation_angle[i])

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

    def py_stage3(self, dst, t, dt):
        # As per GTVF integrator in stage 3 we will update the velocities from
        # t + dt / 2. to t + dt time step
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            for j in range(3):
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + dtb2 * dst.force[i3 + j] / dst.total_mass[i]

            # move angular velocity to t + dt
            # omega_dot is
            omega_dot = dst.torque[i3 + 2] / dst.izz[i]
            dst.omega[i3 + 2] = dst.omega[i3 + 2] + omega_dot * dtb2

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id):
        # Update the velocities to t + dt time step
        # some variables to update the positions seamlessly
        bid, i3, i9 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])
        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - xcm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class RigidBody2DScheme(Scheme):
    def __init__(self,
                 rigid_bodies,
                 boundaries,
                 integrator="gtvf",
                 dim=2,
                 gx=0.0,
                 gy=0.0,
                 gz=0.0,
                 kernel_choice="1",
                 kernel_factor=3):
        self.rigid_bodies = rigid_bodies

        if boundaries == None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # assert(dim, 2)

        self.dim = dim

        self.kernel = CubicSpline

        self.integrator = integrator

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

    def add_user_options(self, group):
        choices = ['rk2', 'gtvf']
        group.add_argument("--integrator",
                           action="store",
                           dest='integrator',
                           default="gtvf",
                           choices=choices,
                           help="Specify what integrator to use " % choices)

    def consume_user_options(self, options):
        _vars = ['integrator']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        if self.integrator == 'rk2':
            return self._get_rk2_equations()
        elif self.integrator == 'gtvf':
            return self._get_gtvf_equations()

    def _get_rk2_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        equations = []
        g1 = []
        all = list(set(self.rigid_bodies + self.boundaries))

        for name in self.rigid_bodies:
            g1.append(
                BodyForce(dest=name,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        equations.append(Group(equations=g1))

        # computation of total force and torque at cener of mass
        g3 = []
        for name in self.rigid_bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        return equations

    def _get_gtvf_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        g1 = []
        all = list(set(self.rigid_bodies + self.boundaries))

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # There will be no equations for stage 1

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []

        for name in self.rigid_bodies:
            g1.append(
                BodyForce(dest=name,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        stage2.append(Group(equations=g1))

        # computation of total force and torque at cener of mass
        g3 = []
        for name in self.rigid_bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        stage2.append(Group(equations=g3, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        if self.integrator == 'rk2':
            integrator_cls = EPECIntegrator
            rbstep = RK2RigidBody2DStep()
        elif self.integrator == 'gtvf':
            integrator_cls = GTVFIntegrator
            rbstep = GTVFRigidBody2DStep()

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = rbstep

        cls = integrator_cls

        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

            add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0')

            nb = int(np.max(pa.body_id) + 1)

            # dem_id = props.pop('dem_id', None)

            consts = {
                'total_mass': np.zeros(nb, dtype=float),
                'xcm': np.zeros(3 * nb, dtype=float),
                'xcm0': np.zeros(3 * nb, dtype=float),
                'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                # moment of inertia izz (this is only for 2d)
                'izz': np.zeros(nb, dtype=float),
                # moment of inertia inverse in body frame
                'mib': np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in global frame
                'mig': np.zeros(9 * nb, dtype=float),
                # total force at the center of mass
                'force': np.zeros(3 * nb, dtype=float),
                # torque about the center of mass
                'torque': np.zeros(3 * nb, dtype=float),
                # velocity, acceleration of CM.
                'vcm': np.zeros(3 * nb, dtype=float),
                'vcm0': np.zeros(3 * nb, dtype=float),
                # angular momentum
                'L': np.zeros(3 * nb, dtype=float),
                'L0': np.zeros(3 * nb, dtype=float),
                # angular velocity in global frame
                'omega': np.zeros(3 * nb, dtype=float),
                'omega0': np.zeros(3 * nb, dtype=float),
                'orientation_angle': np.zeros(nb, dtype=float),
                'orientation_angle0': np.zeros(nb, dtype=float),
                'nb': nb
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            # compute the properties of the body
            set_total_mass(pa)
            set_center_of_mass(pa)
            set_moment_of_inertia_izz(pa)
            set_body_frame_position_vectors(pa)

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'dx0', 'dy0',
                'dz0', 'm', 'body_id'
            ])

    def _set_particle_velocities(self, pa):
        for i in range(max(pa.body_id) + 1):
            fltr = np.where(pa.body_id == i)
            bid = i
            i9 = 9 * bid
            i3 = 3 * bid

            for j in fltr:
                dx = (pa.R[i9 + 0] * pa.dx0[j] + pa.R[i9 + 1] * pa.dy0[j] +
                      pa.R[i9 + 2] * pa.dz0[j])
                dy = (pa.R[i9 + 3] * pa.dx0[j] + pa.R[i9 + 4] * pa.dy0[j] +
                      pa.R[i9 + 5] * pa.dz0[j])
                dz = (pa.R[i9 + 6] * pa.dx0[j] + pa.R[i9 + 7] * pa.dy0[j] +
                      pa.R[i9 + 8] * pa.dz0[j])

                du = pa.omega[i3 + 1] * dz - pa.omega[i3 + 2] * dy
                dv = pa.omega[i3 + 2] * dx - pa.omega[i3 + 0] * dz
                dw = pa.omega[i3 + 0] * dy - pa.omega[i3 + 1] * dx

                pa.u[j] = pa.vcm[i3 + 0] + du
                pa.v[j] = pa.vcm[i3 + 1] + dv
                pa.w[j] = pa.vcm[i3 + 2] + dw

    def set_linear_velocity(self, pa, linear_vel):
        pa.vcm[:] = linear_vel

        self._set_particle_velocities(pa)

    def set_angular_velocity(self, pa, angular_vel):
        pa.omega[:] = angular_vel[:]

        self._set_particle_velocities(pa)

    def get_solver(self):
        return self.solver
