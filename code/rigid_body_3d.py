import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (QuinticSpline)

from pysph.sph.wc.gtvf import GTVFIntegrator
from rigid_body_common import (set_total_mass, set_center_of_mass,
                               set_body_frame_position_vectors,
                               set_body_frame_normal_vectors,
                               set_moment_of_inertia_and_its_inverse,
                               BodyForce, SumUpExternalForces,
                               normalize_R_orientation, RigidBodyLVC,
                               RigidBodyDongRigidRigid,
                               RigidBodyDongRigidWall,
                               ComputeContactForceNormals,
                               ComputeContactForceDistanceAndClosestPoint,
                               ComputeContactForce)

# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)
from numpy import sin, cos


class GTVFRigidBody3DStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
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

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
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
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3 + j] = dst.xcm[i3 + j] + dt * dst.vcm[i3 + j]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3 + 0]],
                                  [-dst.omega[i3 + 1], dst.omega[i3 + 0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9 + 9] = dst.R[i9:i9 + 9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9 + 9])

            # update the moment of inertia
            R = dst.R[i9:i9 + 9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(
                R,
                dst.inertia_tensor_inverse_body_frame[i9:i9 + 9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9 + 9] = (np.matmul(
                tmp, R_t)).ravel()[:]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal,
               d_is_boundary):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
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

        # update normal vectors of the boundary
        if d_is_boundary[d_idx] == 1:
            d_normal[idx3 + 0] = (d_R[i9 + 0] * d_normal0[idx3] +
                                  d_R[i9 + 1] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 2] * d_normal0[idx3 + 2])
            d_normal[idx3 + 1] = (d_R[i9 + 3] * d_normal0[idx3] +
                                  d_R[i9 + 4] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 5] * d_normal0[idx3 + 2])
            d_normal[idx3 + 2] = (d_R[i9 + 6] * d_normal0[idx3] +
                                  d_R[i9 + 7] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 8] * d_normal0[idx3 + 2])

    def py_stage3(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
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

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class LeapFrogRigidBody3DStep(IntegratorStep):
    """
    This step has to be used when Leap frog integrator is used
    such as 'GTVFIntegrator'

    FIXME: This is not implemented yet.
    """
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.xcm0[3*i+j] = dst.xcm[3*i+j]
                dst.vcm0[3*i+j] = dst.vcm[3*i+j]

                # save the current angular momentum
                dst.ang_mom0[j] = dst.ang_mom[j]

            # save the current orientation
            for j in range(9):
                dst.R0[9*i+j] = dst.R[9*i+j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dtb2 * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dtb2

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dtb2 * dst.torque[i3:i3+3])

            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
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
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dt * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dt * dst.torque[i3:i3+3])
            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

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
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw


class RK2RigidBody3DStep(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.xcm0[3*i+j] = dst.xcm[3*i+j]
                dst.vcm0[3*i+j] = dst.vcm[3*i+j]

                # save the current angular momentum
                dst.ang_mom0[j] = dst.ang_mom[j]

            # save the current orientation
            for j in range(9):
                dst.R0[9*i+j] = dst.R[9*i+j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dtb2 * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dtb2

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dtb2 * dst.torque[i3:i3+3])

            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
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
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dt * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dt * dst.torque[i3:i3+3])
            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

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
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw


class RigidBody3DScheme(Scheme):
    def __init__(self, rigid_bodies, boundaries, dim,
                 kr=1e5, kf=1e5, fric_coeff=0.5, gx=0.0, gy=0.0, gz=0.0):
        self.rigid_bodies = rigid_bodies

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        if rigid_bodies is None:
            self.rigid_bodies = []
        else:
            self.rigid_bodies = rigid_bodies

        # rigid body parameters
        self.dim = dim

        self.kernel = QuinticSpline

        self.integrator = "gtvf"

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff

        self.solver = None

    def add_user_options(self, group):
        # choices = ['bui', 'canelas']
        # group.add_argument("--dem",
        #                    action="store",
        #                    dest='dem',
        #                    default="bui",
        #                    choices=choices,
        #                    help="DEM interaction " % choices)

        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e5,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

    def consume_user_options(self, options):
        _vars = ['kr', 'kf', 'fric_coeff']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equations()

    def _get_gtvf_equations(self):
        # elastic solid equations
        all = self.rigid_bodies

        stage1 = []
        # ==============================
        # Stage 2 equations
        # ==============================
        stage2 = []
        #######################
        # Handle rigid bodies #
        #######################
        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceNormals(dest=name,
                                               sources=self.boundaries))

            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceDistanceAndClosestPoint(
                        dest=name, sources=self.boundaries))
            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    BodyForce(dest=name,
                              sources=None,
                              gx=self.gx,
                              gy=self.gy,
                              gz=self.gz))
            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForce(dest=name,
                                        sources=None,
                                        kr=self.kr,
                                        kf=self.kf,
                                        fric_coeff=self.fric_coeff))

            # for name in self.rigid_bodies:
            #     g5.append(RigidBodyDongRigidRigid(dest=name,
            #                                       sources=self.rigid_bodies))

            #     if len(self.boundaries) > 0:
            #         g5.append(RigidBodyDongRigidWall(dest=name,
            #                                          sources=self.boundaries))

            stage2.append(Group(equations=g5, real=False))

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name, sources=None))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        bodystep = GTVFRigidBody3DStep()
        integrator_cls = GTVFIntegrator

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

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

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties(pa, 'contact_force_normal_x',
                           'contact_force_normal_y',
                           'contact_force_normal_z',
                           'contact_force_normal_wij',

                           'contact_force_normal_tmp_x',
                           'contact_force_normal_tmp_y',
                           'contact_force_normal_tmp_z',

                           'contact_force_dist_tmp',
                           'contact_force_dist',

                           'overlap',
                           'ft_x',
                           'ft_y',
                           'ft_z',
                           'fn_x',
                           'fn_y',
                           'fn_z',
                           'delta_lt_x',
                           'delta_lt_y',
                           'delta_lt_z',
                           'vx_source',
                           'vy_source',
                           'vz_source',
                           'x_source',
                           'y_source',
                           'z_source',
                           'ti_x',
                           'ti_y',
                           'ti_z',
                           'closest_point_dist_to_source')

            add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0')

            add_properties(pa, 'rho_fsi', 'm_fsi', 'p_fsi')

            nb = int(np.max(pa.body_id) + 1)

            # dem_id = props.pop('dem_id', None)

            consts = {
                'total_mass':
                np.zeros(nb, dtype=float),
                'xcm':
                np.zeros(3 * nb, dtype=float),
                'xcm0':
                np.zeros(3 * nb, dtype=float),
                'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                # moment of inertia izz (this is only for 2d)
                'izz':
                np.zeros(nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_global_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_global_frame':
                np.zeros(9 * nb, dtype=float),
                # total force at the center of mass
                'force':
                np.zeros(3 * nb, dtype=float),
                # torque about the center of mass
                'torque':
                np.zeros(3 * nb, dtype=float),
                # velocity, acceleration of CM.
                'vcm':
                np.zeros(3 * nb, dtype=float),
                'vcm0':
                np.zeros(3 * nb, dtype=float),
                # angular momentum
                'ang_mom':
                np.zeros(3 * nb, dtype=float),
                'ang_mom0':
                np.zeros(3 * nb, dtype=float),
                # angular velocity in global frame
                'omega':
                np.zeros(3 * nb, dtype=float),
                'omega0':
                np.zeros(3 * nb, dtype=float),
                'nb':
                nb
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            # compute the properties of the body
            set_total_mass(pa)
            set_center_of_mass(pa)

            # this function will compute
            # inertia_tensor_body_frame
            # inertia_tensor_inverse_body_frame
            # inertia_tensor_global_frame
            # inertia_tensor_inverse_global_frame
            # of the rigid body
            set_moment_of_inertia_and_its_inverse(pa)

            set_body_frame_position_vectors(pa)

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])
            # print(equations)

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

            # make normals of particle other than boundary particle as zero
            # for i in range(len(pa.x)):
            #     if pa.is_boundary[i] == 0:
            #         pa.normal[3 * i] = 0.
            #         pa.normal[3 * i + 1] = 0.
            #         pa.normal[3 * i + 2] = 0.

            # normal vectors in terms of body frame
            set_body_frame_normal_vectors(pa)

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'normal',
                'is_boundary', 'fz', 'm', 'body_id', 'h'
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

        # set the angular momentum
        for i in range(max(pa.body_id) + 1):
            i9 = 9 * i
            i3 = 3 * i
            pa.ang_mom[i3:i3 + 3] = np.matmul(
                pa.inertia_tensor_global_frame[i9:i9 + 9].reshape(3, 3),
                pa.omega[i3:i3 + 3])[:]

        self._set_particle_velocities(pa)

    def get_solver(self):
        return self.solver
