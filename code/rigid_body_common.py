"""
The force interaction between, two rigid bodies, and rigid body wall and
rigid body and an elastic body (with and with out erosion is different and
needs different equations)
"""
import numpy as np
from pysph.sph.equation import Equation

from numpy import sqrt
from math import sin, pi, log

# constants
M_PI = pi


def add_properties_stride(pa, stride=1, *props):
    for prop in props:
        pa.add_property(name=prop, stride=stride)


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    # loop over all the bodies
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.xcm[3 * i] = np.sum(pa.m[fltr] * pa.x[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 1] = np.sum(pa.m[fltr] * pa.y[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 2] = np.sum(pa.m[fltr] * pa.z[fltr]) / pa.total_mass[i]


def set_moment_of_inertia_izz(pa):
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        izz = np.sum(pa.m[fltr] * ((pa.x[fltr] - pa.xcm[3 * i])**2. +
                                   (pa.y[fltr] - pa.xcm[3 * i + 1])**2.))
        pa.izz[i] = izz


def set_moment_of_inertia_and_its_inverse(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        pa.inertia_tensor_body_frame[9 * i:9 * i + 9] = I[:]

        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.inertia_tensor_inverse_body_frame[9 * i:9 * i + 9] = I_inv[:]

        # set the moment of inertia inverse in global frame
        # NOTE: This will be only computed once to compute the angular
        # momentum in the beginning.
        pa.inertia_tensor_global_frame[9 * i:9 * i + 9] = I[:]
        # set the moment of inertia inverse in global frame
        pa.inertia_tensor_inverse_global_frame[9 * i:9 * i + 9] = I_inv[:]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]
        for j in fltr:
            pa.dx0[j] = pa.x[j] - cm_i[0]
            pa.dy0[j] = pa.y[j] - cm_i[1]
            pa.dz0[j] = pa.z[j] - cm_i[2]


def set_body_frame_normal_vectors(pa):
    """Save the normal vectors w.r.t body frame"""
    pa.normal0[:] = pa.normal[:]


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy
        d_fz[d_idx] = d_m[d_idx] * self.gz


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        xcm = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        xcm = dst.xcm
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            frc[i3] += fx[j]
            frc[i3 + 1] += fy[j]
            frc[i3 + 2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - xcm[i3]
            dy = y[j] - xcm[i3 + 1]
            dz = z[j] - xcm[i3 + 2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
            trq[i3 + 2] += (dx * fy[j] - dy * fx[j])


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class ComputeContactForceNormalsMohseniRB(Equation):
    """Shoya Mohseni Mofidi, Particle Based Numerical Simulation Study of Solid
    Particle Erosion of Ductile Materials Leading to an Erosion Model, Including
    the Particle Shape Effect

    Equation 22 of Materials 2022

    Compute the normals on the rigid body particles (secondary surface) which is
    interacting with wall (primary surface). Here we expect the wall has a
    property identifying its boundary particles. Currently this equation only
    assumes the simulation has only one rigid body with a single wall.

    """
    def initialize(self, d_idx,
                   d_m,
                   d_contact_force_normal_tmp_x,
                   d_contact_force_normal_tmp_y,
                   d_contact_force_normal_tmp_z,
                   d_contact_force_normal_wij,
                   d_contact_force_is_boundary,
                   d_is_boundary,
                   d_total_no_bodies,
                   dt, t):
        i, t1, t2 = declare('int', 3)

        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i
            d_contact_force_normal_tmp_x[t2] = 0.
            d_contact_force_normal_tmp_y[t2] = 0.
            d_contact_force_normal_tmp_z[t2] = 0.
            d_contact_force_normal_wij[t2] = 0.

        d_contact_force_is_boundary[d_idx] = d_is_boundary[d_idx]

    def loop(self, d_idx,
             d_rho, d_m, RIJ, XIJ,
             s_idx,
             d_contact_force_normal_tmp_x,
             d_contact_force_normal_tmp_y,
             d_contact_force_normal_tmp_z,
             d_contact_force_normal_wij,
             s_contact_force_is_boundary,
             s_dem_id,
             d_dem_id,
             d_normal,
             d_total_no_bodies,
             dt, t, WIJ):
        t1, t2 = declare('int', 3)

        if s_contact_force_is_boundary[s_idx] == 1.:
            if d_dem_id[d_idx] < s_dem_id[s_idx]:

                t1 = d_total_no_bodies[0] * d_idx

                tmp = d_m[d_idx] / (d_rho[d_idx] * RIJ) * WIJ

                t2 = t1 + s_dem_id[s_idx]
                d_contact_force_normal_tmp_x[t2] += XIJ[0] * tmp
                d_contact_force_normal_tmp_y[t2] += XIJ[1] * tmp
                d_contact_force_normal_tmp_z[t2] += XIJ[2] * tmp

                d_contact_force_normal_wij[t2] += tmp * RIJ

    def post_loop(self, d_idx,
                  d_contact_force_normal_x,
                  d_contact_force_normal_y,
                  d_contact_force_normal_z,
                  d_contact_force_normal_tmp_x,
                  d_contact_force_normal_tmp_y,
                  d_contact_force_normal_tmp_z,
                  d_contact_force_normal_wij,
                  d_total_no_bodies,
                  dt, t):
        i, t1, t2 = declare('int', 3)
        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i
            if d_contact_force_normal_wij[t2] > 1e-12:
                d_contact_force_normal_x[t2] = (d_contact_force_normal_tmp_x[t2] / d_contact_force_normal_wij[t2])
                d_contact_force_normal_y[t2] = (d_contact_force_normal_tmp_y[t2] / d_contact_force_normal_wij[t2])
                d_contact_force_normal_z[t2] = (d_contact_force_normal_tmp_z[t2] / d_contact_force_normal_wij[t2])

                # normalize the normal
                magn = (d_contact_force_normal_x[t2]**2. +
                        d_contact_force_normal_y[t2]**2. +
                        d_contact_force_normal_z[t2]**2.)**0.5

                d_contact_force_normal_x[t2] /= magn
                d_contact_force_normal_y[t2] /= magn
                d_contact_force_normal_z[t2] /= magn
            else:
                d_contact_force_normal_x[t2] = 0.
                d_contact_force_normal_y[t2] = 0.
                d_contact_force_normal_z[t2] = 0.


class ComputeContactForceDistanceAndClosestPointMohseniRB(Equation):
    """Shoya Mohseni Mofidi, Particle Based Numerical Simulation Study of Solid
    Particle Erosion of Ductile Materials Leading to an Erosion Model, Including
    the Particle Shape Effect

    Equation 21 of Materials 2022

    Compute the normals on the rigid body particles (secondary surface) which is
    interacting with wall (primary surface). Here we expect the wall has a
    property identifying its boundary particles. Currently this equation only
    assumes the simulation has only one rigid body with a single wall.

    """
    def initialize(self, d_idx,
                   d_m,
                   d_contact_force_dist,
                   d_contact_force_dist_tmp,
                   d_contact_force_normal_wij,
                   d_normal,
                   d_closest_point_dist_to_source,
                   d_vx_source,
                   d_vy_source,
                   d_vz_source,
                   d_x_source,
                   d_y_source,
                   d_z_source,
                   d_idx_source,
                   d_spacing0,
                   d_total_no_bodies,
                   dt, t):
        i, t1, t2 = declare('int', 3)

        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i
            d_contact_force_dist[t2] = 0.
            d_contact_force_dist_tmp[t2] = 0.
            d_contact_force_normal_wij[t2] = 0.

            d_closest_point_dist_to_source[t2] = 4. * d_spacing0[0]
            d_vx_source[t2] = 0.
            d_vy_source[t2] = 0.
            d_vz_source[t2] = 0.
            d_x_source[t2] = 0.
            d_y_source[t2] = 0.
            d_z_source[t2] = 0.
            d_idx_source[t2] = -1

    def loop(self, d_idx, d_m, d_rho,
             s_idx, s_x, s_y, s_z, s_u, s_v, s_w,
             d_contact_force_normal_x,
             d_contact_force_normal_y,
             d_contact_force_normal_z,
             d_contact_force_normal_wij,
             d_contact_force_dist_tmp,
             d_contact_force_dist,
             s_contact_force_is_boundary,
             d_closest_point_dist_to_source,
             d_vx_source,
             d_vy_source,
             d_vz_source,
             d_x_source,
             d_y_source,
             d_z_source,
             d_idx_source,
             d_dem_id,
             s_dem_id,
             # s_E,
             # s_G,
             # s_nu,
             d_total_no_bodies,
             d_spacing0,
             d_dem_id_source,
             # d_contact_force_weight_denominator,
             # d_E_source,
             # d_G_source,
             # d_nu_source,
             dt, t, WIJ, RIJ, XIJ):
        i, t1, t2, t3 = declare('int', 4)

        if s_contact_force_is_boundary[s_idx] == 1.:
            if d_dem_id[d_idx] != s_dem_id[s_idx]:
                t1 = d_total_no_bodies[0] * d_idx
                t2 = t1 + s_dem_id[s_idx]

                d_dem_id_source[t2] = s_dem_id[s_idx]

                tmp = d_m[d_idx] / (d_rho[d_idx]) * WIJ
                tmp_1 = (d_contact_force_normal_x[t2] * XIJ[0] +
                         d_contact_force_normal_y[t2] * XIJ[1] +
                         d_contact_force_normal_z[t2] * XIJ[2])
                d_contact_force_dist_tmp[t2] += tmp_1 * tmp

                d_contact_force_normal_wij[t2] += tmp

                if RIJ < d_closest_point_dist_to_source[t2]:
                    d_closest_point_dist_to_source[t2] = RIJ
                    d_x_source[t2] = s_x[s_idx]
                    d_y_source[t2] = s_y[s_idx]
                    d_z_source[t2] = s_z[s_idx]
                    d_vx_source[t2] = s_u[s_idx]
                    d_vy_source[t2] = s_v[s_idx]
                    d_vz_source[t2] = s_w[s_idx]
                    d_idx_source[t2] = s_idx
                    # d_E_source[t2] = s_E[s_idx]
                    # d_G_source[t2] = s_G[s_idx]
                    # d_nu_source[t2] = s_nu[s_idx]

                if RIJ < d_spacing0[0]:
                    # Check the direction
                    dx = XIJ[0] / RIJ
                    dy = XIJ[1] / RIJ
                    dz = XIJ[2] / RIJ
                    tmp = -(d_contact_force_normal_x[t2] * dx +
                            d_contact_force_normal_y[t2] * dy +
                            d_contact_force_normal_z[t2] * dz)

                    # d_contact_force_weight_denominator[t2] += tmp

    def post_loop(self, d_idx,
                  d_contact_force_dist_tmp,
                  d_contact_force_dist,
                  d_contact_force_normal_wij,
                  d_total_no_bodies,
                  dt, t):
        i, t1, t2 = declare('int', 3)
        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i

            if d_contact_force_normal_wij[t2] > 1e-12:
                d_contact_force_dist[t2] = (d_contact_force_dist_tmp[t2] /
                                            d_contact_force_normal_wij[t2])
            else:
                d_contact_force_dist[t2] = 0.


class ComputeContactForceMohseniRB(Equation):
    def __init__(self, dest, sources, kr=1e7, kf=1e5, fric_coeff=0.5):
        self.fric_coeff = fric_coeff
        self.kr = kr
        self.kf = kf
        super(ComputeContactForceMohseniRB, self).__init__(dest, sources)

    def post_loop(self,
                  d_idx,
                  d_m,
                  d_body_id,
                  d_contact_force_normal_x,
                  d_contact_force_normal_y,
                  d_contact_force_normal_z,
                  d_contact_force_dist,
                  d_contact_force_dist_tmp,
                  d_contact_force_normal_wij,
                  d_overlap,
                  d_u,
                  d_v,
                  d_w,
                  d_x,
                  d_y,
                  d_z,
                  d_fx,
                  d_fy,
                  d_fz,
                  d_ft_x,
                  d_ft_y,
                  d_ft_z,
                  d_fn_x,
                  d_fn_y,
                  d_fn_z,
                  d_delta_lt_x,
                  d_delta_lt_y,
                  d_delta_lt_z,
                  d_vx_source,
                  d_vy_source,
                  d_vz_source,
                  d_x_source,
                  d_y_source,
                  d_z_source,
                  d_dem_id_source,
                  d_ti_x,
                  d_ti_y,
                  d_ti_z,
                  # d_eta,
                  d_spacing0,
                  d_total_no_bodies,
                  # d_E,
                  # d_G,
                  # d_nu,
                  # d_E_source,
                  # d_G_source,
                  # d_nu_source,
                  dt, t):
        i, t1, t2 = declare('int', 3)
        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i
            overlap = d_spacing0[0] - d_contact_force_dist[t2]
            if overlap > 0. and overlap != d_spacing0[0]:
                # ===============================
                # compute the stiffness coefficients
                # ===============================
                # tmp_5 = (1. - d_nu[d_idx]**2.) / d_E[d_idx]
                # tmp_6 = (1. - d_nu_source[t2]**2.) / d_E_source[t2]
                # E_star = 1 / (tmp_5 + tmp_6)

                # kr = 1.7 * E_star * d_spacing0[0]**2.
                # kr = 1.7 * E_star * d_spacing0[0]
                kr = self.kr

                # tmp_5 = (1. - d_nu[d_idx]) / d_G[d_idx]
                # tmp_6 = (1. - d_nu_source[t2]) / d_G_source[t2]
                # tmp_7 = (1. - d_nu[d_idx]/2.) / d_G[d_idx]
                # tmp_8 = (1. - d_nu_source[t2]/2.) / d_G_source[t2]
                # tmp_9 = (tmp_5 + tmp_6) / (tmp_7 + tmp_8)

                kf = self.kf
                # kf = 2. / 7. * kr

                vij_x = d_u[d_idx] - d_vx_source[t2]
                vij_y = d_v[d_idx] - d_vy_source[t2]
                vij_z = d_w[d_idx] - d_vz_source[t2]

                # the tangential vector is
                ni_x = d_contact_force_normal_x[t2]
                ni_y = d_contact_force_normal_y[t2]
                ni_z = d_contact_force_normal_z[t2]

                vij_dot_ni = vij_x * ni_x + vij_y * ni_y + vij_z * ni_z

                d_overlap[t2] = overlap
                tmp = kr * overlap

                # ===============================
                # compute the damping coefficient
                # ===============================
                # eta = d_eta[d_body_id[d_idx] * d_total_no_bodies[0] + d_dem_id_source[t2]]
                # eta = eta * kr**0.5
                # FixMe
                eta = 0.
                # ===============================
                # compute the damping coefficient
                # ===============================

                fn_x = (tmp - eta * vij_dot_ni) * ni_x
                fn_y = (tmp - eta * vij_dot_ni) * ni_y
                fn_z = (tmp - eta * vij_dot_ni) * ni_z

                # check if there is relative motion
                vij_magn = (vij_x**2. + vij_y**2. + vij_z**2.)**0.5
                if vij_magn < 1e-12:
                    d_delta_lt_x[t2] = 0.
                    d_delta_lt_y[t2] = 0.
                    d_delta_lt_z[t2] = 0.

                    d_ft_x[t2] = 0.
                    d_ft_y[t2] = 0.
                    d_ft_z[t2] = 0.

                    d_ti_x[t2] = 0.
                    d_ti_y[t2] = 0.
                    d_ti_z[t2] = 0.

                else:
                    tx_tmp = vij_x - ni_x * vij_dot_ni
                    ty_tmp = vij_y - ni_y * vij_dot_ni
                    tz_tmp = vij_z - ni_z * vij_dot_ni

                    ti_magn = (tx_tmp**2. + ty_tmp**2. + tz_tmp**2.)**0.5

                    ti_x = 0.
                    ti_y = 0.
                    ti_z = 0.

                    if ti_magn > 1e-12:
                        ti_x = tx_tmp / ti_magn
                        ti_y = ty_tmp / ti_magn
                        ti_z = tz_tmp / ti_magn

                    # save the normals to output and view in viewer
                    d_ti_x[d_idx] = ti_x
                    d_ti_y[d_idx] = ti_y
                    d_ti_z[d_idx] = ti_z

                    # this is correct
                    delta_lt_x_star = d_delta_lt_x[t2] + vij_x * dt
                    delta_lt_y_star = d_delta_lt_y[t2] + vij_y * dt
                    delta_lt_z_star = d_delta_lt_z[t2] + vij_z * dt

                    delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                       delta_lt_y_star * ti_y +
                                       delta_lt_z_star * ti_z)

                    d_delta_lt_x[t2] = delta_lt_dot_ti * ti_x
                    d_delta_lt_y[t2] = delta_lt_dot_ti * ti_y
                    d_delta_lt_z[t2] = delta_lt_dot_ti * ti_z

                    ft_x_star = -kf * d_delta_lt_x[t2]
                    ft_y_star = -kf * d_delta_lt_y[t2]
                    ft_z_star = -kf * d_delta_lt_z[t2]

                    ft_magn = (ft_x_star**2. + ft_y_star**2. + ft_z_star**2.)**0.5
                    fn_magn = (fn_x**2. + fn_y**2. + fn_z**2.)**0.5

                    ft_magn_star = min(self.fric_coeff * fn_magn, ft_magn)
                    # compute the tangential force, by equation 27
                    d_ft_x[t2] = -ft_magn_star * ti_x
                    d_ft_y[t2] = -ft_magn_star * ti_y
                    d_ft_z[t2] = -ft_magn_star * ti_z

                    # reset the spring length
                    modified_delta_lt_x = -d_ft_x[t2] / kf
                    modified_delta_lt_y = -d_ft_y[t2] / kf
                    modified_delta_lt_z = -d_ft_z[t2] / kf

                    lt_magn = (modified_delta_lt_x**2. + modified_delta_lt_y**2. +
                               modified_delta_lt_z**2.)**0.5

                    d_delta_lt_x[t2] = modified_delta_lt_x / lt_magn
                    d_delta_lt_y[t2] = modified_delta_lt_y / lt_magn
                    d_delta_lt_z[t2] = modified_delta_lt_z / lt_magn

                    # repulsive force
                    d_fn_x[t2] = fn_x
                    d_fn_y[t2] = fn_y
                    d_fn_z[t2] = fn_z

            else:
                d_overlap[t2] = 0.
                d_ft_x[t2] = 0.
                d_ft_y[t2] = 0.
                d_ft_z[t2] = 0.
                # reset the spring length
                d_delta_lt_x[t2] = 0.
                d_delta_lt_y[t2] = 0.
                d_delta_lt_z[t2] = 0.

                # reset the normal force
                d_fn_x[t2] = 0.
                d_fn_y[t2] = 0.
                d_fn_z[t2] = 0.

            # add the force
            d_fx[d_idx] += d_fn_x[t2] + d_ft_x[t2]
            d_fy[d_idx] += d_fn_y[t2] + d_ft_y[t2]
            d_fz[d_idx] += d_fn_z[t2] + d_ft_z[t2]


class TransferContactForceMohseniRB(Equation):
    def loop(self, d_idx, d_m, d_rho,
             s_idx, s_x, s_y, s_z, s_u, s_v, s_w,
             d_contact_force_normal_x,
             d_contact_force_normal_y,
             d_contact_force_normal_z,
             d_contact_force_normal_wij,
             d_contact_force_dist_tmp,
             d_contact_force_dist,
             s_contact_force_is_boundary,
             d_closest_point_dist_to_source,
             d_vx_source,
             d_vy_source,
             d_vz_source,
             d_x_source,
             d_y_source,
             d_z_source,
             d_dem_id,
             s_dem_id,
             d_total_no_bodies,
             d_spacing0,
             d_normal, dt, t, WIJ, RIJ, XIJ,
             d_fn_x,
             d_fn_y,
             d_fn_z,
             d_ft_x,
             d_ft_y,
             d_ft_z,
             s_fn_x,
             s_fn_y,
             s_fn_z,
             s_ft_x,
             s_ft_y,
             s_ft_z,
             s_idx_source,
             s_contact_force_normal_x,
             s_contact_force_normal_y,
             s_contact_force_normal_z,
             # s_contact_force_weight_denominator,
             d_fx,
             d_fy,
             d_fz,
             d_contact_force_is_boundary):
        i, t1, t2, t3, t4 = declare('int', 5)

        if d_contact_force_is_boundary[d_idx] == 1. and s_contact_force_is_boundary[s_idx] == 1. and d_dem_id[d_idx] > s_dem_id[s_idx]:
            t1 = d_total_no_bodies[0] * s_idx
            t2 = t1 + d_dem_id[d_idx]

            if s_idx_source[t2] == d_idx:
                d_fx[d_idx] -= (s_fn_x[t2] + s_ft_x[t2])
                d_fy[d_idx] -= (s_fn_y[t2] + s_ft_y[t2])
                d_fz[d_idx] -= (s_fn_z[t2] + s_ft_z[t2])
