"""
Basic Equations for Solid Mechanics
###################################

References
----------
"""

from numpy import sqrt, fabs
import numpy
from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from boundary_particles import (ComputeNormalsEDAC, SmoothNormalsEDAC,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.wc.transport_velocity import SetWallVelocity

from pysph.examples.solid_mech.impact import add_properties
from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.integrator import Integrator

import numpy as np
from math import sqrt, acos, log, exp
from math import pi as M_PI


def get_youngs_mod_from_G_nu(G, nu):
    return 2. * G * (1. + nu)


def get_youngs_mod_from_K_G(K, G):
    return 9. * K * G / (3. * K + G)


def get_poisson_ratio_from_E_G(e, g):
    return e / (2. * g) - 1.


# ===========================
# Equations for ULSPH by Gray
# ===========================

class ElasticSolidContinuityEquationU(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_u, d_v, d_w, s_idx, s_m, s_u,
             s_v, s_w, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ElasticSolidContinuityEquationUSolid(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_u, d_v, d_w, s_idx, s_m, s_ub, s_vb, s_wb,
             DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij

# ===========================
# Equations for ULSPH by Gray
# ===========================


class ElasticSolidContinuityEquationUhat(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_uhat, d_vhat, d_what, s_idx, s_m, s_uhat,
             s_vhat, s_what, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ElasticSolidContinuityEquationETVFCorrection(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho, s_m, s_u, s_v, s_w, s_uhat, s_vhat, s_what, DWIJ):
        tmp0 = s_rho[s_idx] * (s_uhat[s_idx] - s_u[s_idx]) - d_rho[d_idx] * (
            d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho[s_idx] * (s_vhat[s_idx] - s_v[s_idx]) - d_rho[d_idx] * (
            d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho[s_idx] * (s_what[s_idx] - s_w[s_idx]) - d_rho[d_idx] * (
            d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m[s_idx] / s_rho[s_idx] * vijdotdwij


class SetHIJForInsideParticles(Equation):
    def __init__(self, dest, sources, kernel_factor):
        # depends on the kernel used
        self.kernel_factor = kernel_factor
        super(SetHIJForInsideParticles, self).__init__(dest, sources)

    def initialize(self, d_idx, d_h_b, d_h):
        # back ground pressure h (This will be the usual h value)
        d_h_b[d_idx] = d_h[d_idx]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_is_boundary,
                 d_normal, d_normal_norm, d_h_b, s_m, s_x, s_y, s_z, s_h,
                 s_is_boundary, SPH_KERNEL, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('int')
        xij = declare('matrix(3)')

        # if the particle is boundary set it's h_b to be zero
        if d_is_boundary[d_idx] == 1:
            d_h_b[d_idx] = 0.
        # if it is not the boundary then set its h_b according to the minimum
        # distance to the boundary particle
        else:
            # get the minimum distance to the boundary particle
            min_dist = 0
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                if s_is_boundary[s_idx] == 1:
                    # find the distance
                    xij[0] = d_x[d_idx] - s_x[s_idx]
                    xij[1] = d_y[d_idx] - s_y[s_idx]
                    xij[2] = d_z[d_idx] - s_z[s_idx]
                    rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)

                    if rij > min_dist:
                        min_dist = rij

            # doing this out of desperation
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                if s_is_boundary[s_idx] == 1:
                    # find the distance
                    xij[0] = d_x[d_idx] - s_x[s_idx]
                    xij[1] = d_y[d_idx] - s_y[s_idx]
                    xij[2] = d_z[d_idx] - s_z[s_idx]
                    rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)

                    if rij < min_dist:
                        min_dist = rij

            if min_dist > 0.:
                d_h_b[d_idx] = min_dist / self.kernel_factor + min_dist / 50


class ElasticSolidMomentumEquation(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p, d_s00, d_s01,
             d_s02, d_s11, d_s12, d_s22, s_s00, s_s01, s_s02, s_s11, s_s12,
             s_s22, d_au, d_av, d_aw, WIJ, DWIJ):
        pa = d_p[d_idx]
        pb = s_p[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        rhoa21 = 1. / (rhoa * rhoa)
        rhob21 = 1. / (rhob * rhob)

        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        s00b = s_s00[s_idx]
        s01b = s_s01[s_idx]
        s02b = s_s02[s_idx]

        s10b = s_s01[s_idx]
        s11b = s_s11[s_idx]
        s12b = s_s12[s_idx]

        s20b = s_s02[s_idx]
        s21b = s_s12[s_idx]
        s22b = s_s22[s_idx]

        # Add pressure to the deviatoric components
        s00a = s00a - pa
        s00b = s00b - pb

        s11a = s11a - pa
        s11b = s11b - pb

        s22a = s22a - pa
        s22b = s22b - pb

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += (mb * (s00a * rhoa21 + s00b * rhob21) * DWIJ[0] + mb *
                        (s01a * rhoa21 + s01b * rhob21) * DWIJ[1] + mb *
                        (s02a * rhoa21 + s02b * rhob21) * DWIJ[2])

        d_av[d_idx] += (mb * (s10a * rhoa21 + s10b * rhob21) * DWIJ[0] + mb *
                        (s11a * rhoa21 + s11b * rhob21) * DWIJ[1] + mb *
                        (s12a * rhoa21 + s12b * rhob21) * DWIJ[2])

        d_aw[d_idx] += (mb * (s20a * rhoa21 + s20b * rhob21) * DWIJ[0] + mb *
                        (s21a * rhoa21 + s21b * rhob21) * DWIJ[1] + mb *
                        (s22a * rhoa21 + s22b * rhob21) * DWIJ[2])


class ElasticSolidComputeAuHatETVFSun2019(Equation):
    def __init__(self, dest, sources, mach_no):
        self.mach_no = mach_no
        super(ElasticSolidComputeAuHatETVFSun2019, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_cs, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_cs[d_idx] * 2. * d_h[d_idx] / dt

        tmp1 = s_m[s_idx] / s_rho[s_idx]

        d_auhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[0]
        d_avhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[1]
        d_awhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # first put a clearance
        magn_auhat = sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
                          d_avhat[d_idx] * d_avhat[d_idx] +
                          d_awhat[d_idx] * d_awhat[d_idx])

        if magn_auhat > 1e-12:
            # Now apply the filter for boundary particles and adjacent particles
            if d_h_b[d_idx] < d_h[d_idx]:
                if d_is_boundary[d_idx] == 1:
                    # since it is boundary make its shifting acceleration zero
                    d_auhat[d_idx] = 0.
                    d_avhat[d_idx] = 0.
                    d_awhat[d_idx] = 0.
                else:
                    # implies this is a particle adjacent to boundary particle

                    # check if the particle is going away from the continuum
                    # or into the continuum
                    au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
                                     d_avhat[d_idx] * d_normal[idx3 + 1] +
                                     d_awhat[d_idx] * d_normal[idx3 + 2])

                    # remove the normal acceleration component
                    if au_dot_normal > 0.:
                        d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                        d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                        d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class AdamiBoundaryConditionExtrapolateNoSlip(Equation):
    """
    Taken from

    [1] A numerical study on ice failure process and ice-ship interactions by
    Smoothed Particle Hydrodynamics
    [2] Adami 2012 boundary conditions paper.
    [3] LOQUAT: an open-source GPU-accelerated SPH solver for geotechnical modeling

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(AdamiBoundaryConditionExtrapolateNoSlip, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_p, d_wij, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22, d_is_solid_boundary):
        if d_is_solid_boundary[d_idx] == 1.:
            d_s00[d_idx] = 0.0
            d_s01[d_idx] = 0.0
            d_s02[d_idx] = 0.0
            d_s11[d_idx] = 0.0
            d_s12[d_idx] = 0.0
            d_s22[d_idx] = 0.0
            d_p[d_idx] = 0.0
            d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             d_au, d_av, d_aw, s_idx, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22,
             s_p, s_rho, WIJ, XIJ, d_is_solid_boundary, s_is_solid_boundary):
        if d_is_solid_boundary[d_idx] == 1.:
            if s_is_solid_boundary[s_idx] == 0.:
                d_s00[d_idx] += s_s00[s_idx] * WIJ
                d_s01[d_idx] += s_s01[s_idx] * WIJ
                d_s02[d_idx] += s_s02[s_idx] * WIJ
                d_s11[d_idx] += s_s11[s_idx] * WIJ
                d_s12[d_idx] += s_s12[s_idx] * WIJ
                d_s22[d_idx] += s_s22[s_idx] * WIJ

                gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
                    (self.gy - d_av[d_idx])*XIJ[1] + \
                    (self.gz - d_aw[d_idx])*XIJ[2]

                d_p[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx]*gdotxij*WIJ

                # denominator of Eq. (27)
                d_wij[d_idx] += WIJ

    def post_loop(self, d_wij, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_p, d_is_solid_boundary):
        # extrapolated pressure at the ghost particle
        if d_is_solid_boundary[d_idx] == 1.:
            if d_wij[d_idx] > 1e-14:
                d_s00[d_idx] /= d_wij[d_idx]
                d_s01[d_idx] /= d_wij[d_idx]
                d_s02[d_idx] /= d_wij[d_idx]
                d_s11[d_idx] /= d_wij[d_idx]
                d_s12[d_idx] /= d_wij[d_idx]
                d_s22[d_idx] /= d_wij[d_idx]

                d_p[d_idx] /= d_wij[d_idx]


class AdamiBoundaryConditionExtrapolateFreeSlip(Equation):
    """
    Taken from

    [1] A numerical study on ice failure process and ice-ship interactions by
    Smoothed Particle Hydrodynamics
    [2] Adami 2012 boundary conditions paper.
    [3] LOQUAT: an open-source GPU-accelerated SPH solver for geotechnical modeling

    """
    def initialize(self, d_idx, d_p, d_wij, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22):
        d_s00[d_idx] = 0.0
        d_s01[d_idx] = 0.0
        d_s02[d_idx] = 0.0
        d_s11[d_idx] = 0.0
        d_s12[d_idx] = 0.0
        d_s22[d_idx] = 0.0
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             s_idx, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22, s_p, WIJ):
        d_s00[d_idx] += s_s00[s_idx] * WIJ
        d_s01[d_idx] -= s_s01[s_idx] * WIJ
        d_s02[d_idx] -= s_s02[s_idx] * WIJ
        d_s11[d_idx] += s_s11[s_idx] * WIJ
        d_s12[d_idx] -= s_s12[s_idx] * WIJ
        d_s22[d_idx] += s_s22[s_idx] * WIJ

        d_p[d_idx] += s_p[s_idx] * WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_wij, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_s00[d_idx] /= d_wij[d_idx]
            d_s01[d_idx] /= d_wij[d_idx]
            d_s02[d_idx] /= d_wij[d_idx]
            d_s11[d_idx] /= d_wij[d_idx]
            d_s12[d_idx] /= d_wij[d_idx]
            d_s22[d_idx] /= d_wij[d_idx]

            d_p[d_idx] /= d_wij[d_idx]


class ComputePrincipalStress2D(Equation):
    def initialize(self, d_idx, d_sigma_1, d_sigma_2, d_sigma00, d_sigma01,
                   d_sigma02, d_sigma11, d_sigma12, d_sigma22):
        # https://www.ecourses.ou.edu/cgi-bin/eBook.cgi?doc=&topic=me&chap_sec=07.2&page=theory
        tmp1 = (d_sigma00[d_idx] + d_sigma11[d_idx]) / 2

        tmp2 = (d_sigma00[d_idx] - d_sigma11[d_idx]) / 2

        tmp3 = sqrt(tmp2**2. + d_sigma01[d_idx]**2.)

        d_sigma_1[d_idx] = tmp1 + tmp3
        d_sigma_2[d_idx] = tmp1 - tmp3


class HookesDeviatoricStressRate(Equation):
    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12,
                   d_as22):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as02[d_idx] = 0.0

        d_as11[d_idx] = 0.0
        d_as12[d_idx] = 0.0

        d_as22[d_idx] = 0.0

    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_v00,
             d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_v22, d_as00,
             d_as01, d_as02, d_as11, d_as12, d_as22,
             d_G, d_plastic_strain_rate):
        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v02 = d_v02[d_idx]

        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]
        v12 = d_v12[d_idx]

        v20 = d_v20[d_idx]
        v21 = d_v21[d_idx]
        v22 = d_v22[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s02 = d_s02[d_idx]

        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]
        s12 = d_s12[d_idx]

        s20 = d_s02[d_idx]
        s21 = d_s12[d_idx]
        s22 = d_s22[d_idx]

        # strain rate tensor is symmetric
        eps00 = v00
        eps01 = 0.5 * (v01 + v10)
        eps02 = 0.5 * (v02 + v20)

        eps10 = eps01
        eps11 = v11
        eps12 = 0.5 * (v12 + v21)

        eps20 = eps02
        eps21 = eps12
        eps22 = v22

        tmp = (eps00*eps00 + eps01*eps01 + eps02*eps02 +
               eps10*eps10 + eps11*eps11 + eps12*eps12 +
               eps20*eps20 + eps21*eps21 + eps22*eps22)

        d_plastic_strain_rate[d_idx] = sqrt(2. / 3. * tmp)

        # rotation tensor is asymmetric
        omega00 = 0.0
        omega01 = 0.5 * (v01 - v10)
        omega02 = 0.5 * (v02 - v20)

        omega10 = -omega01
        omega11 = 0.0
        omega12 = 0.5 * (v12 - v21)

        omega20 = -omega02
        omega21 = -omega12
        omega22 = 0.0

        tmp = 2.0 * d_G[d_idx]
        trace = 1.0 / 3.0 * (eps00 + eps11 + eps22)

        # S_00
        d_as00[d_idx] = tmp*( eps00 - trace ) + \
                        ( s00*omega00 + s01*omega01 + s02*omega02) + \
                        ( s00*omega00 + s10*omega01 + s20*omega02)

        # S_01
        d_as01[d_idx] = tmp*(eps01) + \
                        ( s00*omega10 + s01*omega11 + s02*omega12) + \
                        ( s01*omega00 + s11*omega01 + s21*omega02)

        # S_02
        d_as02[d_idx] = tmp*eps02 + \
                        (s00*omega20 + s01*omega21 + s02*omega22) + \
                        (s02*omega00 + s12*omega01 + s22*omega02)

        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
                        (s10*omega10 + s11*omega11 + s12*omega12) + \
                        (s01*omega10 + s11*omega11 + s21*omega12)

        # S_12
        d_as12[d_idx] = tmp*eps12 + \
                        (s10*omega20 + s11*omega21 + s12*omega22) + \
                        (s02*omega10 + s12*omega11 + s22*omega12)

        # S_22
        d_as22[d_idx] = tmp*(eps22 - trace) + \
                        (s20*omega20 + s21*omega21 + s22*omega22) + \
                        (s02*omega20 + s12*omega21 + s22*omega22)


class SolidMechStep(IntegratorStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
               d_what, d_auhat, d_avhat, d_awhat, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

    def stage2(self, d_idx, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, dt):
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s00[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s01[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s02[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s11[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s12[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s22[d_idx] + dt * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]

        d_rho[d_idx] += dt * d_arho[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class GTVFSolidMechStepEDAC(SolidMechStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, d_ap, d_e,
               d_ae, d_T, d_G, d_specific_heat, d_JC_T_room,
               d_eff_plastic_strain, d_plastic_strain_rate, d_JC_A, d_JC_B,
               d_JC_n, d_JC_C, d_JC_psr_0, d_JC_T_melt, d_JC_m,
               d_yield_stress, d_damage_1, d_damage_2, d_damage_3,
               d_damage_4, d_damage_5, d_epsilon_failure, d_damage_D, dt):
        # find the Johnson Cook yield stress value
        # compute yield stress (sigma_y)
        # 2. A smoothed particle hydrodynamics (SPH) model for simulating
        # surface erosion by impacts of foreign particles, by X.W. Dong
        eps = d_eff_plastic_strain[d_idx]
        psr = d_plastic_strain_rate[d_idx]
        term_1 = (d_JC_A[0] + d_JC_B[0] * eps**d_JC_n[0])
        term_2 = (1 + d_JC_C[0] * log(psr / d_JC_psr_0[0]))
        # equation 25 of [2]
        t_room = d_JC_T_room[0]
        # normalized temperature T^*
        # equation 24 of [2]
        t_star = (d_T[d_idx] - t_room) / (d_JC_T_melt[0] - t_room)
        term_3 = (1. - t_star**d_JC_m[0])
        d_yield_stress[d_idx] = term_1 * term_2 * term_3

        # Strain failure (Equation 27 Dong2016)
        D1 = d_damage_1[0]
        D2 = d_damage_2[0]
        D3 = d_damage_3[0]
        D4 = d_damage_4[0]
        D5 = d_damage_5[0]

        # term_1 = (D1 + D2 * exp(D3 * sigma_star))
        # term_2 = (1 + D4 * log(psr / d_JC_psr_0[0]))
        # # equation 25 of [2]
        # t_room = d_JC_T_room[0]
        # # normalized temperature T^*
        # # equation 24 of [2]
        # t_star = (d_T[d_idx] - t_room) / (d_JC_T_melt[0] - t_room)
        # term_3 = (1. + D5 * t_star)
        # d_epsilon_failure[d_idx] = term_1 * term_2 * term_3

        # Update the positions
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # Update the density
        d_rho[d_idx] += dt * d_arho[d_idx]

        # Update the pressure
        d_p[d_idx] += dt * d_ap[d_idx]

        # Update the energy
        d_e[d_idx] += dt * d_ae[d_idx]
        d_T[d_idx] = d_e[d_idx] / d_specific_heat[0] + d_JC_T_room[0]

        # update deviatoric stress components
        s00_star = d_s00[d_idx] + dt * d_as00[d_idx]
        s01_star = d_s01[d_idx] + dt * d_as01[d_idx]
        s02_star = d_s02[d_idx] + dt * d_as02[d_idx]
        s11_star = d_s11[d_idx] + dt * d_as11[d_idx]
        s12_star = d_s12[d_idx] + dt * d_as12[d_idx]
        s22_star = d_s22[d_idx] + dt * d_as22[d_idx]

        sij_sij = (
            s00_star * s00_star + s01_star * s01_star +
            s02_star * s02_star + s01_star * s01_star +
            s11_star * s11_star + s12_star * s12_star +
            s02_star * s02_star + s12_star * s12_star +
            s22_star * s22_star)
        s_star = sqrt(3./2. * sij_sij)

        f_y = 1.
        if s_star > 0.:
            f_y = min(d_yield_stress[d_idx] / s_star, 1.)

        # Deviatoric stress is updated
        d_s00[d_idx] = f_y * s00_star
        d_s01[d_idx] = f_y * s01_star
        d_s02[d_idx] = f_y * s02_star
        d_s11[d_idx] = f_y * s11_star
        d_s12[d_idx] = f_y * s12_star
        d_s22[d_idx] = f_y * s22_star

        # Find the plastic strain increment
        delta_eps = 1. / 3. * (1. - f_y) / d_G[d_idx] * s_star
        d_eff_plastic_strain[d_idx] += delta_eps

        # # Compute the damage D (eq 26 Dong2016)
        # d_damage_D[d_idx] = (d_eff_plastic_strain[d_idx] /
        #                      d_epsilon_failure[d_idx])
        # if d_damage_D[d_idx] > 1.:
        #     d_s00[d_idx] = 0.
        #     d_s01[d_idx] = 0.
        #     d_s02[d_idx] = 0.
        #     d_s11[d_idx] = 0.
        #     d_s12[d_idx] = 0.
        #     d_s22[d_idx] = 0.

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


class GTVFSolidMechStepEDACTaylorBar(SolidMechStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, d_ap, d_e,
               d_ae, d_T, d_G, d_specific_heat, d_JC_T_room,
               d_eff_plastic_strain, d_plastic_strain_rate, d_JC_A, d_JC_B,
               d_JC_n, d_JC_C, d_JC_psr_0, d_JC_T_melt, d_JC_m,
               d_yield_stress, dt):
        # find the Johnson Cook yield stress value
        # compute yield stress (sigma_y)
        # 2. A smoothed particle hydrodynamics (SPH) model for simulating
        # surface erosion by impacts of foreign particles, by X.W. Dong
        eps = d_eff_plastic_strain[d_idx]
        psr = d_plastic_strain_rate[d_idx]
        term_1 = (d_JC_A[0] + d_JC_B[0] * eps**d_JC_n[0])
        term_2 = (1 + d_JC_C[0] * log(psr / d_JC_psr_0[0]))
        # equation 25 of [2]
        t_room = d_JC_T_room[0]
        # normalized temperature T^*
        # equation 24 of [2]
        t_star = (d_T[d_idx] - t_room) / (d_JC_T_melt[0] - t_room)
        term_3 = (1. - t_star**d_JC_m[0])
        d_yield_stress[d_idx] = term_1 * term_2 * term_3

        # Update the positions
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # Update the density
        d_rho[d_idx] += dt * d_arho[d_idx]

        # Update the pressure
        d_p[d_idx] += dt * d_ap[d_idx]

        # Update the energy
        d_e[d_idx] += dt * d_ae[d_idx]
        d_T[d_idx] = d_e[d_idx] / d_specific_heat[0] + d_JC_T_room[0]

        # update deviatoric stress components
        s00_star = d_s00[d_idx] + dt * d_as00[d_idx]
        s01_star = d_s01[d_idx] + dt * d_as01[d_idx]
        s02_star = d_s02[d_idx] + dt * d_as02[d_idx]
        s11_star = d_s11[d_idx] + dt * d_as11[d_idx]
        s12_star = d_s12[d_idx] + dt * d_as12[d_idx]
        s22_star = d_s22[d_idx] + dt * d_as22[d_idx]

        sij_sij = (
            s00_star * s00_star + s01_star * s01_star +
            s02_star * s02_star + s01_star * s01_star +
            s11_star * s11_star + s12_star * s12_star +
            s02_star * s02_star + s12_star * s12_star +
            s22_star * s22_star)
        s_star = sqrt(3./2. * sij_sij)

        f_y = 1.
        if s_star > 0.:
            f_y = min(d_yield_stress[d_idx] / s_star, 1.)

        # Deviatoric stress is updated
        d_s00[d_idx] = f_y * s00_star
        d_s01[d_idx] = f_y * s01_star
        d_s02[d_idx] = f_y * s02_star
        d_s11[d_idx] = f_y * s11_star
        d_s12[d_idx] = f_y * s12_star
        d_s22[d_idx] = f_y * s22_star

        # Find the plastic strain increment
        delta_eps = 1. / 3. * (1. - f_y) / d_G[d_idx] * s_star
        d_eff_plastic_strain[d_idx] += delta_eps

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


class MakeAuhatZero(Equation):
    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.
        d_avhat[d_idx] = 0.
        d_awhat[d_idx] = 0.


class ElasticSolidContinuityEquationUhatSolid(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_uhat, d_vhat, d_what, s_idx, s_m, s_ubhat,
             s_vbhat, s_wbhat, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ElasticSolidContinuityEquationETVFCorrectionSolid(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho, s_m, s_ub, s_vb, s_wb, s_ubhat, s_vbhat, s_wbhat,
             DWIJ):
        tmp0 = s_rho[s_idx] * (s_ubhat[s_idx] - s_ub[s_idx]) - d_rho[d_idx] * (
            d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho[s_idx] * (s_vbhat[s_idx] - s_vb[s_idx]) - d_rho[d_idx] * (
            d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho[s_idx] * (s_wbhat[s_idx] - s_wb[s_idx]) - d_rho[d_idx] * (
            d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m[s_idx] / s_rho[s_idx] * vijdotdwij


class VelocityGradient2DSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v10, d_v11, d_u,
             d_v, d_w, s_ub, s_vb, s_wb, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class ElasticSolidSetWallVelocityNoSlipU(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_ub, d_vb, d_wb, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

        d_ub[d_idx] = 0.0
        d_vb[d_idx] = 0.0
        d_wb[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_uf, d_vf, d_wf, s_u, s_v, s_w,
             d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ub, d_vb, d_wb, d_u,
                  d_v, d_w, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ub[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vb[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wb[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]

        vn = (d_ub[d_idx]*d_normal[idx3] + d_vb[d_idx]*d_normal[idx3+1]
              + d_wb[d_idx]*d_normal[idx3+2])
        if vn < 0:
            d_ub[d_idx] -= vn*d_normal[idx3]
            d_vb[d_idx] -= vn*d_normal[idx3+1]
            d_wb[d_idx] -= vn*d_normal[idx3+2]


class ElasticSolidSetWallVelocityNoSlipUhat(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf,
                   d_ubhat, d_vbhat, d_wbhat,
                   d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

        d_ubhat[d_idx] = 0.0
        d_vbhat[d_idx] = 0.0
        d_wbhat[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_uf, d_vf, d_wf, s_u, s_v, s_w,
             d_ubhat, d_vbhat, d_wbhat, s_uhat, s_vhat, s_what,
             d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_uhat[s_idx] * WIJ
        d_vf[d_idx] += s_vhat[s_idx] * WIJ
        d_wf[d_idx] += s_what[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ubhat, d_vbhat,
                  d_wbhat, d_uhat, d_vhat, d_what, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ubhat[d_idx] = 2 * d_uhat[d_idx] - d_uf[d_idx]
        d_vbhat[d_idx] = 2 * d_vhat[d_idx] - d_vf[d_idx]
        d_wbhat[d_idx] = 2 * d_what[d_idx] - d_wf[d_idx]

        vn = (d_ubhat[d_idx]*d_normal[idx3] + d_vbhat[d_idx]*d_normal[idx3+1]
              + d_wbhat[d_idx]*d_normal[idx3+2])
        if vn < 0:
            d_ubhat[d_idx] -= vn*d_normal[idx3]
            d_vbhat[d_idx] -= vn*d_normal[idx3+1]
            d_wbhat[d_idx] -= vn*d_normal[idx3+2]


class MonaghanArtificialViscosity(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = d_cs[d_idx]

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class AddGravityToStructure(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(AddGravityToStructure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class BuiFukagawaDampingGraularSPH(Equation):
    def __init__(self, dest, sources, damping_coeff=0.02):
        self.damping_coeff = damping_coeff
        super(BuiFukagawaDampingGraularSPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_h, d_rho,
                   d_E, d_u, d_v, d_w):
        tmp1 = d_rho[d_idx] * d_h[d_idx]**2.
        tmp = self.damping_coeff * (d_E[d_idx] / tmp1)**0.5

        d_au[d_idx] -= tmp * d_u[d_idx]
        d_av[d_idx] -= tmp * d_v[d_idx]
        d_aw[d_idx] -= tmp * d_w[d_idx]


class MakeSolidBoundaryParticlesAccelerationsZero(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw, d_h, d_rho, d_E, d_u, d_v,
                   d_w, d_m, d_uhat, d_vhat, d_what, d_arho, d_s22, d_as00,
                   d_as01, d_as02, d_as11, d_as12, d_as22, d_ap, dt,
                   d_auhat, d_avhat, d_awhat, d_is_solid_boundary):
        if d_is_solid_boundary[d_idx] == 1.:
            d_arho[d_idx] = 0.
            d_au[d_idx] = 0.
            d_av[d_idx] = 0.
            d_aw[d_idx] = 0.

            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.

            d_uhat[d_idx] = 0.
            d_vhat[d_idx] = 0.
            d_what[d_idx] = 0.

            d_u[d_idx] = 0.
            d_v[d_idx] = 0.
            d_w[d_idx] = 0.

            d_ap[d_idx] = 0.

            d_as00[d_idx] = 0.
            d_as01[d_idx] = 0.
            d_as02[d_idx] = 0.
            d_as11[d_idx] = 0.
            d_as12[d_idx] = 0.
            d_as22[d_idx] = 0.


class IsothermalEOS(Equation):
    def loop(self, d_idx, d_rho, d_p, d_cs, d_rho_ref):
        d_p[d_idx] = d_cs[d_idx] * d_cs[d_idx] * (
            d_rho[d_idx] - d_rho_ref[d_idx])


class MieGruneisenEOS(Equation):
    def loop(self, d_idx, d_rho, d_p, d_cs, d_rho_ref):
        d_p[d_idx] = d_cs[d_idx] * d_cs[d_idx] * (
            d_rho[d_idx] - d_rho_ref[d_idx])


def setup_johnson_cook_parameters(pa, JC_A, JC_B, JC_C, JC_n, JC_m,
                                  JC_T_melt):
    pa.add_constant('JC_A', JC_A)
    pa.add_constant('JC_B', JC_B)
    pa.add_constant('JC_C', JC_C)
    pa.add_constant('JC_T_melt', JC_T_melt)
    pa.add_constant('JC_n', JC_n)
    pa.add_constant('JC_m', JC_m)
    pa.add_constant('JC_psr_0', 1.)

    add_properties(pa, 'plastic_strain_rate', 'eff_plastic_strain',
                   'epsilon_failure', 'yield_stress')
    pa.plastic_strain_rate[:] = 0.
    pa.eff_plastic_strain[:] = 0.
    pa.yield_stress[:] = 0.

    pa.add_output_arrays(['eff_plastic_strain', 'yield_stress'])


def setup_damage_parameters(pa, damage_1, damage_2, damage_3, damage_4,
                            damage_5):
    pa.add_constant('damage_1', damage_1)
    pa.add_constant('damage_2', damage_2)
    pa.add_constant('damage_3', damage_3)
    pa.add_constant('damage_4', damage_4)
    pa.add_constant('damage_5', damage_5)

    add_properties(pa, 'is_damaged', 'epsilon_failure', 'damage_D')
    pa.is_damaged[:] = 0.
    pa.epsilon_failure[:] = damage_1

    pa.add_output_arrays(['is_damaged', 'epsilon_failure'])


def setup_mie_gruniesen_parameters(pa, mie_gruneisen_sigma,
                                   mie_gruneisen_S):
    """
    Write the ranges of the expected values

    """
    pa.add_constant('mie_gruneisen_gamma', mie_gruneisen_sigma)
    pa.add_constant('mie_gruneisen_S', mie_gruneisen_S)
