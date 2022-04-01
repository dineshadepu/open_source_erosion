"""
Basic Equations for Solid Mechanics
###################################

References
----------
"""

from numpy import sqrt, fabs, log
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

from pysph.sph.integrator import Integrator

import numpy as np
from math import sqrt, acos
from math import pi as M_PI, exp


def get_nu_from_G_K(G, K):
    return (3. * K - 2. * G) / (6. * K + 2. * G)


def get_youngs_mod_from_G_nu(G, nu):
    return 2. * G * (1 + nu)


def add_properties_stride(pa, stride=1, *props):
    for prop in props:
        pa.add_property(name=prop, stride=stride)


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

class ContinuityEquationUhat(Equation):
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


class ContinuityEquationETVFCorrection(Equation):
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


class VelocityGradient2DUhat(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_uhat, d_vhat, d_what, d_idx, s_idx, s_m, s_rho, d_v00,
             d_v01, d_v10, d_v11, s_uhat, s_vhat, s_what, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DUhat(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_uhat, d_vhat, d_what, s_uhat,
             s_vhat, s_what, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class VelocityGradient2DSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v10, d_v11, d_u,
             d_v, d_w, s_ug, s_vg, s_wg, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ug[s_idx]
        vij[1] = d_v[d_idx] - s_vg[s_idx]
        vij[2] = d_w[d_idx] - s_wg[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_u, d_v, d_w, s_ug, s_vg, s_wg,
             DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ug[s_idx]
        vij[1] = d_v[d_idx] - s_vg[s_idx]
        vij[2] = d_w[d_idx] - s_wg[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class VelocityGradient2DUhatSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_uhat, d_vhat, d_what, d_idx, s_idx, s_m, s_rho, d_v00,
             d_v01, d_v10, d_v11, s_ughat, s_vghat, s_wghat, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ughat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vghat[s_idx]
        vij[2] = d_what[d_idx] - s_wghat[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DUhatSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_uhat, d_vhat, d_what, s_ughat,
             s_vghat, s_wghat, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ughat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vghat[s_idx]
        vij[2] = d_what[d_idx] - s_wghat[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class SetHIJForInsideParticles(Equation):
    def __init__(self, dest, sources, h, kernel_factor):
        # h value of usual particle
        self.h = h
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


class MomentumEquationSolids(Equation):
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


class MonaghanArtificialStressCorrection(Equation):
    def loop(self, d_idx, s_idx, s_m, d_r00, d_r01, d_r02, d_r11, d_r12, d_r22,
             s_r00, s_r01, s_r02, s_r11, s_r12, s_r22, d_au, d_av, d_aw,
             d_wdeltap, d_n, WIJ, DWIJ):

        r00a = d_r00[d_idx]
        r01a = d_r01[d_idx]
        r02a = d_r02[d_idx]

        # r10a = d_r01[d_idx]
        r11a = d_r11[d_idx]
        r12a = d_r12[d_idx]

        # r20a = d_r02[d_idx]
        # r21a = d_r12[d_idx]
        r22a = d_r22[d_idx]

        r00b = s_r00[s_idx]
        r01b = s_r01[s_idx]
        r02b = s_r02[s_idx]

        # r10b = s_r01[s_idx]
        r11b = s_r11[s_idx]
        r12b = s_r12[s_idx]

        # r20b = s_r02[s_idx]
        # r21b = s_r12[s_idx]
        r22b = s_r22[s_idx]

        # compute the kernel correction term
        # if wdeltap is less than zero then no correction
        # needed
        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

            art_stress00 = fab * (r00a + r00b)
            art_stress01 = fab * (r01a + r01b)
            art_stress02 = fab * (r02a + r02b)

            art_stress10 = art_stress01
            art_stress11 = fab * (r11a + r11b)
            art_stress12 = fab * (r12a + r12b)

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = fab * (r22a + r22b)
        else:
            art_stress00 = 0.0
            art_stress01 = 0.0
            art_stress02 = 0.0

            art_stress10 = art_stress01
            art_stress11 = 0.0
            art_stress12 = 0.0

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = 0.0

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += mb * (art_stress00 * DWIJ[0] + art_stress01 * DWIJ[1] +
                             art_stress02 * DWIJ[2])

        d_av[d_idx] += mb * (art_stress10 * DWIJ[0] + art_stress11 * DWIJ[1] +
                             art_stress12 * DWIJ[2])

        d_aw[d_idx] += mb * (art_stress20 * DWIJ[0] + art_stress21 * DWIJ[1] +
                             art_stress22 * DWIJ[2])


class ComputeAuHatETVF(Equation):
    def __init__(self, dest, sources, pb):
        self.pb = pb
        super(ComputeAuHatETVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_h_b, d_auhat, d_avhat, d_awhat,
             WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        if d_h_b[d_idx] > 0.:
            tmp = -self.pb * s_m[s_idx] * rhoa21
            SPH_KERNEL.gradient(XIJ, RIJ, d_h_b[d_idx], dwijhat)

            d_auhat[d_idx] += tmp * dwijhat[0]
            d_avhat[d_idx] += tmp * dwijhat[1]
            d_awhat[d_idx] += tmp * dwijhat[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        if d_is_boundary[d_idx] == 1:
            # since it is boundary make its shifting acceleration zero
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.


class ComputeAuHatETVFTangentialCorrection(Equation):
    """
    This equation computes the background pressure force for ETVF.

    FIXME: This is not compatible for variable smoothing lengths
    """
    def __init__(self, dest, sources, pb):
        self.pb = pb
        super(ComputeAuHatETVFTangentialCorrection,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -self.pb * s_m[s_idx] * rhoa21
        SPH_KERNEL.gradient(XIJ, RIJ, d_h[d_idx], dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        if d_h_b[d_idx] < d_h[d_idx]:
            if d_is_boundary[d_idx] == 1:
                # since it is boundary make its shifting acceleration zero
                d_auhat[d_idx] = 0.
                d_avhat[d_idx] = 0.
                d_awhat[d_idx] = 0.
            else:
                # implies this is a particle adjacent to boundary particle so
                # nullify the normal component
                au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
                                 d_avhat[d_idx] * d_normal[idx3 + 1] +
                                 d_awhat[d_idx] * d_normal[idx3 + 2])

                # remove the normal acceleration component
                d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class ComputeKappaSun2019PST(Equation):
    def __init__(self, dest, sources, limit_angle):
        self.limit_angle = limit_angle
        super(ComputeKappaSun2019PST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa):
        # back ground pressure h (This will be the usual h value)
        d_kappa[d_idx] = 1.

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, d_normal, s_normal, s_x, s_y,
                 s_z, s_normal_norm, d_h_b, d_kappa, NBRS, N_NBRS):
        i, didx3, sidx3 = declare('int', 3)
        s_idx = declare('int')

        didx3 = 3 * d_idx

        # if d_idx == 239:
        #     print(d_normal[didx3])
        #     print(d_normal[didx3 + 1])
        #     print(d_normal[didx3 + 2])

        max_angle = 0.

        if d_h_b[d_idx] < d_h[d_idx]:
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                dx = d_x[d_idx] - s_x[s_idx]
                dy = d_y[d_idx] - s_y[s_idx]
                dz = d_z[d_idx] - s_z[s_idx]

                rij = (dx * dx + dy * dy + dz * dz)**0.5
                if rij > 1e-12:
                    if s_normal_norm[d_idx] == 1e-8:
                        sidx3 = 3 * s_idx

                        angle = acos(d_normal[didx3] * s_normal[sidx3] +
                                     d_normal[didx3 + 1] *
                                     s_normal[sidx3 + 1] +
                                     d_normal[didx3 + 2] *
                                     s_normal[sidx3 + 2]) * 180. / M_PI

                        if angle > max_angle:
                            max_angle = angle

                        # cos(15) (15 degrees)
                        if max_angle > self.limit_angle:
                            d_kappa[d_idx] = 0.
                            break
                        # find the angle between the normals
        else:
            d_kappa[d_idx] = 1.


class ComputeAuHatETVFSun2019(Equation):
    def __init__(self, dest, sources, mach_no, u_max, rho0, dim=2):
        self.mach_no = mach_no
        self.u_max = u_max
        self.dim = dim
        self.rho0 = rho0
        super(ComputeAuHatETVFSun2019, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_c0_ref, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ,
             RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx] / dt

        tmp1 = s_m[s_idx] / self.rho0

        d_auhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[0]
        d_avhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[1]
        d_awhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary, d_rho_splash):
        """Save the auhat avhat awhat
        First we make all the particles with div_r < dim - 0.5 as zero.

        Now if the particle is a free surface particle and not a free particle,
        which identified through our normal code (d_h_b < d_h), we cut off the
        normal component

        """
        idx3 = declare('int')
        idx3 = 3 * d_idx

        auhat = d_auhat[d_idx]
        avhat = d_avhat[d_idx]
        awhat = d_awhat[d_idx]

        if d_rho_splash[d_idx] < 0.5 * self.rho0:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
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
                    au_dot_normal = (auhat * d_normal[idx3] +
                                     avhat * d_normal[idx3 + 1] +
                                     awhat * d_normal[idx3 + 2])

                    # if it is going away from the continuum then nullify the
                    # normal component.
                    if au_dot_normal > 0.:
                        d_auhat[d_idx] = auhat - au_dot_normal * d_normal[idx3]
                        d_avhat[d_idx] = avhat - au_dot_normal * d_normal[idx3 + 1]
                        d_awhat[d_idx] = awhat - au_dot_normal * d_normal[idx3 + 2]


class ComputeAuHatETVFSun2019Solid(Equation):
    def __init__(self, dest, sources, mach_no, u_max):
        self.mach_no = mach_no
        self.u_max = u_max
        super(ComputeAuHatETVFSun2019Solid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_c0_ref, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx] / dt
        # tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx]

        tmp1 = s_m[s_idx] / s_rho[s_idx]

        d_auhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[0]
        d_avhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[1]
        d_awhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary, d_rho_splash, d_rho_ref, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        if d_rho_splash[d_idx] < 0.5 * d_rho_ref[0]:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
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
                    # d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                    # d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                    # d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class SummationDensitySplash(Equation):
    def initialize(self, d_idx, d_rho_splash):
        d_rho_splash[d_idx] = 0.0

    def loop(self, d_idx, d_rho_splash, s_idx, s_m, WIJ):
        d_rho_splash[d_idx] += s_m[s_idx]*WIJ


# #######################
# IPST equations start #
# #######################
def setup_ipst(pa, kernel):
    from boundary_particles import (add_boundary_identification_properties,
                                    get_boundary_identification_etvf_equations)

    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz ipst_distance_to_bp ipst_wij'.split()

    for prop in props:
        pa.add_property(prop)

    pa.add_constant('ipst_chi0', 0.)
    pa.add_property('ipst_chi')

    equations = [
        Group(
            equations=[
                CheckUniformityIPST(dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=kernel(dim=2))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = min(pa.ipst_chi)

    # ===================================================
    # save the initial distance to the boundary particles
    # ===================================================
    # ========================
    # compute the normal first
    # ========================
    boundary_equations_1 = get_boundary_identification_etvf_equations(
        destinations=[pa.name], sources=[pa.name])
    sph_eval = SPHEvaluator(arrays=[pa], equations=boundary_equations_1, dim=2,
                            kernel=kernel(dim=2))

    sph_eval.evaluate(dt=0.1)

    # ========================
    # compute the distance now
    # ========================
    pa.add_property('ipst_distance_to_bp0')

    equations = [
        Group(
            equations=[
                SaveTheInitialDistanceToBoundaryParticles(
                    dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=kernel(dim=2))

    sph_eval.evaluate(dt=0.1)


class SaveTheInitialDistanceToBoundaryParticles(Equation):
    def initialize(self, d_idx, d_ipst_dx, d_ipst_dy, d_ipst_dz,
                   d_ipst_distance_to_bp0, d_ipst_wij):
        d_ipst_distance_to_bp0[d_idx] = 0.
        d_ipst_wij[d_idx] = 0.

    def loop(self, d_idx, s_idx, s_is_boundary,
             s_rho, s_m, d_h, d_ipst_dx, d_ipst_dy,
             d_ipst_dz, d_normal, d_ipst_distance_to_bp0,
             d_ipst_wij,
             WIJ, XIJ, RIJ, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        Vj = s_m[s_idx] / s_rho[s_idx]  # volume of j

        if s_is_boundary[s_idx] == 1:
            d_ipst_wij[d_idx] += Vj * WIJ

            # before adding the correction of position, cut off the normal component
            dist = (-XIJ[0] * d_normal[idx3] +
                    -XIJ[1] * d_normal[idx3 + 1] +
                    -XIJ[2] * d_normal[idx3 + 2])
            d_ipst_distance_to_bp0[d_idx] += dist * Vj * WIJ

    def post_loop(self, d_idx, d_ipst_distance_to_bp0, d_ipst_wij):
        # adjustments to particle near free surface
        if d_ipst_wij[d_idx] > 1e-12:
            d_ipst_distance_to_bp0[d_idx] /= d_ipst_wij[d_idx]

class SavePositionsIPSTBeforeMoving(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_ipst_x[d_idx] = d_x[d_idx]
        d_ipst_y[d_idx] = d_y[d_idx]
        d_ipst_z[d_idx] = d_z[d_idx]


class AdjustPositionIPST(Equation):
    def __init__(self, dest, sources, u_max):
        self.u_max = u_max
        super(AdjustPositionIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_dx, d_ipst_dy, d_ipst_dz,
                   d_ipst_distance_to_bp, d_ipst_wij):
        d_ipst_dx[d_idx] = 0.0
        d_ipst_dy[d_idx] = 0.0
        d_ipst_dz[d_idx] = 0.0

        d_ipst_distance_to_bp[d_idx] = 0.
        d_ipst_wij[d_idx] = 0.

    def loop(self, d_idx, s_idx, s_is_boundary,
             s_rho, s_m, d_h, d_ipst_dx, d_ipst_dy,
             d_ipst_dz, d_normal, d_ipst_distance_to_bp,
             d_ipst_wij,
             WIJ, XIJ, RIJ, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        tmp = self.u_max * dt
        Vj = s_m[s_idx] / s_rho[s_idx]  # volume of j

        nij_x = 0.
        nij_y = 0.
        nij_z = 0.
        if RIJ > 1e-12:
            nij_x = XIJ[0] / RIJ
            nij_y = XIJ[1] / RIJ
            nij_z = XIJ[2] / RIJ

        d_ipst_dx[d_idx] += tmp * Vj * nij_x * WIJ
        d_ipst_dy[d_idx] += tmp * Vj * nij_y * WIJ
        d_ipst_dz[d_idx] += tmp * Vj * nij_z * WIJ

        if s_is_boundary[s_idx] == 1:
            d_ipst_wij[d_idx] += Vj * WIJ

            # before adding the correction of position, cut off the normal component
            dist = (-XIJ[0] * d_normal[idx3] +
                    -XIJ[1] * d_normal[idx3 + 1] +
                    -XIJ[2] * d_normal[idx3 + 2])
            d_ipst_distance_to_bp[d_idx] += dist * Vj * WIJ

    def post_loop(self, d_idx, d_x, d_y, d_z, d_ipst_dx, d_ipst_dy, d_ipst_dz,
                  d_normal, d_spacing0,
                  d_is_boundary,
                  d_ipst_distance_to_bp,
                  d_ipst_distance_to_bp0,
                  d_ipst_wij):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # before adding the correction of position, cut off the normal component
        dr_dot_normal = (d_ipst_dx[d_idx] * d_normal[idx3] +
                         d_ipst_dy[d_idx] * d_normal[idx3 + 1] +
                         d_ipst_dz[d_idx] * d_normal[idx3 + 2])

        # if it is going away from the continuum then nullify the
        # normal component.
        if dr_dot_normal > 0.:
            # remove the normal acceleration component
            d_ipst_dx[d_idx] -= dr_dot_normal * d_normal[idx3]
            d_ipst_dy[d_idx] -= dr_dot_normal * d_normal[idx3 + 1]
            d_ipst_dz[d_idx] -= dr_dot_normal * d_normal[idx3 + 2]

        # # adjustments to particle near free surface
        # if d_ipst_wij[d_idx] > 1e-12:
        #     d_ipst_distance_to_bp[d_idx] /= d_ipst_wij[d_idx]

        # diff = d_ipst_distance_to_bp[d_idx] - d_ipst_distance_to_bp0[d_idx]

        # if d_is_boundary[d_idx] != 1:
        #     if d_ipst_distance_to_bp0[d_idx] < 1.5 * d_spacing0[0]:
        #         if diff < 0.:
        #             d_ipst_dx[d_idx] += diff * d_normal[idx3]
        #             d_ipst_dy[d_idx] += diff * d_normal[idx3 + 1]
        #             d_ipst_dz[d_idx] += diff * d_normal[idx3 + 2]

        #         if diff > d_spacing0[0] / 8. and diff < 2. * d_spacing0[0]:
        #             d_ipst_dx[d_idx] += diff * d_normal[idx3]
        #             d_ipst_dy[d_idx] += diff * d_normal[idx3 + 1]
        #             d_ipst_dz[d_idx] += diff * d_normal[idx3 + 2]
        #             # ipst_distance_to_bp - ipst_distance_to_bp0

        d_x[d_idx] = d_x[d_idx] + d_ipst_dx[d_idx]
        d_y[d_idx] = d_y[d_idx] + d_ipst_dy[d_idx]
        d_z[d_idx] = d_z[d_idx] + d_ipst_dz[d_idx]


class CheckUniformityIPST(Equation):
    """
    For this specific equation one has to update the NNPS

    """
    def __init__(self, dest, sources, tolerance=0.2, debug=False):
        self.inhomogenity = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(CheckUniformityIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_chi):
        d_ipst_chi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_ipst_chi, d_h, WIJ, XIJ, RIJ,
             dt):
        d_ipst_chi[d_idx] += d_h[d_idx] * d_h[d_idx] * WIJ

    def reduce(self, dst, t, dt):
        chi_max = serial_reduce_array(dst.ipst_chi, 'min')
        self.inhomogenity = fabs(chi_max - dst.ipst_chi0[0])

    def converged(self):
        debug = self.debug
        inhomogenity = self.inhomogenity

        if inhomogenity > self.tolerance:
            if debug:
                print("Not converged:", inhomogenity)
            return -1.0
        else:
            if debug:
                print("Converged:", inhomogenity)
            return 1.0


class ComputeAuhatETVFIPSTFluids(Equation):
    def __init__(self, dest, sources, rho0, dim=2):
        self.rho0 = rho0
        super(ComputeAuhatETVFIPSTFluids, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z,
                   d_auhat, d_avhat, d_awhat, d_rho_splash, dt):
        if d_rho_splash[d_idx] < 0.5 * self.rho0:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
            dt_square_inv = 2. / (dt * dt)
            d_auhat[d_idx] = (d_x[d_idx] - d_ipst_x[d_idx]) * dt_square_inv
            d_avhat[d_idx] = (d_y[d_idx] - d_ipst_y[d_idx]) * dt_square_inv
            d_awhat[d_idx] = (d_z[d_idx] - d_ipst_z[d_idx]) * dt_square_inv


class ComputeAuhatETVFIPSTSolids(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z,
                   d_auhat, d_avhat, d_awhat, d_rho_splash, d_rho_ref, dt):
        if d_rho_splash[d_idx] < 0.5 * d_rho_ref[0]:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
            dt_square_inv = 2. / (dt * dt)
            d_auhat[d_idx] = (d_x[d_idx] - d_ipst_x[d_idx]) * dt_square_inv
            d_avhat[d_idx] = (d_y[d_idx] - d_ipst_y[d_idx]) * dt_square_inv
            d_awhat[d_idx] = (d_z[d_idx] - d_ipst_z[d_idx]) * dt_square_inv
    # def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
    #               d_awhat, d_is_boundary):
    #     idx3 = declare('int')
    #     idx3 = 3 * d_idx

    #     # first put a clearance
    #     magn_auhat = sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
    #                       d_avhat[d_idx] * d_avhat[d_idx] +
    #                       d_awhat[d_idx] * d_awhat[d_idx])

    #     if magn_auhat > 1e-12:
    #         # tmp = min(magn_auhat, self.u_max * 0.5)
    #         tmp = magn_auhat
    #         d_auhat[d_idx] = tmp * d_auhat[d_idx] / magn_auhat
    #         d_avhat[d_idx] = tmp * d_avhat[d_idx] / magn_auhat
    #         d_awhat[d_idx] = tmp * d_awhat[d_idx] / magn_auhat

    #         # Now apply the filter for boundary particles and adjacent particles
    #         if d_h_b[d_idx] < d_h[d_idx]:
    #             if d_is_boundary[d_idx] == 1:
    #                 # since it is boundary make its shifting acceleration zero
    #                 d_auhat[d_idx] = 0.
    #                 d_avhat[d_idx] = 0.
    #                 d_awhat[d_idx] = 0.
    #             else:
    #                 # implies this is a particle adjacent to boundary particle

    #                 # check if the particle is going away from the continuum
    #                 # or into the continuum
    #                 au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
    #                                  d_avhat[d_idx] * d_normal[idx3 + 1] +
    #                                  d_awhat[d_idx] * d_normal[idx3 + 2])

    #                 # if it is going away from the continuum then nullify the
    #                 # normal component.
    #                 if au_dot_normal > 0.:
    #                     # remove the normal acceleration component
    #                     d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
    #                     d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
    #                     d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class ResetParticlePositionsIPST(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_x[d_idx] = d_ipst_x[d_idx]
        d_y[d_idx] = d_ipst_y[d_idx]
        d_z[d_idx] = d_ipst_z[d_idx]


######################
# IPST equations end #
######################


class GTVFEOS(Equation):
    def initialize(self, d_idx, d_rho, d_p, d_c0_ref, d_rho_ref):
        d_p[d_idx] = d_c0_ref[0] * d_c0_ref[0] * (d_rho[d_idx] - d_rho_ref[0])

    def post_loop(self, d_idx, d_rho, d_p0, d_p, d_p_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p_ref[0])


class GTVFSetP0(Equation):
    def initialize(self, d_idx, d_rho, d_p0, d_p, d_p_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p_ref[0])


class ComputeAuHatGTVF(Equation):
    def __init__(self, dest, sources):
        super(ComputeAuHatGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p0, s_rho, s_m, d_auhat, d_avhat,
             d_awhat, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, HIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -d_p0[d_idx] * s_m[s_idx] * rhoa21

        SPH_KERNEL.gradient(XIJ, RIJ, 0.5 * HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class AdamiBoundaryConditionExtrapolateNoSlip(Equation):
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
        d_s01[d_idx] += s_s01[s_idx] * WIJ
        d_s02[d_idx] += s_s02[s_idx] * WIJ
        d_s11[d_idx] += s_s11[s_idx] * WIJ
        d_s12[d_idx] += s_s12[s_idx] * WIJ
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


class ComputeDivVelocity(Equation):
    def initialize(self, d_idx, d_div_vel):
        d_div_vel[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_div_vel, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_u, s_v, s_w, s_uhat, s_vhat, s_what, DWIJ):

        tmp = s_m[s_idx] / s_rho[s_idx]

        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - d_u[d_idx] - (s_uhat[s_idx] - s_u[s_idx])
        vij[1] = d_vhat[d_idx] - d_v[d_idx] - (s_vhat[s_idx] - s_v[s_idx])
        vij[2] = d_what[d_idx] - d_w[d_idx] - (s_what[s_idx] - s_w[s_idx])

        d_div_vel[d_idx] += tmp * -(vij[0] * DWIJ[0] + vij[1] * DWIJ[1] +
                                    vij[2] * DWIJ[2])


class ComputeDivDeviatoricStressOuterVelocity(Equation):
    def initialize(self, d_idx, d_s00u_x, d_s00v_y, d_s00w_z, d_s01u_x,
                   d_s01v_y, d_s01w_z, d_s02u_x, d_s02v_y, d_s02w_z, d_s11u_x,
                   d_s11v_y, d_s11w_z, d_s12u_x, d_s12v_y, d_s12w_z, d_s22u_x,
                   d_s22v_y, d_s22w_z):
        d_s00u_x[d_idx] = 0.0
        d_s00v_y[d_idx] = 0.0
        d_s00w_z[d_idx] = 0.0

        d_s01u_x[d_idx] = 0.0
        d_s01v_y[d_idx] = 0.0
        d_s01w_z[d_idx] = 0.0

        d_s02u_x[d_idx] = 0.0
        d_s02v_y[d_idx] = 0.0
        d_s02w_z[d_idx] = 0.0

        d_s11u_x[d_idx] = 0.0
        d_s11v_y[d_idx] = 0.0
        d_s11w_z[d_idx] = 0.0

        d_s12u_x[d_idx] = 0.0
        d_s12v_y[d_idx] = 0.0
        d_s12w_z[d_idx] = 0.0

        d_s22u_x[d_idx] = 0.0
        d_s22v_y[d_idx] = 0.0
        d_s22w_z[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, d_uhat, d_vhat, d_what, d_s00,
             d_s01, d_s02, d_s11, d_s12, d_s22, d_s00u_x, d_s00v_y, d_s00w_z,
             d_s01u_x, d_s01v_y, d_s01w_z, d_s02u_x, d_s02v_y, d_s02w_z,
             d_s11u_x, d_s11v_y, d_s11w_z, d_s12u_x, d_s12v_y, d_s12w_z,
             d_s22u_x, d_s22v_y, d_s22w_z, s_m, s_rho, s_u, s_v, s_w, s_uhat,
             s_vhat, s_what, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22, DWIJ):

        tmp = s_m[s_idx] / s_rho[s_idx]

        ud = d_uhat[d_idx] - d_u[d_idx]
        vd = d_vhat[d_idx] - d_v[d_idx]
        wd = d_what[d_idx] - d_w[d_idx]

        us = s_uhat[s_idx] - s_u[s_idx]
        vs = s_vhat[s_idx] - s_v[s_idx]
        ws = s_what[s_idx] - s_w[s_idx]

        d_s00u_x[d_idx] += tmp * -(d_s00[d_idx] * ud -
                                   s_s00[s_idx] * us) * DWIJ[0]
        d_s00v_y[d_idx] += tmp * -(d_s00[d_idx] * vd -
                                   s_s00[s_idx] * vs) * DWIJ[1]
        d_s00w_z[d_idx] += tmp * -(d_s00[d_idx] * wd -
                                   s_s00[s_idx] * ws) * DWIJ[2]

        d_s01u_x[d_idx] += tmp * -(d_s01[d_idx] * ud -
                                   s_s01[s_idx] * us) * DWIJ[0]
        d_s01v_y[d_idx] += tmp * -(d_s01[d_idx] * vd -
                                   s_s01[s_idx] * vs) * DWIJ[1]
        d_s01w_z[d_idx] += tmp * -(d_s01[d_idx] * wd -
                                   s_s01[s_idx] * ws) * DWIJ[2]

        d_s02u_x[d_idx] += tmp * -(d_s02[d_idx] * ud -
                                   s_s02[s_idx] * us) * DWIJ[0]
        d_s02v_y[d_idx] += tmp * -(d_s02[d_idx] * vd -
                                   s_s02[s_idx] * vs) * DWIJ[1]
        d_s02w_z[d_idx] += tmp * -(d_s02[d_idx] * wd -
                                   s_s02[s_idx] * ws) * DWIJ[2]

        d_s11u_x[d_idx] += tmp * -(d_s11[d_idx] * ud -
                                   s_s11[s_idx] * us) * DWIJ[0]
        d_s11v_y[d_idx] += tmp * -(d_s11[d_idx] * vd -
                                   s_s11[s_idx] * vs) * DWIJ[1]
        d_s11w_z[d_idx] += tmp * -(d_s11[d_idx] * wd -
                                   s_s11[s_idx] * ws) * DWIJ[2]

        d_s12u_x[d_idx] += tmp * -(d_s12[d_idx] * ud -
                                   s_s12[s_idx] * us) * DWIJ[0]
        d_s12v_y[d_idx] += tmp * -(d_s12[d_idx] * vd -
                                   s_s12[s_idx] * vs) * DWIJ[1]
        d_s12w_z[d_idx] += tmp * -(d_s12[d_idx] * wd -
                                   s_s12[s_idx] * ws) * DWIJ[2]

        d_s22u_x[d_idx] += tmp * -(d_s22[d_idx] * ud -
                                   s_s22[s_idx] * us) * DWIJ[0]
        d_s22v_y[d_idx] += tmp * -(d_s22[d_idx] * vd -
                                   s_s22[s_idx] * vs) * DWIJ[1]
        d_s22w_z[d_idx] += tmp * -(d_s22[d_idx] * wd -
                                   s_s22[s_idx] * ws) * DWIJ[2]


class HookesDeviatoricStressRateETVFCorrection(Equation):
    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12, d_as22,
                   d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_s00u_x,
                   d_s00v_y, d_s00w_z, d_s01u_x, d_s01v_y, d_s01w_z, d_s02u_x,
                   d_s02v_y, d_s02w_z, d_s11u_x, d_s11v_y, d_s11w_z, d_s12u_x,
                   d_s12v_y, d_s12w_z, d_s22u_x, d_s22v_y, d_s22w_z,
                   d_div_vel):
        d_as00[d_idx] += (d_s00u_x[d_idx] + d_s00v_y[d_idx] + d_s00w_z[d_idx] +
                          d_s00[d_idx] * d_div_vel[d_idx])
        d_as01[d_idx] += (d_s01u_x[d_idx] + d_s01v_y[d_idx] + d_s01w_z[d_idx] +
                          d_s01[d_idx] * d_div_vel[d_idx])
        d_as02[d_idx] += (d_s02u_x[d_idx] + d_s02v_y[d_idx] + d_s02w_z[d_idx] +
                          d_s02[d_idx] * d_div_vel[d_idx])

        d_as11[d_idx] += (d_s11u_x[d_idx] + d_s11v_y[d_idx] + d_s11w_z[d_idx] +
                          d_s11[d_idx] * d_div_vel[d_idx])
        d_as12[d_idx] += (d_s12u_x[d_idx] + d_s12v_y[d_idx] + d_s12w_z[d_idx] +
                          d_s12[d_idx] * d_div_vel[d_idx])

        d_as22[d_idx] += (d_s22u_x[d_idx] + d_s22v_y[d_idx] + d_s22w_z[d_idx] +
                          d_s22[d_idx] * d_div_vel[d_idx])


class EDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_uhat, s_vhat,
             s_what, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_what[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        rhoj = s_rho[s_idx]
        pj = s_p[s_idx]

        vij_dot_dwij = -(VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] +
                         VIJ[2] * DWIJ[2])

        vhatij_dot_dwij = -(vhatij[0] * DWIJ[0] + vhatij[1] * DWIJ[1] +
                            vhatij[2] * DWIJ[2])

        # vhatij_dot_dwij = (VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] +
        #                    VIJ[2]*DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += (pi - rhoi * cs2) * Vj * vij_dot_dwij

        #######################################################
        # second term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += -pi * Vj * vhatij_dot_dwij

        ########################################################
        # third term on the rhs of Eq 19 of the current paper #
        ########################################################
        tmp0 = pj * (s_uhat[s_idx] - s_u[s_idx]) - pi * (d_uhat[d_idx] -
                                                         d_u[d_idx])

        tmp1 = pj * (s_vhat[s_idx] - s_v[s_idx]) - pi * (d_vhat[d_idx] -
                                                         d_v[d_idx])

        tmp2 = pj * (s_what[s_idx] - s_w[s_idx]) - pi * (d_what[d_idx] -
                                                         d_w[d_idx])

        tmpdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)
        d_ap[d_idx] += Vj * tmpdotdwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class MakeSurfaceParticlesPressureApZero(Equation):
    def initialize(self, d_idx, d_is_boundary, d_p, d_ap):
        # if the particle is boundary set it's h_b to be zero
        if d_is_boundary[d_idx] == 1:
            d_ap[d_idx] = 0.
            d_p[d_idx] = 0.


class MakeSurfaceParticlesPressureApZeroEDACUpdated(Equation):
    def initialize(self, d_idx, d_edac_is_boundary, d_p, d_ap):
        # if the particle is boundary set it's h_b to be zero
        if d_edac_is_boundary[d_idx] == 1:
            d_ap[d_idx] = 0.
            d_p[d_idx] = 0.


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
               d_eps00, d_eps01, d_eps02, d_eps11, d_eps12, d_eps22, d_aeps00,
               d_aeps01, d_aeps02, d_aeps11, d_aeps12, d_aeps22, d_as01,
               d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
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

        # update strain  components
        d_eps00[d_idx] = d_eps00[d_idx] + dt * d_aeps00[d_idx]
        d_eps01[d_idx] = d_eps01[d_idx] + dt * d_aeps01[d_idx]
        d_eps02[d_idx] = d_eps02[d_idx] + dt * d_aeps02[d_idx]
        d_eps11[d_idx] = d_eps11[d_idx] + dt * d_aeps11[d_idx]
        d_eps12[d_idx] = d_eps12[d_idx] + dt * d_aeps12[d_idx]
        d_eps22[d_idx] = d_eps22[d_idx] + dt * d_aeps22[d_idx]

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


class SolidMechStepEDAC(SolidMechStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00, d_as01,
               d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01, d_sigma02,
               d_sigma11, d_sigma12, d_sigma22, d_eps00, d_eps01, d_eps02,
               d_eps11, d_eps12, d_eps22, d_aeps00, d_aeps01, d_aeps02,
               d_aeps11, d_aeps12, d_aeps22, d_p, d_ap, dt):
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

        # update strain  components
        d_eps00[d_idx] = d_eps00[d_idx] + dt * d_aeps00[d_idx]
        d_eps01[d_idx] = d_eps01[d_idx] + dt * d_aeps01[d_idx]
        d_eps02[d_idx] = d_eps02[d_idx] + dt * d_aeps02[d_idx]
        d_eps11[d_idx] = d_eps11[d_idx] + dt * d_aeps11[d_idx]
        d_eps12[d_idx] = d_eps12[d_idx] + dt * d_aeps12[d_idx]
        d_eps22[d_idx] = d_eps22[d_idx] + dt * d_aeps22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]

        d_rho[d_idx] += dt * d_arho[d_idx]

        d_p[d_idx] += dt * d_ap[d_idx]


class SolidMechStepErosion(SolidMechStep):
    """This step follows GTVF paper by Zhang 2017

    Also the stepper algorithm provided by "Frissane 2019 - 3D smoothed particle
    hydrodynamics modeling for high"

    """
    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00, d_as01,
               d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01, d_sigma02,
               d_sigma11, d_sigma12, d_sigma22, d_eps00, d_eps01, d_eps02,
               d_eps11, d_eps12, d_eps22, d_aeps00, d_aeps01, d_aeps02,
               d_aeps11, d_aeps12, d_aeps22, d_p, d_e, d_ae, d_T, d_G,
               d_specific_heat, d_JC_T_room, d_eff_plastic_strain,
               d_plastic_strain_rate, d_JC_A, d_JC_B, d_JC_n, d_JC_C,
               d_JC_psr_0, d_JC_T_melt, d_JC_m, d_yield_stress, d_damage_1,
               d_damage_2, d_damage_3, d_damage_4, d_damage_5,
               d_epsilon_failure, d_damage_D, d_is_damaged, d_f_y, d_sij_star,
               dt):
        # Update the positions
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # Update the density
        d_rho[d_idx] += dt * d_arho[d_idx]

        # Update the pressure
        # d_p[d_idx] += dt * d_ap[d_idx]

        # Update the energy
        d_e[d_idx] += dt * d_ae[d_idx]
        d_T[d_idx] = d_e[d_idx] / d_specific_heat[0] + d_JC_T_room[0]

        # update strain  components
        d_eps00[d_idx] = d_eps00[d_idx] + dt * d_aeps00[d_idx]
        d_eps01[d_idx] = d_eps01[d_idx] + dt * d_aeps01[d_idx]
        d_eps02[d_idx] = d_eps02[d_idx] + dt * d_aeps02[d_idx]
        d_eps11[d_idx] = d_eps11[d_idx] + dt * d_aeps11[d_idx]
        d_eps12[d_idx] = d_eps12[d_idx] + dt * d_aeps12[d_idx]
        d_eps22[d_idx] = d_eps22[d_idx] + dt * d_aeps22[d_idx]

        # update deviatoric stress components
        s00_star = d_s00[d_idx] + dt * d_as00[d_idx]
        s01_star = d_s01[d_idx] + dt * d_as01[d_idx]
        s02_star = d_s02[d_idx] + dt * d_as02[d_idx]
        s11_star = d_s11[d_idx] + dt * d_as11[d_idx]
        s12_star = d_s12[d_idx] + dt * d_as12[d_idx]
        s22_star = d_s22[d_idx] + dt * d_as22[d_idx]

        # ========================================
        # find the Johnson Cook yield stress value
        # ========================================
        # # compute yield stress (sigma_y)
        # # 2. A smoothed particle hydrodynamics (SPH) model for simulating
        # # surface erosion by impacts of foreign particles, by X.W. Dong
        eps = d_eff_plastic_strain[d_idx]
        psr = d_plastic_strain_rate[d_idx]
        term_1 = (d_JC_A[0] + d_JC_B[0] * eps**d_JC_n[0])
        term_2 = 1.
        tmp_term_2 = psr / d_JC_psr_0[0]

        if tmp_term_2 > 1e-12:
            term_2 = (1 + d_JC_C[0] * log(tmp_term_2))

        # equation 25 of [2]
        t_room = d_JC_T_room[0]
        # normalized temperature T^*
        # equation 24 of [2]
        t_star = (d_T[d_idx] - t_room) / (d_JC_T_melt[0] - t_room)
        if t_star > 0.:
            term_3 = (1. - t_star**d_JC_m[0])
        else:
            term_3 = 1.
        d_yield_stress[d_idx] = term_1 * term_2 * term_3

        sij_sij = (
            s00_star * s00_star + s01_star * s01_star +
            s02_star * s02_star + s01_star * s01_star +
            s11_star * s11_star + s12_star * s12_star +
            s02_star * s02_star + s12_star * s12_star +
            s22_star * s22_star)
        s_star = sqrt(3./2. * sij_sij)
        d_sij_star[d_idx] = s_star

        f_y = 1.
        if s_star > 1e-12:
            f_y = min(d_yield_stress[d_idx] / s_star, 1.)

        d_f_y[d_idx] = f_y

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

        # Strain failure (Equation 27 Dong2016)
        D1 = d_damage_1[0]
        D2 = d_damage_2[0]
        D3 = d_damage_3[0]
        D4 = d_damage_4[0]
        D5 = d_damage_5[0]

        sigma_star = 0.
        if s_star > 1e-12:
            sigma_star = -d_p[d_idx] / s_star
        term_1 = (D1 + D2 * exp(D3 * sigma_star))
        term_2 = 1.
        tmp_term_2 = psr / d_JC_psr_0[0]
        if tmp_term_2 > 1e-12:
            term_2 = (1 + D4 * log(tmp_term_2))

        # equation 25 of [2]
        t_room = d_JC_T_room[0]
        # normalized temperature T^*
        # equation 24 of [2]
        t_star = (d_T[d_idx] - t_room) / (d_JC_T_melt[0] - t_room)
        term_3 = (1. + D5 * t_star)
        d_epsilon_failure[d_idx] = term_1 * term_2 * term_3

        if d_is_damaged[d_idx] == 0.:
            # Compute the damage D (eq 26 Dong2016)
            d_damage_D[d_idx] = abs(d_eff_plastic_strain[d_idx] /
                                    d_epsilon_failure[d_idx])
            if d_damage_D[d_idx] > 1.:
                d_is_damaged[d_idx] = 1.
        else:
            d_s00[d_idx] = 0.
            d_s01[d_idx] = 0.
            d_s02[d_idx] = 0.
            d_s11[d_idx] = 0.
            d_s12[d_idx] = 0.
            d_s22[d_idx] = 0.

        # update sigma
        # update deviatoric stress components
        # d_s00[d_idx] = d_s00[d_idx] + dt * d_as00[d_idx]
        # d_s01[d_idx] = d_s01[d_idx] + dt * d_as01[d_idx]
        # d_s02[d_idx] = d_s02[d_idx] + dt * d_as02[d_idx]
        # d_s11[d_idx] = d_s11[d_idx] + dt * d_as11[d_idx]
        # d_s12[d_idx] = d_s12[d_idx] + dt * d_as12[d_idx]
        # d_s22[d_idx] = d_s22[d_idx] + dt * d_as22[d_idx]

        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


class EDACIntegrator(Integrator):
    def initial_acceleration(self, t, dt):
        pass

    def one_timestep(self, t, dt):
        self.compute_accelerations(0, update_nnps=False)
        self.stage1()
        self.do_post_stage(dt, 1)
        self.update_domain()

        self.compute_accelerations(1)

        self.stage2()
        self.do_post_stage(dt, 2)


class SolidMechETVFEDACIntegStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw,
               d_uhat, d_vhat, d_what, d_auhat, d_avhat, d_awhat, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_rho, d_vol, d_ap,
               d_avol, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, dt):
        d_vol[d_idx] += dt * d_avol[d_idx]

        # set the density from the volume
        d_rho[d_idx] = d_m[d_idx] / d_vol[d_idx]

        d_p[d_idx] += dt * d_ap[d_idx]

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


class StiffEOS(Equation):
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        super(StiffEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_p, d_c0_ref, d_rho_ref):
        tmp = d_rho[d_idx] / d_rho_ref[0]
        tmp1 = d_rho_ref[0] * d_c0_ref[0] * d_c0_ref[0] / self.gamma
        d_p[d_idx] = tmp1 * (pow(tmp, self.gamma) - 1.)


class MakeAuhatZero(Equation):
    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.
        d_avhat[d_idx] = 0.
        d_awhat[d_idx] = 0.


class HookesDeviatoricStressRateWithStrainTrace(Equation):
    r""" **Rate of change of stress **

    .. math::
        \frac{dS^{ij}}{dt} = 2\mu\left(\epsilon^{ij} - \frac{1}{3}\delta^{ij}
        \epsilon^{ij}\right) + S^{ik}\Omega^{jk} + \Omega^{ik}S^{kj}

    where

    .. math::

        \epsilon^{ij} = \frac{1}{2}\left(\frac{\partial v^i}{\partial x^j} +
        \frac{\partial v^j}{\partial x^i}\right)\\

        \Omega^{ij} = \frac{1}{2}\left(\frac{\partial v^i}{\partial x^j} -
           \frac{\partial v^j}{\partial x^i} \right)

    """

    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12,
                   d_as22, d_aeps00, d_aeps01, d_aeps02, d_aeps11, d_aeps12,
                   d_aeps22):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as02[d_idx] = 0.0

        d_as11[d_idx] = 0.0
        d_as12[d_idx] = 0.0

        d_as22[d_idx] = 0.0

        d_aeps00[d_idx] = 0.0
        d_aeps01[d_idx] = 0.0
        d_aeps02[d_idx] = 0.0

        d_aeps11[d_idx] = 0.0
        d_aeps12[d_idx] = 0.0

        d_aeps22[d_idx] = 0.0

    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_v00,
             d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_v22, d_as00,
             d_as01, d_as02, d_as11, d_as12, d_as22, d_aeps00, d_aeps01,
             d_aeps02, d_aeps11, d_aeps12, d_aeps22, d_plastic_strain_rate,
             d_G):

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

        d_aeps00[d_idx] = eps00
        d_aeps01[d_idx] = eps01
        d_aeps02[d_idx] = eps02
        d_aeps11[d_idx] = eps11
        d_aeps12[d_idx] = eps12
        d_aeps22[d_idx] = eps22

        # compute the plastic strain rate
        Dij_Dij = (
            eps00 * eps00 + eps01 * eps01 +
            eps02 * eps02 + eps01 * eps01 +
            eps11 * eps11 + eps12 * eps12 +
            eps02 * eps02 + eps12 * eps12 +
            eps22 * eps22)
        d_plastic_strain_rate[d_idx] = sqrt(Dij_Dij)

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

        tmp = 2.0 * d_G[0]
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


class SolidsScheme(Scheme):
    """

    There are three schemes

    1. GRAY
    2. GTVF
    3. ETVF

    ETVF scheme in particular has 2 PST techniques

    1. SUN2019
    2. IPST


    Using the following commands one can use these schemes

    1. GRAY

    python file_name.py --no-volume-correction --no-uhat --no-shear-tvf-correction --pst gray -d filename_pst_gray_output

    2. GTVF

    python file_name.py --no-volume-correction --uhat --no-shear-tvf-correction --pst gtvf -d filename_pst_gtvf_output

    3. ETVF

    python file_name.py --volume-correction --no-uhat --shear-tvf-correction --pst $(sun2019 or ipst) -d filename_etvf_pst_$(sun2019_or_ipst)_output


    ipst has additional arguments such as `ipst_max_iterations`, this can be
    changed using command line arguments


    # Note

    Additionally one can go for EDAC option

    python file_name.py --edac --surface-p-zero $(rest_of_the_arguments)

    """
    def __init__(self, solids, boundaries, dim, h, pb, u_max, mach_no,
                 hdx, rigid_bodies=[], rigid_boundaries=[],
                 ipst_max_iterations=10,
                 ipst_min_iterations=0,
                 ipst_tolerance=0.2, ipst_interval=1,
                 use_uhat_velgrad=False,
                 use_uhat_cont=False, artificial_vis_alpha=1.0,
                 artificial_vis_beta=0.0, artificial_stress_eps=0.3,
                 continuity_tvf_correction=False,
                 shear_stress_tvf_correction=False,
                 stiff_eos=False, gamma=7., pst="sun2019",
                 kr=1e8, kf=1e5, fric_coeff=0.0, gx=0., gy=0., gz=0.):
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        self.rigid_boundaries = rigid_boundaries
        self.rigid_bodies = rigid_bodies

        self.dim = dim

        # TODO: if the kernel is adaptive this will fail
        self.h = h
        self.hdx = hdx

        # for Monaghan stress
        self.artificial_stress_eps = artificial_stress_eps

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel = QuinticSpline
        self.kernel_factor = 3

        self.use_uhat_cont = use_uhat_cont
        self.use_uhat_velgrad = use_uhat_velgrad

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.continuity_tvf_correction = continuity_tvf_correction
        self.shear_stress_tvf_correction = shear_stress_tvf_correction

        self.surf_p_zero = True

        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.u_max = u_max
        self.mach_no = mach_no

        # attributes for IPST technique
        self.ipst_max_iterations = ipst_max_iterations
        self.ipst_min_iterations = ipst_min_iterations
        self.ipst_tolerance = ipst_tolerance
        self.ipst_interval = ipst_interval

        self.debug = False

        self.stiff_eos = stiff_eos
        self.mie_gruneisen_eos = False
        self.gamma = gamma

        # boundary conditions
        self.adami_velocity_extrapolate = False
        self.no_slip = False
        self.free_slip = False

        self.mohseni_contact_force = False
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=1.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=1.0, type=float,
                           help="Artificial viscosity coefficients, beta")
        add_bool_argument(group, 'adami-velocity-extrapolate',
                          dest='adami_velocity_extrapolate', default=False,
                          help='Use adami velocity extrapolation')

        add_bool_argument(group, 'mie-gruneisen-eos', dest='mie_gruneisen_eos',
                          default=True, help='Use Mie Gruneisen equation of state')

        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e8,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.0,
                           type=float,
                           help="Friction coefficient")

        add_bool_argument(group, 'mohseni-contact-force', dest='mohseni_contact_force',
                          default=False, help='Use Mohseni contact force')

    def consume_user_options(self, options):
        _vars = [
            'artificial_vis_alpha',
            'mie_gruneisen_eos',
            'kr', 'kf', 'fric_coeff', 'mohseni_contact_force'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def check_ipst_time(self, t, dt):
        if int(t / dt) % self.ipst_interval == 0:
            return True
        else:
            return False

    def get_equations(self):
        # from fluids import (SetWallVelocityFreeSlip, ContinuitySolidEquation,
        #                     ContinuitySolidEquationGTVF,
        #                     ContinuitySolidEquationETVFCorrection)

        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               VelocityGradient3D,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress)

        from pysph.sph.solid_mech.basic import (
            EnergyEquationWithStress)

        from contact_force import (
            ComputeContactForceNormals,
            ComputeContactForceDistanceAndClosestPoint,
            ComputeContactForce,
            ComputeContactForceOnRigidBody,
            ContactForceDEMOnElasticBody,
            ContactForceDEMOnRigidBody)

        from rigid_body_common import (BodyForce, SumUpExternalForces)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        # tmp = []
        # if self.adami_velocity_extrapolate is True:
        #     if self.no_slip is True:
        #         if len(self.boundaries) > 0:
        #             for boundary in self.boundaries:
        #                 tmp.append(
        #                     SetWallVelocity(dest=boundary,
        #                                     sources=self.solids))

        #     if self.free_slip is True:
        #         if len(self.boundary) > 0:
        #             for boundary in self.boundaries:
        #                 tmp.append(
        #                     SetWallVelocityFreeSlip(dest=boundary,
        #                                             sources=self.solids))
        #     stage1.append(Group(equations=tmp))
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ContinuityEquation(dest=solid, sources=[solid]+self.boundaries))

            g1.append(VelocityGradient2D(dest=solid, sources=[solid]+self.boundaries))

            # g1.append(
            #     MonaghanArtificialStress(dest=solid, sources=None,
            #                                 eps=self.artificial_stress_eps))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRateWithStrainTrace(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # ------------------------
        # stage 2 equations starts
        # ------------------------

        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        if self.pst in ["sun2019", "ipst"]:
            for solid in self.solids:
                g1.append(
                    SetHIJForInsideParticles(dest=solid, sources=[solid],
                                             h=self.h,
                                             kernel_factor=self.kernel_factor))

                g1.append(SummationDensitySplash(dest=solid,
                                                 sources=self.solids+self.boundaries))
            stage2.append(Group(g1))

        for solid in self.solids:
            if self.mie_gruneisen_eos is True:
                g2.append(MieGruneisenEOS(solid, sources=None))
            else:
                g2.append(IsothermalEOS(solid, sources=None))

        stage2.append(Group(g2))

        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(
                    dest=boundary, sources=self.solids))
        if len(g3) > 0:
            stage2.append(Group(g3))

        # -------------------
        # solve momentum equation for solid
        # -------------------
        g4 = []
        for solid in self.solids:
            # add only if there is some positive value
            if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                g4.append(
                    MonaghanArtificialViscosity(
                        dest=solid, sources=[solid]+self.boundaries,
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationSolids(dest=solid,
                                             sources=[solid]+self.boundaries))

            g4.append(
                EnergyEquationWithStress(
                    dest=solid, sources=[solid]+self.boundaries,
                    alpha=self.artificial_vis_alpha,
                    beta=self.artificial_vis_beta))

            # g4.append(
            #     MonaghanArtificialStressCorrection(dest=solid,
            #                                        sources=[solid]))

            tmp_list = self.solids.copy()
            tmp_list.remove(solid)

        stage2.append(Group(g4))

        # New contact force equations
        if self.mohseni_contact_force == True:
            g9 = []
            for solid in self.solids:
                tmp_list = self.solids.copy() + self.rigid_boundaries + self.rigid_bodies
                tmp_list.remove(solid)
                g9.append(
                    ComputeContactForceNormals(dest=solid,
                                               sources=tmp_list))

            stage2.append(Group(equations=g9, real=False))

            g10 = []
            for solid in self.solids:
                tmp_list = self.solids.copy() + self.rigid_boundaries + self.rigid_bodies
                tmp_list.remove(solid)
                g10.append(
                    ComputeContactForceDistanceAndClosestPoint(
                        dest=solid, sources=tmp_list))
            stage2.append(Group(equations=g10, real=False))

            g11 = []
            for solid in self.solids:
                g11.append(
                    ComputeContactForce(dest=solid,
                                        sources=None,
                                        kr=self.kr,
                                        kf=self.kf,
                                        fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g11, real=False))

        else:
            g9 = []
            for solid in self.solids:
                tmp_list = self.solids.copy() + self.rigid_boundaries + self.rigid_bodies
                tmp_list.remove(solid)
                g9.append(
                    ContactForceDEMOnElasticBody(dest=solid,
                                                 sources=tmp_list,
                                                 kr=self.kr,
                                                 kf=self.kf,
                                                 fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g9, real=False))

        g11 = []
        for solid in self.solids:
            g11.append(
                AddGravityToStructure(dest=solid,
                                      sources=None,
                                      gx=self.gx,
                                      gy=self.gy,
                                      gz=self.gz))

        stage2.append(Group(equations=g11, real=False))

        # ===============================
        # Handle the rigid body equations
        # ===============================
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

            if self.mohseni_contact_force == True:
                g9 = []
                for body in self.rigid_bodies:
                    g9.append(
                        ComputeContactForceNormals(dest=body,
                                                sources=self.solids))

                stage2.append(Group(equations=g9, real=False))

                g10 = []
                for body in self.rigid_bodies:
                    g10.append(
                        ComputeContactForceDistanceAndClosestPoint(
                            dest=body, sources=self.solids))

                stage2.append(Group(equations=g10, real=False))

                g5 = []
                for body in self.rigid_bodies:
                    g5.append(
                        ComputeContactForceOnRigidBody(
                            dest=body,
                            sources=None,
                            kr=self.kr,
                            kf=self.kf,
                            fric_coeff=self.fric_coeff))

                stage2.append(Group(equations=g5, real=False))
            else:
                g9 = []
                for body in self.rigid_bodies:
                    g9.append(
                        ContactForceDEMOnRigidBody(dest=body,
                                                   sources=self.solids,
                                                   kr=self.kr,
                                                   kf=self.kf,
                                                   fric_coeff=self.fric_coeff))

                stage2.append(Group(equations=g9, real=False))

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name, sources=None))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """TODO: Fix the integrator of the boundary. If it is solve_tau then solve for
        deviatoric stress or else no integrator has to be used
        """
        from rigid_body_3d import GTVFRigidBody3DStep

        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        # step = SolidMechStep
        step = SolidMechStepErosion
        step_cls = step

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        bodystep = GTVFRigidBody3DStep()
        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties
        from rigid_body_common import (set_total_mass, set_center_of_mass,
                                       set_body_frame_position_vectors,
                                       set_body_frame_normal_vectors,
                                       set_moment_of_inertia_and_its_inverse,
                                       BodyForce, SumUpExternalForces,
                                       normalize_R_orientation)

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw')

            # For strain energy computation
            add_properties(pa, 'eps00', 'eps01', 'eps02', 'eps11', 'eps12',
                           'eps22')
            add_properties(pa, 'aeps00', 'aeps01', 'aeps02', 'aeps11',
                           'aeps12', 'aeps22')

            # for plastic limiter
            add_properties(pa, 'f_y', 'sij_star')

            pa.add_output_arrays(['eps00', 'eps01', 'eps02', 'eps11', 'eps12',
                                  'eps22'])

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            G = get_shear_modulus(pa.E[0], pa.nu[0])
            pa.add_property('G')
            pa.G[:] = G

            # set the speed of sound
            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            c0_ref = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.add_constant('c0_ref', c0_ref)

            # Energy equation properties
            add_properties(pa, 'ae', 'e', 'T')
            # this is dummy value of AL, will change for different materials
            if 'specific_heat' not in pa.constants:
                pa.add_constant('specific_heat', 875.)

            if 'JC_T_room' not in pa.constants:
                pa.add_constant('JC_T_room', 273.)

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add
            add_properties(pa, 'auhat', 'avhat', 'awhat', 'uhat', 'vhat',
                           'what', 'div_r')

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # output arrays
            pa.add_output_arrays(['sigma00', 'sigma01', 'sigma11'])

            # now add properties specific to the scheme and PST
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            if self.pst == "gtvf":
                add_properties(pa, 'p0')

                if 'p_ref' not in pa.constants:
                    pa.add_constant('p_ref', 0.)

                if 'b_mod' not in pa.constants:
                    pa.add_constant('b_mod', 0.)

                pa.b_mod[0] = get_bulk_mod(pa.G[0], pa.nu[0])
                pa.p_ref[0] = pa.b_mod[0]

            if self.pst == "sun2019" or "ipst":
                # for boundary identification and for sun2019 pst
                pa.add_property('normal', stride=3)
                pa.add_property('normal_tmp', stride=3)
                pa.add_property('normal_norm')

                # check for boundary particle
                pa.add_property('is_boundary', type='int')

                # used to set the particles near the boundary
                pa.add_property('h_b')

            # if the PST is IPST
            if self.pst == "ipst":
                setup_ipst(pa, self.kernel)

            # for edac
            add_properties(pa, 'ap')

            if self.surf_p_zero == True:
                pa.add_property('edac_normal', stride=3)
                pa.add_property('edac_normal_tmp', stride=3)
                pa.add_property('edac_normal_norm')

                # check for edac boundary particle
                pa.add_property('edac_is_boundary', type='int')

                pa.add_property('ap')

            # add the corrected shear stress rate
            if self.shear_stress_tvf_correction == True:
                add_properties(pa, 'div_vel')
                add_properties(pa, 's11v_y', 's01v_y', 's11w_z', 's00w_z',
                               's12w_z', 's12v_y', 's02u_x', 's22w_z',
                               's11u_x', 's22u_x', 's00u_x', 's02w_z',
                               's02v_y', 's00v_y', 's01w_z', 's22v_y',
                               's12u_x', 's01u_x')

            pa.add_output_arrays(['p'])

            # for splash particles
            add_properties(pa, 'rho_splash')
            pa.add_output_arrays(['rho_splash'])

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties_stride(pa, pa.total_no_bodies[0],
                                  'contact_force_normal_x',
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

            # check for boundary particle
            # pa.add_property('contact_force_is_boundary', type='int')
            pa.add_property('contact_force_is_boundary')

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')
            pa.is_boundary[:] = 0

            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            if self.continuity_tvf_correction == True:
                pa.add_property('ughat')
                pa.add_property('vghat')
                pa.add_property('wghat')

            # now add properties specific to the scheme and PST
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            if self.pst == "gtvf":
                add_properties(pa, 'uhat', 'vhat', 'what')

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties_stride(pa, pa.total_no_bodies[0],
                                  'contact_force_normal_x',
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
            nb = int(np.max(pa.body_id) + 1)

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
            from boundary_particles import (
                get_boundary_identification_etvf_equations,
                add_boundary_identification_properties)

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

    def get_solver(self):
        return self.solver


# ---------------------------
# Plasticity code
# ---------------------------
def setup_elastic_plastic_johnsoncook_model(pa):
    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz'.split()

    for prop in props:
        pa.add_property(prop)

    pa.add_constant('ipst_chi0', 0.)
    pa.add_property('ipst_chi')

    equations = [
        Group(
            equations=[
                CheckUniformityIPST(dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=QuinticSpline(dim=2))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = min(pa.ipst_chi)


def get_poisson_ratio_from_E_G(e, g):
    return e / (2. * g) - 1.


def add_plasticity_properties(pa):
    pa.add_property('plastic_limit')
    pa.add_property('J2')


class ComputeJ2(Equation):
    def initialize(self, d_idx, d_p, s_p, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22, d_J2, d_plastic_limit):
        # J2 = tr(A^T, B)
        # compute the second invariant of deviatoric stress
        d_J2[d_idx] = (
            d_s00[d_idx] * d_s00[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s11[d_idx] * d_s11[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s22[d_idx] * d_s22[d_idx]) / 2.


class ComputeJohnsonCookYieldStress(Equation):
    """

    Params:

    a, b, c, n, m: material constants
    T_m: melting temperature
    T_tr: transition temperature

    """
    def __init__(self, dest, sources, a, b, c, n, m, T_m, T_tr,
                 plastic_strain_rate_0):
        self.a = a
        self.b = b
        self.c = c
        self.n = n
        self.m = m
        self.T_m = T_m
        self.T_tr = T_tr
        self.plastic_strain_rate_0 = plastic_strain_rate_0
        super(ComputeJohnsonCookYieldStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, s_p, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22, d_J2, d_plastic_strain, d_plastic_strain_rate,
                   d_sigma_yield):
        tmp1 = (self.a + self.b * pow(d_epsilon_p[d_idx], n))
        tmp2 = 1. + self.c * ln(
            d_plastic_strain_rate[d_idx] / self.plastic_strain_rate_0)

        tmp3_1 = d_T[d_idx] - self.T_tr
        tmp3_2 = self.T_m - self.T_tr
        tmp3 = (1. - pow(tmp3_1 / tmp3_2), self.m)

        d_sigma_yield[d_idx] = tmp1 * tmp2 * tmp3


class MieGruneisenEOS(Equation):
    """
    Eq 15 in Dong 2016

    eq 4 in Dong 2017, Modeling, simulation and analysis of single
    angular-type particles on ductile surfaces

    """
    def __init__(self, dest, sources):
        super(MieGruneisenEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_e, d_p,
                   d_mie_gruneisen_gamma,
                   d_mie_gruneisen_S,
                   d_c0_ref, d_rho_ref):
        # eq 4 in Dong 2017, Modeling, simulation and analysis of single
        # angular-type particles on ductile surfaces
        eta = (d_rho[d_idx] / d_rho_ref[0]) - 1

        tmp = d_mie_gruneisen_gamma[0] * d_rho_ref[0] * d_e[d_idx]
        tmp1 = 1. + (1. - d_mie_gruneisen_gamma[0] * 0.5) * eta
        numer = d_rho_ref[0] * d_c0_ref[0]**2. * eta * tmp1
        denom = (1. - (d_mie_gruneisen_S[0] - 1.) * eta)

        d_p[d_idx] = numer / denom + tmp


class LimitDeviatoricStress(Equation):
    def __init__(self, dest, sources, yield_modulus):
        self.yield_modulus = yield_modulus
        super(LimitDeviatoricStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_J2,
                   d_plastic_limit):
        # J2 = tr(A^T, B)
        # compute the second invariant of deviatoric stress
        d_J2[d_idx] = (
            d_s00[d_idx] * d_s00[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s11[d_idx] * d_s11[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s22[d_idx] * d_s22[d_idx])
        # this is just to check if it is greater than zero. It can be 1e-12
        # because we are dividing it by J2
        if d_J2[d_idx] > 0.1:
            # d_plastic_limit[d_idx] = min(
            #     self.yield_modulus * self.yield_modulus / (3. * d_J2[d_idx]),
            #     1.)
            d_plastic_limit[d_idx] = min(
                self.yield_modulus / sqrt(3. * d_J2[d_idx]), 1.)

        d_s00[d_idx] = d_plastic_limit[d_idx] * d_s00[d_idx]
        d_s01[d_idx] = d_plastic_limit[d_idx] * d_s01[d_idx]
        d_s02[d_idx] = d_plastic_limit[d_idx] * d_s02[d_idx]
        d_s11[d_idx] = d_plastic_limit[d_idx] * d_s11[d_idx]
        d_s12[d_idx] = d_plastic_limit[d_idx] * d_s12[d_idx]
        d_s22[d_idx] = d_plastic_limit[d_idx] * d_s22[d_idx]
