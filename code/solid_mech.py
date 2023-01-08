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


class ComputePrincipalStress2D(Equation):
    def initialize(self, d_idx, d_sigma_1, d_sigma_2, d_sigma00, d_sigma01,
                   d_sigma02, d_sigma11, d_sigma12, d_sigma22):
        # https://www.ecourses.ou.edu/cgi-bin/eBook.cgi?doc=&topic=me&chap_sec=07.2&page=theory
        tmp1 = (d_sigma00[d_idx] + d_sigma11[d_idx]) / 2

        tmp2 = (d_sigma00[d_idx] - d_sigma11[d_idx]) / 2

        tmp3 = sqrt(tmp2**2. + d_sigma01[d_idx]**2.)

        d_sigma_1[d_idx] = tmp1 + tmp3
        d_sigma_2[d_idx] = tmp1 - tmp3


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
               d_is_static,
               dt):
        if d_is_static[d_idx] == 0.:
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

            # update deviatoric stress components
            s00_star = d_s00[d_idx] + dt * d_as00[d_idx]
            s01_star = d_s01[d_idx] + dt * d_as01[d_idx]
            s02_star = d_s02[d_idx] + dt * d_as02[d_idx]
            s11_star = d_s11[d_idx] + dt * d_as11[d_idx]
            s12_star = d_s12[d_idx] + dt * d_as12[d_idx]
            s22_star = d_s22[d_idx] + dt * d_as22[d_idx]
            # scale the trial stress back to the yield surface
            sij_sij = (
                s00_star * s00_star + s01_star * s01_star +
                s02_star * s02_star + s01_star * s01_star +
                s11_star * s11_star + s12_star * s12_star +
                s02_star * s02_star + s12_star * s12_star +
                s22_star * s22_star)
            sij_star = sqrt(3./2. * sij_sij)
            d_sij_star[d_idx] = sij_star

            f_y = 1.
            if sij_star > 1e-12:
                f_y = min(d_yield_stress[d_idx] / sij_star, 1.)

            d_f_y[d_idx] = f_y

            # Deviatoric stress is updated
            d_s00[d_idx] = f_y * s00_star
            d_s01[d_idx] = f_y * s01_star
            d_s02[d_idx] = f_y * s02_star
            d_s11[d_idx] = f_y * s11_star
            d_s12[d_idx] = f_y * s12_star
            d_s22[d_idx] = f_y * s22_star

            # # update strain  components
            # d_eps00[d_idx] = d_eps00[d_idx] + dt * d_aeps00[d_idx]
            # d_eps01[d_idx] = d_eps01[d_idx] + dt * d_aeps01[d_idx]
            # d_eps02[d_idx] = d_eps02[d_idx] + dt * d_aeps02[d_idx]
            # d_eps11[d_idx] = d_eps11[d_idx] + dt * d_aeps11[d_idx]
            # d_eps12[d_idx] = d_eps12[d_idx] + dt * d_aeps12[d_idx]
            # d_eps22[d_idx] = d_eps22[d_idx] + dt * d_aeps22[d_idx]

            # ========================================
            # find the Johnson Cook yield stress value
            # ========================================
            # compute yield stress (sigma_y)
            # 2. A smoothed particle hydrodynamics (SPH) model for simulating
            # surface erosion by impacts of foreign particles, by X.W. Dong
            # Effective plastic strain rate is given differently in different
            # papers
            delta_eps = 1. / 3. * (1. - f_y) / d_G[d_idx] * sij_star
            d_eff_plastic_strain[d_idx] += delta_eps
            eps = sqrt(2. / 3.) * d_eff_plastic_strain[d_idx]
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

            # Strain failure (Equation 27 Dong2016)
            D1 = d_damage_1[0]
            D2 = d_damage_2[0]
            D3 = d_damage_3[0]
            D4 = d_damage_4[0]
            D5 = d_damage_5[0]

            sigma_star = 0.
            if sij_star > 1e-12:
                sigma_star = -d_p[d_idx] / sij_star
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
                # d_damage_D[d_idx] = abs(d_eff_plastic_strain[d_idx] /
                #                         d_epsilon_failure[d_idx])
                # d_damage_D[d_idx] = (d_eff_plastic_strain[d_idx] /
                #                      d_epsilon_failure[d_idx])
                d_damage_D[d_idx] += delta_eps / d_epsilon_failure[d_idx]
                if d_damage_D[d_idx] > 1.:
                    d_is_damaged[d_idx] = 1.
            else:
                d_s00[d_idx] = 0.
                d_s01[d_idx] = 0.
                d_s02[d_idx] = 0.
                d_s11[d_idx] = 0.
                d_s12[d_idx] = 0.
                d_s22[d_idx] = 0.
                # tmp = 1.

            d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
            d_sigma01[d_idx] = d_s01[d_idx]
            d_sigma02[d_idx] = d_s02[d_idx]
            d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
            d_sigma12[d_idx] = d_s12[d_idx]
            d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


class SolidMechStepSimpleYieldErosion(SolidMechStep):
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
               d_plastic_strain_rate, d_yield_stress, d_is_damaged, d_f_y,
               d_sij_star, d_is_static, dt):
        if d_is_static[d_idx] == 0.:
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

            sij_sij = (
                s00_star * s00_star + s01_star * s01_star +
                s02_star * s02_star + s01_star * s01_star +
                s11_star * s11_star + s12_star * s12_star +
                s02_star * s02_star + s12_star * s12_star +
                s22_star * s22_star)
            sij_star = sqrt(3./2. * sij_sij)
            d_sij_star[d_idx] = sij_star

            f_y = 1.
            if sij_star > 1e-12:
                f_y = min(d_yield_stress[0] / sij_star, 1.)

            d_f_y[d_idx] = f_y

            # Deviatoric stress is updated
            d_s00[d_idx] = f_y * s00_star
            d_s01[d_idx] = f_y * s01_star
            d_s02[d_idx] = f_y * s02_star
            d_s11[d_idx] = f_y * s11_star
            d_s12[d_idx] = f_y * s12_star
            d_s22[d_idx] = f_y * s22_star

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


class StaticBoundaryParticles(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat,
                   d_u, d_v, d_w, d_is_static):
        if d_is_static[d_idx] == 1.:
            d_au[d_idx] = 0.
            d_av[d_idx] = 0.
            d_aw[d_idx] = 0.

            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.

            d_u[d_idx] = 0.
            d_v[d_idx] = 0.
            d_w[d_idx] = 0.


class SetStaticBoundaryParticlesStress(Equation):
    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_p, d_wij, d_is_static):
        if d_is_static[d_idx] == 1.:
            d_s00[d_idx] = 0.
            d_s01[d_idx] = 0.
            d_s02[d_idx] = 0.
            d_s11[d_idx] = 0.
            d_s12[d_idx] = 0.
            d_s22[d_idx] = 0.
            d_p[d_idx] = 0.
            d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p, d_s00, d_s01,
             d_s02, d_s11, d_s12, d_s22, s_s00, s_s01, s_s02, s_s11, s_s12,
             s_s22, d_wij, WIJ, DWIJ, d_is_static, s_is_static):
        if d_is_static[d_idx] == 1.:
            if s_is_static[s_idx] == 0.:
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
                  d_p, d_is_static):
        if d_is_static[d_idx] == 1.:
            # extrapolated pressure at the ghost particle
            if d_wij[d_idx] > 1e-14:
                d_s00[d_idx] /= d_wij[d_idx]
                d_s01[d_idx] /= d_wij[d_idx]
                d_s02[d_idx] /= d_wij[d_idx]
                d_s11[d_idx] /= d_wij[d_idx]
                d_s12[d_idx] /= d_wij[d_idx]
                d_s22[d_idx] /= d_wij[d_idx]

                d_p[d_idx] /= d_wij[d_idx]


class SetDamagedParticlesStressToZero(Equation):
    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_p,
                   d_sigma00, d_sigma01, d_sigma02, d_sigma11, d_sigma12,
                   d_sigma22, d_wij, d_is_damaged):
        if d_is_damaged[d_idx] == 1.:
            d_s00[d_idx] = 0.
            d_s01[d_idx] = 0.
            d_s02[d_idx] = 0.
            d_s11[d_idx] = 0.
            d_s12[d_idx] = 0.
            d_s22[d_idx] = 0.
            d_p[d_idx] = 0.
            d_sigma00[d_idx] = 0.
            d_sigma01[d_idx] = 0.
            d_sigma02[d_idx] = 0.
            d_sigma11[d_idx] = 0.
            d_sigma12[d_idx] = 0.
            d_sigma22[d_idx] = 0.


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
                  d_awhat, d_is_boundary, d_rho_ref, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

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


class SolidsScheme(Scheme):
    def __init__(self, solids, dim, h, hdx, rigid_bodies=[], rigid_boundaries=[],
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 artificial_stress_eps=0.3, kr=1e8, kf=1e5,
                 fric_coeff=0.0, mach_no=0.1, u_max=1., gx=0., gy=0., gz=0.):
        self.solids = solids

        self.rigid_boundaries = rigid_boundaries
        self.rigid_bodies = rigid_bodies

        self.dim = dim

        # TODO: if the kernel is adaptive this will fail
        self.h = h
        self.hdx = hdx

        # for Monaghan stress
        self.artificial_stress_eps = artificial_stress_eps

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.use_ctvf = False
        self.kernel_factor = 3
        self.mach_no = mach_no
        self.u_max = u_max

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

        # add_bool_argument(group, 'mie-gruneisen-eos', dest='mie_gruneisen_eos',
        #                   default=True, help='Use Mie Gruneisen equation of state')

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

        add_bool_argument(group, 'use-ctvf', dest='use_ctvf',
                          default=False,
                          help='Use particle shifting')

    def consume_user_options(self, options):
        _vars = [
            'artificial_vis_alpha',
            # 'mie_gruneisen_eos',
            'kr', 'kf', 'fric_coeff', 'use_ctvf'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
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
            ComputeContactForceNormalsMohseniEB,
            ComputeContactForceDistanceAndClosestPointMohseniEB,
            ComputeContactForceMohseniEB,
            TransferContactForceMohseniEB)

        from rigid_body_common import (
            BodyForce, SumUpExternalForces,
            ComputeContactForceNormalsMohseniRB,
            ComputeContactForceDistanceAndClosestPointMohseniRB,
            ComputeContactForceMohseniRB,
            TransferContactForceMohseniRB)

        stage1 = []
        g1 = []
        all = list(set(self.solids))

        for solid in self.solids:
            g1.append(ContinuityEquation(dest=solid, sources=[solid]))

            g1.append(VelocityGradient3D(dest=solid, sources=[solid]))

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

        if self.use_ctvf:
            for solid in self.solids:
                g1.append(
                    SetHIJForInsideParticles(dest=solid, sources=[solid],
                                             h=self.h,
                                             kernel_factor=self.kernel_factor))

                # g1.append(SummationDensitySplash(dest=solid,
                #                                  sources=self.solids+self.boundaries))
            stage2.append(Group(g1))

        for solid in self.solids:
            # if self.mie_gruneisen_eos is True:
            g2.append(MieGruneisenEOS(solid, sources=None))
            # else:
            # g2.append(IsothermalEOS(solid, sources=None))

        stage2.append(Group(g2))

        # Make accelerations and velocity of the boundary particles to zero
        # computation of total force and torque at center of mass
        tmp = []
        for solid in self.solids:
            tmp.append(SetStaticBoundaryParticlesStress(dest=solid, sources=[solid]))
        stage2.append(Group(tmp))

        # Make stress of the damaged particles to zero
        tmp = []
        for solid in self.solids:
            tmp.append(SetDamagedParticlesStressToZero(dest=solid, sources=[solid]))
        stage2.append(Group(tmp))

        # -------------------
        # solve momentum equation for solid
        # -------------------
        g4 = []
        for solid in self.solids:
            # add only if there is some positive value
            if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                g4.append(
                    MonaghanArtificialViscosity(
                        dest=solid, sources=[solid],
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationSolids(dest=solid,
                                             sources=[solid]))

            if self.use_ctvf:
                g4.append(
                    ComputeAuHatETVFSun2019Solid(
                        dest=solid, sources=[solid],
                        mach_no=self.mach_no, u_max=self.u_max))

            g4.append(
                EnergyEquationWithStress(
                    dest=solid, sources=[solid],
                    alpha=self.artificial_vis_alpha,
                    beta=self.artificial_vis_beta))

            # g4.append(
            #     MonaghanArtificialStressCorrection(dest=solid,
            #                                        sources=[solid]))

            # tmp_list = self.solids.copy()
            # tmp_list.remove(solid)

        stage2.append(Group(g4))

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

            # ==========================================================
            # Compute the force on the elastic solid due to rigid bodies
            # and on rigid body due to elastic solid
            # ==========================================================
            g9 = []
            for solid in self.solids:
                g9.append(
                    ComputeContactForceNormalsMohseniEB(
                        dest=solid, sources=self.rigid_bodies))

            stage2.append(Group(equations=g9, real=False))

            g10 = []
            for solid in self.solids:
                g10.append(
                    ComputeContactForceDistanceAndClosestPointMohseniEB(
                        dest=solid, sources=self.rigid_bodies))

            stage2.append(Group(equations=g10, real=False))

            g5 = []
            for solid in self.solids:
                g5.append(
                    ComputeContactForceMohseniEB(
                        dest=solid,
                        sources=None,
                        kr=self.kr,
                        kf=self.kf,
                        fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for body in self.rigid_bodies:
                g5.append(
                    TransferContactForceMohseniEB(
                        dest=body, sources=self.solids))
            stage2.append(Group(equations=g5, real=False))
            # ==========================================================
            # Compute the force on the elastic solid due to rigid bodies
            # and on rigid body due to elastic solid
            # ENDS
            # ==========================================================

            # ==============================================================
            # Compute the force on the rigid body due to other rigid bodies
            # ==============================================================
            g9 = []
            for body in self.rigid_bodies:
                g9.append(
                    ComputeContactForceNormalsMohseniRB(
                        dest=body, sources=self.rigid_bodies))

            stage2.append(Group(equations=g9, real=False))

            g10 = []
            for body in self.rigid_bodies:
                g10.append(
                    ComputeContactForceDistanceAndClosestPointMohseniRB(
                        dest=body, sources=self.rigid_bodies))

            stage2.append(Group(equations=g10, real=False))

            g5 = []
            for body in self.rigid_bodies:
                g5.append(
                    ComputeContactForceMohseniRB(
                        dest=body,
                        sources=None,
                        kr=self.kr,
                        kf=self.kf,
                        fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for body in self.rigid_bodies:
                g5.append(
                    TransferContactForceMohseniRB(
                        dest=body, sources=self.rigid_bodies))
            stage2.append(Group(equations=g5, real=False))
            # ==========================================================
            # Compute the force on the elastic solid due to rigid bodies
            # and on rigid body due to elastic solid
            # ENDS
            # ==========================================================

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name, sources=None))

            stage2.append(Group(equations=g6, real=False))

        # Make accelerations and velocity of the boundary particles to zero
        # computation of total force and torque at center of mass
        g6 = []
        for solid in self.solids:
            g6.append(StaticBoundaryParticles(dest=solid, sources=None))

        stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from rigid_body_3d import GTVFRigidBody3DStep

        kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        step = SolidMechStepErosion
        # step = SolidMechStepSimpleYieldErosion
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
                           'as12', 'as22', 'arho', 'au', 'av', 'aw', 'wij')

            # For strain energy computation
            add_properties(pa, 'eps00', 'eps01', 'eps02', 'eps11', 'eps12',
                           'eps22')
            add_properties(pa, 'aeps00', 'aeps01', 'aeps02', 'aeps11',
                           'aeps12', 'aeps22')

            # for plastic limiter
            add_properties(pa, 'f_y', 'sij_star')

            # pa.add_output_arrays(['eps00', 'eps01', 'eps02', 'eps11', 'eps12',
            #                       'eps22'])

            # this will change
            kernel = QuinticSpline(dim=self.dim)
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
            pa.add_output_arrays(['sigma00', 'sigma01', 'sigma11', 'T'])

            # now add properties specific to the scheme and PST
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            # for boundary identification and for sun2019 pst
            pa.add_property('normal', stride=3)
            pa.add_property('normal_tmp', stride=3)
            pa.add_property('normal_norm')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')

            # used to set the particles near the boundary
            pa.add_property('h_b')

            # for edac
            add_properties(pa, 'ap')

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
                                  'closest_point_dist_to_source',
                                  'contact_force_weight_denominator',
                                  )

            pa.add_property(name='dem_id_source',
                            stride=pa.total_no_bodies[0],
                            type='int')
            pa.add_property(name='idx_source',
                            stride=pa.total_no_bodies[0],
                            type='int')
            add_properties(pa, 'au_contact', 'av_contact', 'aw_contact')


            # check for boundary particle
            # pa.add_property('contact_force_is_boundary', type='int')
            pa.add_property('contact_force_is_boundary')

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

            pa.add_property(name='dem_id_source', stride=pa.total_no_bodies[0],
                            type='int')
            pa.add_property(name='idx_source',
                            stride=pa.total_no_bodies[0],
                            type='int')

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


class MieGruneisenEOS(Equation):
    """
    Eq 15 in Dong 2016

    eq 4 in Dong 2017, Modeling, simulation and analysis of single
    angular-type particles on ductile surfaces

    """
    def __init__(self, dest, sources):
        super(MieGruneisenEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_e, d_p, d_mie_gruneisen_gamma,
                   d_mie_gruneisen_S, d_c0_ref, d_rho_ref):
        # eq 4 in Dong 2017, Modeling, simulation and analysis of single
        # angular-type particles on ductile surfaces
        eta = (d_rho[d_idx] / d_rho_ref[0]) - 1

        tmp = d_mie_gruneisen_gamma[0] * d_rho_ref[0] * d_e[d_idx]
        tmp1 = 1. + (1. - d_mie_gruneisen_gamma[0] * 0.5) * eta
        numer = d_rho_ref[0] * d_c0_ref[0]**2. * eta * tmp1
        denom = (1. - (d_mie_gruneisen_S[0] - 1.) * eta)

        d_p[d_idx] = numer / denom + tmp
