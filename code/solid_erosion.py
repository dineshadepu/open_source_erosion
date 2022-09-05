import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from textwrap import dedent
from compyle.api import declare

from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from numpy import fabs
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from boundary_particles import (ComputeNormalsEDAC, SmoothNormalsEDAC,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.wc.transport_velocity import SetWallVelocity

from pysph.examples.solid_mech.impact import add_properties

from pysph.sph.integrator import Integrator

from math import sqrt, acos, log, exp
from math import pi as M_PI

from rigid_body_common import (set_total_mass, set_center_of_mass,
                               set_body_frame_position_vectors,
                               set_body_frame_normal_vectors,
                               set_moment_of_inertia_and_its_inverse,
                               BodyForce, SumUpExternalForces,
                               normalize_R_orientation,
                               GTVFRigidBody3DStep)
# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)

from solid_mech import (ContinuityEquationUhat,
                        ContinuityEquationETVFCorrection,
                        EDACEquation, SetHIJForInsideParticles,
                        AdamiBoundaryConditionExtrapolateNoSlip,
                        MomentumEquationSolids, ComputeAuHatETVFSun2019)


def get_poisson_ratio_from_K_and_G(K, G):
    """
    Elastoplastic deformation during projectile-wall collision
    Author: Paul W.Cleary
    https://doi.org/10.1016/j.apm.2009.04.004
    """
    numerator = (3. * K / G) - 2.
    denominator = 2. * ((3. * K / G) + 1.)
    return numerator / denominator


def get_youngs_mod_from_G_and_nu(G, nu):
    return 2. * G * (1. + nu)


class BulkModEOS(Equation):
    r""" Compute the pressure using linear variation of density

    :math:`p = K (\rho / \rho_0 - 1)`

    """
    def loop(self, d_idx, d_rho, d_p, d_b_mod, d_rho_ref):
        d_p[d_idx] = d_b_mod[0] * (d_rho[d_idx] / d_rho_ref[0] - 1.)


class SolidMechErosionStep(IntegratorStep):
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
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, d_e, d_T, d_ae,
               d_specific_heat, dt):
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

        d_e[d_idx] += d_ae[d_idx] * dt

        d_T[d_idx] = d_specific_heat[d_idx]

        d_rho[d_idx] += dt * d_arho[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class SolidErosionScheme(Scheme):
    """1. This scheme will model the solid erosion of an isotropic material due
to an external rigid body impact.

    2. Both rigid body 2d and 3d schemes will be implmented. And compared their
    validity

    """
    def __init__(self, solids, boundaries, dim, h, kernel_factor, pb, edac_nu,
                 u_max, mach_no, hdx=1.3, ipst_max_iterations=10,
                 ipst_tolerance=0.2, ipst_interval=1, use_uhat=False,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 artificial_stress_eps=0.3, continuity_tvf_correction=False,
                 shear_stress_tvf_correction=False, kernel_choice="1",
                 stiff_eos=False, gamma=7., pst="sun2019"):
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        self.dim = dim

        # TODO: if the kernel is adaptive this will fail
        self.h = h
        self.hdx = hdx

        # for Monaghan stress
        self.artificial_stress_eps = artificial_stress_eps

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel_choice = "1"
        self.kernel = QuinticSpline
        self.kernel_factor = kernel_factor

        self.use_uhat = use_uhat

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.continuity_tvf_correction = continuity_tvf_correction
        self.shear_stress_tvf_correction = shear_stress_tvf_correction

        self.edac_nu = edac_nu
        self.surf_p_zero = True
        self.edac = False

        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.u_max = u_max
        self.mach_no = mach_no

        # attributes for IPST technique
        self.ipst_max_iterations = ipst_max_iterations
        self.ipst_tolerance = ipst_tolerance
        self.ipst_interval = ipst_interval

        self.debug = False

        self.stiff_eos = stiff_eos
        self.gamma = gamma

        self.solver = None

    def add_user_options(self, group):
        add_bool_argument(
            group, 'surf-p-zero', dest='surf_p_zero', default=True,
            help='Make the surface pressure and acceleration to be zero')

        add_bool_argument(group, 'uhat', dest='use_uhat', default=False,
                          help='Use Uhat as a process')

        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=1.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        add_bool_argument(
            group, 'continuity-tvf-correction',
            dest='continuity_tvf_correction', default=True,
            help='Add the extra continuty term arriving due to TVF')

        add_bool_argument(
            group, 'shear-stress-tvf-correction',
            dest='shear_stress_tvf_correction', default=True,
            help='Add the extra shear stress rate term arriving due to TVF')

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution equation EDAC')

        choices = ['sun2019', 'ipst', 'gray', 'gtvf', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        group.add_argument("--ipst-max-iterations", action="store",
                           dest="ipst_max_iterations", default=10, type=int,
                           help="Max iterations of IPST")

        group.add_argument("--ipst-interval", action="store",
                           dest="ipst_interval", default=1, type=int,
                           help="Frequency at which IPST is to be done")

        group.add_argument("--ipst-tolerance", action="store", type=float,
                           dest="ipst_tolerance", default=None,
                           help="Tolerance limit of IPST")

        add_bool_argument(group, 'debug', dest='debug', default=False,
                          help='Check if the IPST converged')

        choices = ["1", "2", "3", "4", "5", "6", "7", "8"]
        group.add_argument(
            "--kernel-choice", action="store", dest='kernel_choice',
            default="1", choices=choices,
            help="""Specify what kernel to use (one of %s).
                           1. QuinticSpline
                           2. WendlandQuintic
                           3. CubicSpline
                           4. WendlandQuinticC4
                           5. Gaussian
                           6. SuperGaussian
                           7. Gaussian
                           8. Gaussian""" % choices)

        add_bool_argument(group, 'stiff-eos', dest='stiff_eos', default=False,
                          help='use stiff equation of state')

    def consume_user_options(self, options):
        _vars = [
            'surf_p_zero',
            'use_uhat',
            'artificial_vis_alpha',
            'shear_stress_tvf_correction',
            'edac',
            'pst',
            'debug',
            'ipst_max_iterations',
            'ipst_tolerance',
            'ipst_interval',
            'kernel_choice',
            'stiff_eos',
            'continuity_tvf_correction',
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.kernel_choice == "1":
            self.kernel = QuinticSpline
            self.kernel_factor = 3
        elif self.kernel_choice == "2":
            self.kernel = WendlandQuintic
            self.kernel_factor = 2
        elif self.kernel_choice == "3":
            self.kernel = CubicSpline
            self.kernel_factor = 2
        elif self.kernel_choice == "4":
            self.kernel = WendlandQuinticC4
            self.kernel_factor = 2
            self.h = self.h / self.hdx * 2.0
        elif self.kernel_choice == "5":
            self.kernel = Gaussian
            self.kernel_factor = 3
        elif self.kernel_choice == "6":
            self.kernel = SuperGaussian
            self.kernel_factor = 3

    def check_ipst_time(self, t, dt):
        if int(t / dt) % self.ipst_interval == 0:
            return True
        else:
            return False

    def get_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               VelocityGradient3D,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        for solid in self.solids:
            g1.append(ContinuityEquationUhat(dest=solid, sources=all))
            g1.append(
                ContinuityEquationETVFCorrection(dest=solid, sources=all))

            g1.append(VelocityGradient2D(dest=solid, sources=all))
        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # edac pressure evolution equation
        if self.edac is True:
            gtmp = []
            for solid in self.solids:
                gtmp.append(
                    EDACEquation(dest=solid, sources=all, nu=self.edac_nu))

            stage1.append(Group(gtmp))

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
            stage2.append(Group(g1))

        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(dest=boundary,
                                                        sources=self.solids))
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
                        dest=solid, sources=all,
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationSolids(dest=solid, sources=all))

            g4.append(
                ComputeAuHatETVFSun2019(dest=solid,
                                        sources=[solid] + self.boundaries,
                                        mach_no=self.mach_no,
                                        u_max=self.u_max))

        stage2.append(Group(g4))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """TODO: Fix the integrator of the boundary. If it is solve_tau then solve for
        deviatoric stress or else no integrator has to be used
        """
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        step_cls = SolidMechErosionStep

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        # TODO: ADD RIGID BODY STEPPERS

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw', 'e', 'ae')

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            G = get_shear_modulus(pa.E[0], pa.nu[0])
            pa.add_constant('G', G)

            # set the speed of sound
            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            c0_ref = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.add_constant('c0_ref', c0_ref)

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add
            add_properties(pa, 'auhat', 'avhat', 'awhat', 'uhat', 'vhat',
                           'what')

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # for boundary identification and for sun2019 pst
            pa.add_property('normal', stride=3)
            pa.add_property('normal_tmp', stride=3)
            pa.add_property('normal_norm')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')

            # used to set the particles near the boundary
            pa.add_property('h_b')

            add_properties(pa, 'ap')

            pa.add_output_arrays(['p'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what')

            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

    def get_solver(self):
        return self.solver


class SolidErosionWCSPHScheme(Scheme):
    """1. This scheme will model the solid erosion of an isotropic material due
to an external rigid body impact.

    2. Both rigid body 2d and 3d schemes will be implmented. And compared their
    validity

    """
    def __init__(self, solids, boundaries, rigid_bodies, dim, h, kernel_factor,
                 pb, edac_nu, hdx=1.3, artificial_vis_alpha=1.0,
                 artificial_vis_beta=0.0, artificial_stress_eps=0.3):
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        self.rigid_bodies = rigid_bodies
        self.dim = dim

        # TODO: if the kernel is adaptive this will fail
        self.h = h
        self.hdx = hdx

        # for Monaghan stress
        self.artificial_stress_eps = artificial_stress_eps

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.edac_nu = edac_nu
        self.edac = False

        self.solver = None

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=1.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution equation EDAC')

    def consume_user_options(self, options):
        _vars = [
            'artificial_vis_alpha',
            'edac'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    # def attributes_changed(self):
    #     if self.kernel_choice == "1":
    #         self.kernel = QuinticSpline
    #         self.kernel_factor = 3
    #     elif self.kernel_choice == "2":
    #         self.kernel = WendlandQuintic
    #         self.kernel_factor = 2
    #     elif self.kernel_choice == "3":
    #         self.kernel = CubicSpline
    #         self.kernel_factor = 2
    #     elif self.kernel_choice == "4":
    #         self.kernel = WendlandQuinticC4
    #         self.kernel_factor = 2
    #         self.h = self.h / self.hdx * 2.0
    #     elif self.kernel_choice == "5":
    #         self.kernel = Gaussian
    #         self.kernel_factor = 3
    #     elif self.kernel_choice == "6":
    #         self.kernel = SuperGaussian
    #         self.kernel_factor = 3

    def get_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               VelocityGradient3D,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress,
                                                MonaghanArtificialStressCorrection)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress,
                                                MonaghanArtificialStressCorrection,
                                                EnergyEquationWithStress)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        for solid in self.solids:
            g1.append(ContinuityEquation(dest=solid, sources=all))
            g1.append(VelocityGradient2D(dest=solid, sources=all))
            g1.append(
                MonaghanArtificialStress(dest=solid, sources=None,
                                         eps=self.artificial_stress_eps))
        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # edac pressure evolution equation
        if self.edac is True:
            gtmp = []
            for solid in self.solids:
                gtmp.append(
                    EDACEquation(dest=solid, sources=all, nu=self.edac_nu))

            stage1.append(Group(gtmp))

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        for solid in self.solids:
            g2.append(IsothermalEOS(solid, sources=None))

        stage2.append(Group(g2))

        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(dest=boundary,
                                                        sources=self.solids))
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
                        dest=solid, sources=all,
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationSolids(dest=solid, sources=all))

            g4.append(EnergyEquationWithStress(dest=solid, sources=all))

            g4.append(
                MonaghanArtificialStressCorrection(dest=solid,
                                                   sources=all))

        stage2.append(Group(g4))

        #######################
        # Handle rigid bodies #
        #######################
        tmp = []
        # update the contacts first
        # for name in self.rigid_bodies:
        #     tmp.append(
        #         # see the previous examples and write down the sources
        #         UpdateTangentialContactsLVCForce(dest=name, sources=self.rigid_bodies+self.boundaries))
        # stage2.append(Group(equations=tmp, real=False))

        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                BodyForce(dest=name,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        for name in self.rigid_bodies:
            if self.dem == "canelas":
                g5.append(RigidBodyCanelasRigidRigid(dest=name, sources=self.rigid_bodies,
                                                     Cn=self.Cn))
            elif self.dem == "bui":
                g5.append(RigidBodyBuiRigidRigid(dest=name, sources=self.rigid_bodies,
                                                 en=self.en))

            if len(self.boundaries) > 0:
                if self.dem == "canelas":
                    g5.append(RigidBodyCanelasRigidWall(dest=name, sources=self.boundaries,
                                                        Cn=self.Cn))
                elif self.dem == "bui":
                    g5.append(RigidBodyBuiRigidRigid(dest=name, sources=self.boundaries,
                                                     en=self.en))

        stage2.append(Group(equations=g5, real=False))

        # computation of total force and torque at center of mass
        g6 = []
        for name in self.rigid_bodies:
            g6.append(SumUpExternalForces(dest=name, sources=None))

        stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        step_cls = SolidMechErosionStep
        bodystep = GTVFRigidBody3DStep()

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ratio as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw', 'e', 'ae')

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            G = get_shear_modulus(pa.E[0], pa.nu[0])
            pa.add_constant('G', G)

            # set the speed of sound
            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            c0_ref = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.add_constant('c0_ref', c0_ref)

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # now add properties specific to the scheme and PST
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            # for edac
            if self.edac is True:
                add_properties(pa, 'ap')

            pa.add_output_arrays(['p'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what')

            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            # now add properties specific to the scheme and PST
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

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
            for i in range(len(pa.x)):
                if pa.is_boundary[i] == 0:
                    pa.normal[3 * i] = 0.
                    pa.normal[3 * i + 1] = 0.
                    pa.normal[3 * i + 2] = 0.

            # normal vectors in terms of body frame
            set_body_frame_normal_vectors(pa)

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')
            pa.add_property('wij')

            ##########################################
            # Add dem contact force model properties #
            ##########################################
            # create the array to save the tangential interaction particles
            # index and other variables
            # HERE `tng` is tangential
            limit = pa.max_tng_contacts_limit[0]
            pa.add_property('tng_idx', stride=limit, type="int")
            pa.tng_idx[:] = -1
            pa.add_property('tng_idx_dem_id', stride=limit, type="int")
            pa.tng_idx_dem_id[:] = -1

            # if self.contact_model == "LVC":
            pa.add_property('tng_fx', stride=limit)
            pa.add_property('tng_fy', stride=limit)
            pa.add_property('tng_fz', stride=limit)

            pa.add_property('tng_x', stride=limit)
            pa.add_property('tng_y', stride=limit)
            pa.add_property('tng_z', stride=limit)
            # pa.add_property('tng_fx0', stride=limit)
            # pa.add_property('tng_fy0', stride=limit)
            # pa.add_property('tng_fz0', stride=limit)
            pa.tng_fx[:] = 0.
            pa.tng_fy[:] = 0.
            pa.tng_fz[:] = 0.
            # pa.tng_fx0[:] = 0.
            # pa.tng_fy0[:] = 0.
            # pa.tng_fz0[:] = 0.

            pa.add_property('total_tng_contacts', type="int")
            pa.total_tng_contacts[:] = 0

            if self.dem == "bui":
                pa.add_property('bui_total_contacts', type="int")
                pa.bui_total_contacts[:] = 0
                pa.add_output_arrays(['bui_total_contacts'])

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'normal',
                'is_boundary', 'fz', 'm', 'body_id'
            ])

    def get_solver(self):
        return self.solver


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
    Johnson Cook Model

    Equation 3.20 of [1] and equation 23 of [2]

    1. The study of performances of kernel types in solid dynamic problems by SPH
    By Shuangshuang, (We use this reference)

    2. A smoothed particle hydrodynamics (SPH) model for simulating
    surface erosion by impacts of foreign particles, by X.W. Dong

    """
    def initialize(self, d_idx, d_p, d_s22, d_J2, d_v00, d_v01, d_v02, d_v10,
                   d_v11, d_v12, d_v20, d_v21, d_v22,
                   d_equivalent_plastic_strain,
                   d_equivalent_plastic_strain_rate, d_yield_stress,
                   d_jc_t_melting, d_jc_t_room, d_jc_a, d_jc_b, d_jc_c, d_jc_m,
                   d_jc_n, d_jc_eps_dot0, d_specific_heat, d_T, d_e, dt):
        # compute current temperature from energy and specific heat
        # equation 25 of [2]
        cp = d_specific_heat[0]
        t_room = d_jc_t_room[0]
        d_T[d_idx] = d_e[d_idx] / cp + t_room

        # normalized temperature T^*
        # equation 24 of [2]
        t_star = (d_T[d_idx] - t_room) / (d_jc_t_melting[0] - t_room)

        # ======================================
        # compute equivalent plastic strain rate
        # ======================================
        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v02 = d_v02[d_idx]

        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]
        v12 = d_v12[d_idx]

        v20 = d_v20[d_idx]
        v21 = d_v21[d_idx]
        v22 = d_v22[d_idx]

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

        tmp = (eps00 * eps00 + eps01 * eps01 + eps02 * eps02 + eps01 * eps01 +
               eps11 * eps11 + eps12 * eps12 + eps02 * eps02 + eps12 * eps12 +
               eps22 * eps22)

        d_equivalent_plastic_strain_rate[d_idx] = sqrt(2. / 3. * tmp)

        d_equivalent_plastic_strain[d_idx] += (
            d_equivalent_plastic_strain_rate[d_idx] * dt)

        # yield stress is divided into 3 parts, tmp1, tmp2 and tmp3
        tmp1 = (d_jc_a[0] +
                d_jc_b[0] * d_equivalent_plastic_strain[d_idx]**d_jc_n[0])
        tmp2 = (
            1. + d_jc_c[0] *
            log(d_equivalent_plastic_strain_rate[d_idx] / d_jc_eps_dot0[0]))
        tmp3 = (1. - t_star**d_jc_m[0])

        # equation 23 in [2]
        d_yield_stress[d_idx] = tmp1 * tmp2 * tmp3


class ComputeVonMisesStress(Equation):
    def initialize(self, d_idx, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_vonmises):
        pa = d_p[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s02 = d_s02[d_idx]

        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]
        s12 = d_s12[d_idx]

        s20 = d_s02[d_idx]
        s21 = d_s12[d_idx]
        s22 = d_s22[d_idx]

        # Add pressure to the deviatoric components
        s00 = s00 - pa

        s11 = s11 - pa

        s22 = s22 - pa

        d_vonmises[d_idx] = (s00 * s00 + s11 * s11 + s22 * s22 - s00 * s11 -
                             s00 * s22 - s11 * s22 + 3. *
                             (s01 * s01 + s02 * s02 + s12 * s12))


class ComputeJohnsonCookDamageParameter(Equation):
    def initialize(self, d_idx, d_p, d_equivalent_plastic_strain,
                   d_equivalent_plastic_strain_rate, d_jc_t_melting,
                   d_jc_t_room, d_jc_damage_D1, d_jc_damage_D2, d_jc_damage_D3,
                   d_jc_damage_D4, d_jc_damage_D5, d_jc_eps_dot0, d_T,
                   d_epsilon_failure, d_jc_damage_D, d_is_jc_damaged,
                   d_vonmises, dt):
        if d_is_jc_damaged[d_idx] == 0:
            sigma_star = 0.
            if d_vonmises[d_idx] > 1e-12:
                sigma_star = -d_p[d_idx] / d_vonmises[d_idx]

            tmp1 = (d_jc_damage_D1[0] +
                    d_jc_damage_D2[0] * exp(d_jc_damage_D3[0] * sigma_star))
            tmp2 = (1. + d_jc_damage_D4[0] * log(
                d_equivalent_plastic_strain_rate[d_idx] / d_jc_eps_dot0[0]))

            t_room = d_jc_t_room[0]
            t_star = (d_T[d_idx] - t_room) / (d_jc_t_melting[0] - t_room)
            tmp3 = (1. + d_jc_damage_D5[0] * t_star)

            d_epsilon_failure[d_idx] = tmp1 * tmp2 * tmp3

            if d_epsilon_failure[d_idx] > 1e-12:
                d_jc_damage_D[d_idx] = (d_equivalent_plastic_strain[d_idx] /
                                        d_epsilon_failure[d_idx])

            if d_jc_damage_D[d_idx] > 1.:
                d_is_jc_damaged[d_idx] = 1


class MakeDamagedParticleShearStressZero(Equation):
    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_is_jc_damaged, dt):
        if d_is_jc_damaged[d_idx] == 1:
            d_s00[d_idx] = 0.
            d_s01[d_idx] = 0.
            d_s02[d_idx] = 0.
            d_s11[d_idx] = 0.
            d_s12[d_idx] = 0.
            d_s22[d_idx] = 0.


class MieGruneisenEOS(Equation):
    """
    3.2 in [1]
    3.12 in [2]
    [1] The study on performances of kernel types in solid dynamic problems
    by smoothed particle hydrodynamics

    [2] 3D smooth particle hydrodynamics modeling for high velocity
    penetrating impact using GPU: Application to a blunt projectile
    penetrating thin steel plates
    """
    def initialize(self, d_idx, d_rho, d_e, d_p, d_c0_ref, d_mie_eos_rho0,
                   d_mie_eos_s, d_mie_eos_c0, d_mie_eos_gamma0, d_mie_eos_C1,
                   d_mie_eos_C2, d_mie_eos_C3):
        # 3.12 in [2]
        eta = (d_rho[d_idx] / d_mie_eos_rho0[0]) - 1.

        # 3.3 eq in [2]
        if eta > 0.:
            p_H = (d_mie_eos_C1[0] * eta + d_mie_eos_C2[0] * eta * eta +
                   d_mie_eos_C3[0] * eta * eta * eta)
        else:
            p_H = d_mie_eos_C1[0] * eta

        tmp = d_mie_eos_gamma0[0] * d_e[d_idx]
        d_p[d_idx] = (1. - 0.5 * d_mie_eos_gamma0[0] * eta) * p_H + tmp


class LimitDeviatoricStress(Equation):
    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_J2,
                   d_plastic_limit, d_yield_stress):
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
        if d_J2[d_idx] > 1e-12:
            d_plastic_limit[d_idx] = min(
                d_yield_stress[d_idx] / sqrt(3. * d_J2[d_idx]), 1.)
        else:
            d_plastic_limit[d_idx] = 1.

        d_s00[d_idx] = d_plastic_limit[d_idx] * d_s00[d_idx]
        d_s01[d_idx] = d_plastic_limit[d_idx] * d_s01[d_idx]
        d_s02[d_idx] = d_plastic_limit[d_idx] * d_s02[d_idx]
        d_s11[d_idx] = d_plastic_limit[d_idx] * d_s11[d_idx]
        d_s12[d_idx] = d_plastic_limit[d_idx] * d_s12[d_idx]
        d_s22[d_idx] = d_plastic_limit[d_idx] * d_s22[d_idx]


class MakeVelocityAccelerationZero(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_auhat,
                   d_avhat, d_awhat, d_make_acce_zero):

        if d_make_acce_zero[d_idx] == 1:
            d_u[d_idx] = 0.
            d_v[d_idx] = 0.
            d_w[d_idx] = 0.
            d_au[d_idx] = 0.
            d_av[d_idx] = 0.
            d_aw[d_idx] = 0.
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.


class DongRigidBodyForceOnSolid(Equation):
    """
    Equation 31 of [1]

    1. A smoothed particle hydrodynamics (SPH) model for simulating
    surface erosion by impacts of foreign particles, by X.W. Dong
    """
    def loop(self, d_idx, d_m, d_au, d_av, d_aw, s_idx, s_normal, d_spacing0,
             DWIJ, XIJ, d_dong_xeta, dt):
        idx3 = declare('int')
        idx3 = 3 * s_idx

        # compute the dp
        intersection = (XIJ[0] * s_normal[idx3] + XIJ[1] * s_normal[idx3 + 1] +
                        XIJ[2] * s_normal[idx3 + 2])
        # if s_idx == 1:
        #     print(d_idx)
        #     print(intersection)

        # then compute the force
        tmp = 2. / (dt * dt) * (d_spacing0[0] / 2. -
                                intersection) * (1. - d_dong_xeta[0])

        # tmp = 2. / (dt) * (d_spacing0[0] / 2. -
        #                    intersection) * (1. - d_dong_xeta[0])

        if intersection < d_spacing0[0] / 2.:
            d_au[d_idx] += tmp * s_normal[idx3]
            d_av[d_idx] += tmp * s_normal[idx3 + 1]
            d_aw[d_idx] += tmp * s_normal[idx3 + 2]


class DongSolidForceOnRigidBody(Equation):
    """
    Equation 31 of [1]

    1. A smoothed particle hydrodynamics (SPH) model for simulating
    surface erosion by impacts of foreign particles, by X.W. Dong
    """
    def loop(self, d_idx, d_fx, d_fy, d_fz, s_idx, s_m, d_normal, s_spacing0,
             DWIJ, XIJ, s_dong_xeta, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # compute the dp
        intersection = (XIJ[0] * d_normal[idx3] + XIJ[1] * d_normal[idx3 + 1] +
                        XIJ[2] * d_normal[idx3 + 2])

        # then compute the force
        tmp = 2. / (dt * dt) * (s_spacing0[0] / 2. - intersection
                                ) * s_m[s_idx] * (1. - s_dong_xeta[0])

        if intersection < s_spacing0[0] / 2.:
            d_fx[d_idx] -= tmp * d_normal[idx3]
            d_fy[d_idx] -= tmp * d_normal[idx3 + 1]
            d_fz[d_idx] -= tmp * d_normal[idx3 + 2]


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
