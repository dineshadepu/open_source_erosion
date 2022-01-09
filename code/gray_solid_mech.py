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
from math import sqrt, acos, log
from math import pi as M_PI


class GraySolidsSchemeTaylorBar(Scheme):
    def __init__(self, solids, boundaries, dim, mach_no,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 gx=0., gy=0., gz=0.):
        # Particle arrays
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # general parameters
        self.dim = dim

        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.mach_no = mach_no

        self.pst = "none"

        # parameters required by equations
        # Artificial viscosity
        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        # boundary conditions
        self.solid_velocity_bc = False
        self.damping = False
        self.damping_coeff = 0.002
        self.mie_gruneisen = False

        # Gravity equation
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=2.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=0.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'solid-velocity-bc', dest='solid_velocity_bc',
            default=False,
            help='Use velocity bc for solid')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.002, type=float,
                           help="Damping coefficient for Bui")

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        choices = ['sun2019', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        add_bool_argument(group, 'mie-gruneisen',
                          dest='mie_gruneisen',
                          default=False,
                          help='Use Mie-Gruneisen eos')

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'damping', 'damping_coeff',
                 'mie_gruneisen', 'pst']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equation()

    def _get_gtvf_equation(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (
            EnergyEquationWithStress as ElasticSolidEnergyEquationWithStress)

        from pysph.sph.solid_mech.basic import (
            MonaghanArtificialStress,
            MomentumEquationWithStress as ElasticSolidMomentumEquationWithStress,
        )

        from solid_mech_common import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,
            ElasticSolidContinuityEquationU,
            ElasticSolidContinuityEquationUSolid,
            VelocityGradient2DSolid,
            HookesDeviatoricStressRate,
            MakeSolidBoundaryParticlesAccelerationsZero,
            MieGruneisenEOS,
            IsothermalEOS,
            AdamiBoundaryConditionExtrapolateNoSlip,
            MonaghanArtificialViscosity,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH,
        )

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationU(
                dest=solid, sources=self.solids))

            g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                g1.append(
                    ElasticSolidContinuityEquationUSolid(
                        dest=solid, sources=self.boundaries))

                g1.append(
                    VelocityGradient2DSolid(dest=solid, sources=self.boundaries))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(
                MonaghanArtificialStress(dest=solid, sources=None,
                                         eps=0.3))
                                         # eps=self.artificial_stress_eps))

            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        tmp = []
        for solid in self.solids:
            if self.mie_gruneisen == True:
                tmp.append(MieGruneisenEOS(dest=solid, sources=None))
            else:
                tmp.append(IsothermalEOS(dest=solid, sources=None))

        stage2.append(Group(tmp))
        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(
                    dest=boundary, sources=self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz
                ))

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

            g4.append(
                ElasticSolidMomentumEquationWithStress(
                    dest=solid,
                    sources=self.solids + self.boundaries))

            g4.append(
                ElasticSolidEnergyEquationWithStress(
                    dest=solid, sources=self.solids + self.boundaries,
                    alpha=self.artificial_vis_alpha,
                    beta=self.artificial_vis_beta))

        stage2.append(Group(g4))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

            if self.damping == True:
                g9.append(
                    BuiFukagawaDampingGraularSPH(
                        dest=solid, sources=None,
                        damping_coeff=self.damping_coeff))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from solid_mech_common import (GTVFSolidMechStepEDACTaylorBar)
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        step_cls = GTVFSolidMechStepEDACTaylorBar

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # Continuity equation properties
            add_properties(pa, 'arho')

            # add the momentum equation variable
            add_properties(pa, 's00', 's01', 's02', 's11', 's12', 's22',
                           'au', 'av', 'aw')
            # Artificial viscosity
            add_properties(pa, 'cs')

            # Tensile instability correction properties
            add_properties(pa, 'r00', 'r01', 'r02', 'r11', 'r12', 'r22')
            kernel = QuinticSpline(dim=self.dim)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)
            # also we need 'n' as a constant.

            # strain rate for Hooke's law
            add_properties(pa, 'v00', 'v01', 'v02', 'v10', 'v11', 'v12', 'v20',
                           'v21', 'v22')

            # set the shear modulus G for Hooke's law
            pa.add_property('G')

            # set the speed of sound and G
            for i in range(len(pa.x)):
                cs = get_speed_of_sound(pa.E[i], pa.nu[i], pa.rho_ref[i])
                G = get_shear_modulus(pa.E[i], pa.nu[i])

                pa.G[i] = G
                pa.cs[i] = cs

            # Shear stress rate for Hooke's law.
            add_properties(pa, 'as00', 'as01', 'as02', 'as11', 'as12', 'as22')

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

            # Properties for integrator and full stress tensor
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

            # for edac
            add_properties(pa, 'ap')

            # for interaction with rigid body
            pa.add_constant('dong_xeta', 0.5)

            # output arrays
            pa.add_output_arrays(['p', 'sigma00', 'sigma01', 'sigma11', 'T', 'e'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what', 'ub', 'vb', 'wb', 'ubhat', 'vbhat',
                           'wbhat')

            add_properties(pa, 'r00', 'r01', 'r02', 'r11', 'r12', 'r22')

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

            pa.add_property('normal', stride=3)
            pa.add_output_arrays(['normal'])
            pa.add_property('normal_tmp', stride=3)

            name = pa.name
            props = ['m', 'rho', 'h']
            for p in props:
                x = pa.get(p)
                if numpy.all(x < 1e-12):
                    msg = f'WARNING: cannot compute normals "{p}" is zero'
                    print(msg)

            seval = SPHEvaluator(
                arrays=[pa], equations=[
                    Group(
                        equations=[ComputeNormals(dest=name, sources=[name])]),
                    Group(
                        equations=[SmoothNormals(dest=name, sources=[name])]),
                ], dim=self.dim)
            seval.evaluate()


class GraySolidsSchemeMetalCutting(Scheme):
    def __init__(self, solids, boundaries, dim, mach_no,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 pst="sun2019", gx=0., gy=0., gz=0.):
        # Particle arrays
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # general parameters
        self.dim = dim

        self.kernel = QuinticSpline
        self.kernel_factor = 2

        # parameters required by equations
        # Artificial viscosity
        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        # Homogenization force
        self.pst = pst
        self.mach_no = mach_no

        # boundary conditions
        self.solid_velocity_bc = False
        self.solid_stress_bc = False
        self.wall_pst = False
        self.damping = False
        self.damping_coeff = 0.002
        self.mie_gruneisen = False

        # Gravity equation
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=2.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=0.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'solid-velocity-bc', dest='solid_velocity_bc',
            default=False,
            help='Use velocity bc for solid')

        add_bool_argument(
            group, 'solid-stress-bc', dest='solid_stress_bc', default=False,
            help='Use stress bc for solid')

        choices = ['sun2019', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=False, help='Add wall as PST source')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.002, type=float,
                           help="Damping coefficient for Bui")

        add_bool_argument(group, 'mie-gruneisen',
                          dest='mie_gruneisen',
                          default=False,
                          help='Use Mie-Gruneisen eos')

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'solid_stress_bc', 'pst',
                 'wall_pst', 'damping', 'damping_coeff', 'mie_gruneisen']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equation()

    def _get_gtvf_equation(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (
            EnergyEquationWithStress as ElasticSolidEnergyEquationWithStress)

        from pysph.sph.solid_mech.basic import (
            MonaghanArtificialStress,

            MomentumEquationWithStress as ElasticSolidMomentumEquationWithStress,
        )

        from solid_mech_common import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,
            ElasticSolidContinuityEquationU,
            ElasticSolidContinuityEquationUSolid,
            VelocityGradient2DSolid,
            HookesDeviatoricStressRate,
            MieGruneisenEOS,
            IsothermalEOS,
            AdamiBoundaryConditionExtrapolateNoSlip,
            MonaghanArtificialViscosity,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH,

        )

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationU(
                dest=solid, sources=self.solids))

            g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                g1.append(
                    ElasticSolidContinuityEquationUSolid(
                        dest=solid, sources=self.boundaries))

                g1.append(
                    VelocityGradient2DSolid(dest=solid, sources=self.boundaries))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(
                MonaghanArtificialStress(dest=solid, sources=None,
                                         eps=0.3))
                                         # eps=self.artificial_stress_eps))

            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        tmp = []
        for solid in self.solids:
            if self.mie_gruneisen == True:
                tmp.append(MieGruneisenEOS(dest=solid, sources=None))
            else:
                tmp.append(IsothermalEOS(dest=solid, sources=None))

        stage2.append(Group(tmp))
        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(
                    dest=boundary, sources=self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz
                ))

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

            g4.append(
                ElasticSolidMomentumEquationWithStress(
                    dest=solid,
                    sources=self.solids + self.boundaries))

            g4.append(
                ElasticSolidEnergyEquationWithStress(
                    dest=solid, sources=self.solids + self.boundaries,
                    alpha=self.artificial_vis_alpha,
                    beta=self.artificial_vis_beta))

        stage2.append(Group(g4))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

            if self.damping == True:
                g9.append(
                    BuiFukagawaDampingGraularSPH(
                        dest=solid, sources=None,
                        damping_coeff=self.damping_coeff))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from solid_mech_common import (GTVFSolidMechStepEDAC)
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        step_cls = GTVFSolidMechStepEDAC

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # Continuity equation properties
            add_properties(pa, 'arho')

            # add the momentum equation variable
            add_properties(pa, 's00', 's01', 's02', 's11', 's12', 's22',
                           'au', 'av', 'aw')
            # Artificial viscosity
            add_properties(pa, 'cs')

            # Tensile instability correction properties
            add_properties(pa, 'r00', 'r01', 'r02', 'r11', 'r12', 'r22')
            kernel = QuinticSpline(dim=self.dim)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)
            # also we need 'n' as a constant.

            # strain rate for Hooke's law
            add_properties(pa, 'v00', 'v01', 'v02', 'v10', 'v11', 'v12', 'v20',
                           'v21', 'v22')

            # set the shear modulus G for Hooke's law
            pa.add_property('G')

            # set the speed of sound and G
            for i in range(len(pa.x)):
                cs = get_speed_of_sound(pa.E[i], pa.nu[i], pa.rho_ref[i])
                G = get_shear_modulus(pa.E[i], pa.nu[i])

                pa.G[i] = G
                pa.cs[i] = cs

            # Shear stress rate for Hooke's law.
            add_properties(pa, 'as00', 'as01', 'as02', 'as11', 'as12', 'as22')

            # Energy equation properties
            add_properties(pa, 'ae', 'e', 'T')
            # this is dummy value of AL, will change for different materials
            pa.add_constant('specific_heat', 875.)
            pa.add_constant('JC_T_room', 273.)

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add

            # Properties for integrator and full stress tensor
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

            # for edac
            add_properties(pa, 'ap')

            # output arrays
            pa.add_output_arrays(['p', 'sigma00', 'sigma01', 'sigma11', 'T', 'e'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what', 'ub', 'vb', 'wb', 'ubhat', 'vbhat',
                           'wbhat')

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

            pa.add_property('normal', stride=3)
            pa.add_output_arrays(['normal'])
            pa.add_property('normal_tmp', stride=3)

            name = pa.name
            props = ['m', 'rho', 'h']
            for p in props:
                x = pa.get(p)
                if numpy.all(x < 1e-12):
                    msg = f'WARNING: cannot compute normals "{p}" is zero'
                    print(msg)

            seval = SPHEvaluator(
                arrays=[pa], equations=[
                    Group(
                        equations=[ComputeNormals(dest=name, sources=[name])]),
                    Group(
                        equations=[SmoothNormals(dest=name, sources=[name])]),
                ], dim=self.dim)
            seval.evaluate()

    def get_solver(self):
        return self.solver


class GraySolidsSchemeSolidErosion(Scheme):
    def __init__(self, solids, rigid_bodies, dim,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 gx=0., gy=0., gz=0.):
        # Particle arrays
        self.solids = solids
        self.rigid_bodies = rigid_bodies

        # general parameters
        self.dim = dim

        self.kernel = QuinticSpline
        self.kernel_factor = 2

        # parameters required by equations
        # Artificial viscosity
        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        # boundary conditions
        self.solid_velocity_bc = False
        self.damping = False
        self.damping_coeff = 0.002
        self.mie_gruneisen = False

        # Gravity equation
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=2.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=0.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'solid-velocity-bc', dest='solid_velocity_bc',
            default=False,
            help='Use velocity bc for solid')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.002, type=float,
                           help="Damping coefficient for Bui")

        add_bool_argument(group, 'mie-gruneisen',
                          dest='mie_gruneisen',
                          default=False,
                          help='Use Mie-Gruneisen eos')

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'damping', 'damping_coeff',
                 'mie_gruneisen']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equation()

    def _get_gtvf_equation(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (
            EnergyEquationWithStress as ElasticSolidEnergyEquationWithStress)

        from pysph.sph.solid_mech.basic import (
            MonaghanArtificialStress,
            MomentumEquationWithStress as ElasticSolidMomentumEquationWithStress,
        )

        from solid_mech_common import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,
            ElasticSolidContinuityEquationU,
            ElasticSolidContinuityEquationUSolid,
            VelocityGradient2DSolid,
            HookesDeviatoricStressRate,
            MakeSolidBoundaryParticlesAccelerationsZero,
            MieGruneisenEOS,
            IsothermalEOS,
            AdamiBoundaryConditionExtrapolateNoSlip,
            MonaghanArtificialViscosity,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH,

            # interaction between to rigid body and elastic solid
            ElasticSolidRigidBodyForce,
            RigidBodyElasticSolidForce
        )

        from rigid_body_3d import (
            BodyForce,
            SumUpExternalForces
        )

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationU(
                dest=solid, sources=self.solids))

            g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(
                MonaghanArtificialStress(dest=solid, sources=None,
                                         eps=0.3))
                                         # eps=self.artificial_stress_eps))

            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        tmp = []
        for solid in self.solids:
            if self.mie_gruneisen == True:
                tmp.append(MieGruneisenEOS(dest=solid, sources=None))
            else:
                tmp.append(IsothermalEOS(dest=solid, sources=None))

        stage2.append(Group(tmp))

        # -------------------
        # boundary conditions
        # -------------------
        for solid in self.solids:
            g3.append(
                AdamiBoundaryConditionExtrapolateNoSlip(
                    dest=solid, sources=self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz
                ))
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
                        dest=solid, sources=[solid],
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(
                ElasticSolidMomentumEquationWithStress(
                    dest=solid,
                    sources=self.solids))

            # Force on the elastic solid due to rigid body
            g4.append(
                ElasticSolidRigidBodyForce(
                    dest=solid,
                    sources=self.rigid_bodies))

            g4.append(
                ElasticSolidEnergyEquationWithStress(
                    dest=solid, sources=self.solids,
                    alpha=self.artificial_vis_alpha,
                    beta=self.artificial_vis_beta))

        stage2.append(Group(g4))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

            if self.damping == True:
                g9.append(
                    BuiFukagawaDampingGraularSPH(
                        dest=solid, sources=None,
                        damping_coeff=self.damping_coeff))

        stage2.append(Group(equations=g9))

        g10 = []
        for solid in self.solids:
            g10.append(
                MakeSolidBoundaryParticlesAccelerationsZero(
                    dest=solid, sources=None))

        stage2.append(Group(equations=g10))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    BodyForce(dest=name,
                              sources=None,
                              gx=self.gx,
                              gy=self.gy,
                              gz=self.gz))

            for name in self.rigid_bodies:
                g5.append(RigidBodyElasticSolidForce(
                    dest=name, sources=self.solids))

            stage2.append(Group(equations=g5, real=False))

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name, sources=None))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from solid_mech_common import (GTVFSolidMechStepEDAC)
        from rigid_body_3d import (GTVFRigidBody3DStep)
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        step_cls = GTVFSolidMechStepEDAC

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        step_cls = GTVFRigidBody3DStep
        for name in self.rigid_bodies:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        from rigid_body_3d import (set_total_mass, set_center_of_mass,
                                   set_moment_of_inertia_and_its_inverse,
                                   set_body_frame_position_vectors,
                                   add_boundary_identification_properties,
                                   get_boundary_identification_etvf_equations,
                                   set_body_frame_normal_vectors)

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # Continuity equation properties
            add_properties(pa, 'arho')

            # add the momentum equation variable
            add_properties(pa, 's00', 's01', 's02', 's11', 's12', 's22',
                           'au', 'av', 'aw', 'wij')
            # Artificial viscosity
            add_properties(pa, 'cs')

            # Tensile instability correction properties
            add_properties(pa, 'r00', 'r01', 'r02', 'r11', 'r12', 'r22')
            kernel = QuinticSpline(dim=self.dim)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)
            # also we need 'n' as a constant.

            # strain rate for Hooke's law
            add_properties(pa, 'v00', 'v01', 'v02', 'v10', 'v11', 'v12', 'v20',
                           'v21', 'v22')

            # set the shear modulus G for Hooke's law
            pa.add_property('G')

            # set the speed of sound and G
            for i in range(len(pa.x)):
                cs = get_speed_of_sound(pa.E[i], pa.nu[i], pa.rho_ref[i])
                G = get_shear_modulus(pa.E[i], pa.nu[i])

                pa.G[i] = G
                pa.cs[i] = cs

            # Shear stress rate for Hooke's law.
            add_properties(pa, 'as00', 'as01', 'as02', 'as11', 'as12', 'as22')

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

            # Properties for integrator and full stress tensor
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

            # for edac
            add_properties(pa, 'ap')

            # for interaction with rigid body
            pa.add_constant('dong_xeta', 0.5)

            # output arrays
            pa.add_output_arrays(['p', 'sigma00', 'sigma01', 'sigma11', 'T', 'e'])

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
