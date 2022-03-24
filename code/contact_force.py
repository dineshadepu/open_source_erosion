from pysph.sph.equation import Equation


class ComputeContactForceNormals(Equation):
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
             d_total_no_bodies,
             dt, t, WIJ):
        t1, t2 = declare('int', 3)

        if s_contact_force_is_boundary[s_idx] == 1.:
            if d_dem_id[d_idx] != s_dem_id[s_idx]:
                t1 = d_total_no_bodies[0] * d_idx
                t2 = t1 + s_dem_id[s_idx]
                tmp = d_m[d_idx] / (d_rho[d_idx] * RIJ) * WIJ

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


class ComputeContactForceDistanceAndClosestPoint(Equation):
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

    def loop(self, d_idx, d_m, d_rho,
             s_idx, s_x, s_y, s_z, s_u, s_v, s_w,
             d_contact_force_normal_x,
             d_contact_force_normal_y,
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
             d_normal, dt, t, WIJ, RIJ, XIJ):
        i, t1, t2, t3 = declare('int', 4)

        if s_contact_force_is_boundary[s_idx] == 1.:
            if d_dem_id[d_idx] != s_dem_id[s_idx]:
                t1 = d_total_no_bodies[0] * d_idx
                t2 = t1 + s_dem_id[s_idx]

                tmp = d_m[d_idx] / (d_rho[d_idx]) * WIJ
                tmp_1 = (d_contact_force_normal_x[t2] * XIJ[0] +
                         d_contact_force_normal_y[t2] * XIJ[1])
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


class ComputeContactForce(Equation):
    """Shoya Mohseni Mofidi, Particle Based Numerical Simulation Study of Solid
    Particle Erosion of Ductile Materials Leading to an Erosion Model, Including
    the Particle Shape Effect

    Equation 24 of Materials 2022

    Compute the normals on the rigid body particles (secondary surface) which is
    interacting with wall (primary surface). Here we expect the wall has a
    property identifying its boundary particles. Currently this equation only
    assumes the simulation has only one rigid body with a single wall.

    """
    def __init__(self, dest, sources, fric_coeff=0.5, kr=1e5, kf=1e3):
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        super(ComputeContactForce, self).__init__(dest, sources)

    def post_loop(self,
                  d_idx,
                  d_m,
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
                  d_au,
                  d_av,
                  d_aw,
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
                  d_ti_x,
                  d_ti_y,
                  d_ti_z,
                  d_spacing0,
                  d_total_no_bodies,
                  dt, t):
        i, t1, t2 = declare('int', 3)
        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i

            overlap = d_spacing0[0] - d_contact_force_dist[t2]
            if overlap > 0. and overlap != d_spacing0[0]:
                vij_x = d_u[d_idx] - d_vx_source[t2]
                vij_y = d_v[d_idx] - d_vy_source[t2]
                vij_z = d_w[d_idx] - d_vz_source[t2]

                ni_x = d_contact_force_normal_x[t2]
                ni_y = d_contact_force_normal_y[t2]
                ni_z = d_contact_force_normal_z[t2]

                vij_dot_ni = vij_x * ni_x + vij_y * ni_y + vij_z * ni_z

                d_overlap[t2] = overlap
                tmp = self.kr * overlap

                fn_x = tmp * ni_x
                fn_y = tmp * ni_y
                fn_z = tmp * ni_z

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

                    # repulsive force
                    d_fn_x[t2] = fn_x
                    d_fn_y[t2] = fn_y
                    d_fn_z[t2] = fn_z

                else:
                    # the tangential vector is
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
                    d_ti_x[t2] = ti_x
                    d_ti_y[t2] = ti_y
                    d_ti_z[t2] = ti_z

                    # this is correct
                    delta_lt_x_star = d_delta_lt_x[t2] + vij_x * dt
                    delta_lt_y_star = d_delta_lt_y[t2] + vij_y * dt
                    delta_lt_z_star = d_delta_lt_z[t2] + vij_z * dt

                    delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                    delta_lt_y_star * ti_y +
                                    delta_lt_z_star * ti_y)

                    d_delta_lt_x[t2] = delta_lt_dot_ti * ti_x
                    d_delta_lt_y[t2] = delta_lt_dot_ti * ti_y
                    d_delta_lt_z[t2] = delta_lt_dot_ti * ti_z

                    ft_x_star = -self.kf * d_delta_lt_x[t2]
                    ft_y_star = -self.kf * d_delta_lt_y[t2]
                    ft_z_star = -self.kf * d_delta_lt_z[t2]

                    ft_magn = (ft_x_star**2. + ft_y_star**2. + ft_z_star**2.)**0.5
                    fn_magn = (fn_x**2. + fn_y**2. + fn_z**2.)**0.5

                    ft_magn_star = min(self.fric_coeff * fn_magn, ft_magn)
                    # compute the tangential force, by equation 27
                    d_ft_x[t2] = -ft_magn_star * ti_x
                    d_ft_y[t2] = -ft_magn_star * ti_y
                    d_ft_z[t2] = -ft_magn_star * ti_z

                    # reset the spring length
                    modified_delta_lt_x = -d_ft_x[t2] / self.kf
                    modified_delta_lt_y = -d_ft_y[t2] / self.kf
                    modified_delta_lt_z = -d_ft_z[t2] / self.kf

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
            d_au[d_idx] += (d_fn_x[t2] + d_ft_x[t2]) / d_m[d_idx]
            d_av[d_idx] += (d_fn_y[t2] + d_ft_y[t2]) / d_m[d_idx]
            d_aw[d_idx] += (d_fn_z[t2] + d_ft_z[t2]) / d_m[d_idx]


class ComputeContactForceOnRigidBody(Equation):
    """Shoya Mohseni Mofidi, Particle Based Numerical Simulation Study of Solid
    Particle Erosion of Ductile Materials Leading to an Erosion Model, Including
    the Particle Shape Effect

    Equation 24 of Materials 2022

    Compute the normals on the rigid body particles (secondary surface) which is
    interacting with wall (primary surface). Here we expect the wall has a
    property identifying its boundary particles. Currently this equation only
    assumes the simulation has only one rigid body with a single wall.

    """
    def __init__(self, dest, sources, fric_coeff=0.5, kr=1e5, kf=1e3):
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        super(ComputeContactForceOnRigidBody, self).__init__(dest, sources)

    def post_loop(self,
                  d_idx,
                  d_m,
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
                  d_ti_x,
                  d_ti_y,
                  d_ti_z,
                  d_spacing0,
                  d_total_no_bodies,
                  dt, t):
        i, t1, t2 = declare('int', 3)
        t1 = d_total_no_bodies[0] * d_idx

        for i in range(d_total_no_bodies[0]):
            t2 = t1 + i

            overlap = d_spacing0[0] - d_contact_force_dist[t2]
            if overlap > 0. and overlap != d_spacing0[0]:
                vij_x = d_u[d_idx] - d_vx_source[t2]
                vij_y = d_v[d_idx] - d_vy_source[t2]
                vij_z = d_w[d_idx] - d_vz_source[t2]

                ni_x = d_contact_force_normal_x[t2]
                ni_y = d_contact_force_normal_y[t2]
                ni_z = d_contact_force_normal_z[t2]

                vij_dot_ni = vij_x * ni_x + vij_y * ni_y + vij_z * ni_z

                d_overlap[t2] = overlap
                tmp = self.kr * overlap

                fn_x = tmp * ni_x
                fn_y = tmp * ni_y
                fn_z = tmp * ni_z

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

                    # repulsive force
                    d_fn_x[t2] = fn_x
                    d_fn_y[t2] = fn_y
                    d_fn_z[t2] = fn_z

                else:
                    # the tangential vector is
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
                    d_ti_x[t2] = ti_x
                    d_ti_y[t2] = ti_y
                    d_ti_z[t2] = ti_z

                    # this is correct
                    delta_lt_x_star = d_delta_lt_x[t2] + vij_x * dt
                    delta_lt_y_star = d_delta_lt_y[t2] + vij_y * dt
                    delta_lt_z_star = d_delta_lt_z[t2] + vij_z * dt

                    delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                       delta_lt_y_star * ti_y +
                                       delta_lt_z_star * ti_y)

                    d_delta_lt_x[t2] = delta_lt_dot_ti * ti_x
                    d_delta_lt_y[t2] = delta_lt_dot_ti * ti_y
                    d_delta_lt_z[t2] = delta_lt_dot_ti * ti_z

                    ft_x_star = -self.kf * d_delta_lt_x[t2]
                    ft_y_star = -self.kf * d_delta_lt_y[t2]
                    ft_z_star = -self.kf * d_delta_lt_z[t2]

                    ft_magn = (ft_x_star**2. + ft_y_star**2. + ft_z_star**2.)**0.5
                    fn_magn = (fn_x**2. + fn_y**2. + fn_z**2.)**0.5

                    ft_magn_star = min(self.fric_coeff * fn_magn, ft_magn)
                    # compute the tangential force, by equation 27
                    d_ft_x[t2] = -ft_magn_star * ti_x
                    d_ft_y[t2] = -ft_magn_star * ti_y
                    d_ft_z[t2] = -ft_magn_star * ti_z

                    # reset the spring length
                    modified_delta_lt_x = -d_ft_x[t2] / self.kf
                    modified_delta_lt_y = -d_ft_y[t2] / self.kf
                    modified_delta_lt_z = -d_ft_z[t2] / self.kf

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
            d_fx[d_idx] += (d_fn_x[t2] + d_ft_x[t2])
            d_fy[d_idx] += (d_fn_y[t2] + d_ft_y[t2])
            d_fz[d_idx] += (d_fn_z[t2] + d_ft_z[t2])


class ContactForceDEMOnElasticBody(Equation):
    def __init__(self, dest, sources, fric_coeff=0.5, kr=1e5, kf=1e3):
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        super(ContactForceDEMOnElasticBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw, d_rad_s,
             s_idx, s_x, s_y, s_z, s_u, s_v, s_w, s_rad_s,
             d_dem_id,
             s_dem_id,
             dt, t, WIJ, RIJ, XIJ):
        i, t1, t2, t3 = declare('int', 4)

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
            overlap = RIJ - (d_rad_s[d_idx] + s_rad_s[s_idx])
            if overlap < 0:
                nx = - XIJ[0] / RIJ
                ny = - XIJ[1] / RIJ
                nz = - XIJ[2] / RIJ

                fn = self.kr * overlap

                # add the force
                d_au[d_idx] += fn * nx / d_m[d_idx]
                d_av[d_idx] += fn * ny / d_m[d_idx]
                d_aw[d_idx] += fn * nz / d_m[d_idx]


class ContactForceDEMOnRigidBody(Equation):
    def __init__(self, dest, sources, fric_coeff=0.5, kr=1e5, kf=1e3):
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        super(ContactForceDEMOnRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_fx, d_fy, d_fz, d_rad_s,
             s_idx, s_x, s_y, s_z, s_u, s_v, s_w, s_rad_s,
             d_dem_id,
             s_dem_id,
             dt, t, WIJ, RIJ, XIJ):
        if d_dem_id[d_idx] != s_dem_id[s_idx]:
            overlap = RIJ - (d_rad_s[d_idx] + s_rad_s[s_idx])
            if overlap < 0:
                nx = - XIJ[0] / RIJ
                ny = - XIJ[1] / RIJ
                nz = - XIJ[2] / RIJ

                fn = self.kr * overlap

                # add the force
                d_fx[d_idx] += fn * nx
                d_fy[d_idx] += fn * ny
                d_fz[d_idx] += fn * nz
