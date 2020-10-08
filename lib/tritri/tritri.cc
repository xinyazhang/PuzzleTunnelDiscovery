/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "tritri.h"

extern
int tri_tri_intersect_with_isectline(const float V0[3], const float V1[3], const float V2[3],
				     const float U0[3], const float U1[3], const float U2[3],
				     int *coplanar,
				     float isectpt1[3], float isectpt2[3]);
extern
int tri_tri_intersect_with_isectline(const double V0[3], const double V1[3], const double V2[3],
				     const double U0[3], const double U1[3], const double U2[3],
				     int *coplanar,
				     double isectpt1[3], double isectpt2[3]);

namespace tritri {

bool
TriTriIntersect(const Ref<Vector3f> v0,
                const Ref<Vector3f> v1,
                const Ref<Vector3f> v2,
                const Ref<Vector3f> u0,
                const Ref<Vector3f> u1,
                const Ref<Vector3f> u2,
                int *coplanar,
                Ref<Vector3f> isectpt1,
                Ref<Vector3f> isectpt2)
{
	return tri_tri_intersect_with_isectline(v0.data(), v1.data(), v2.data(),
	                                        u0.data(), u1.data(), u2.data(),
	                                        coplanar,
	                                        isectpt1.data(), isectpt2.data());
}

bool
TriTriIntersect(const Ref<Vector3d> v0,
                const Ref<Vector3d> v1,
                const Ref<Vector3d> v2,
                const Ref<Vector3d> u0,
                const Ref<Vector3d> u1,
                const Ref<Vector3d> u2,
                int *coplanar,
                Ref<Vector3d> isectpt1,
                Ref<Vector3d> isectpt2)
{
	return tri_tri_intersect_with_isectline(v0.data(), v1.data(), v2.data(),
	                                        u0.data(), u1.data(), u2.data(),
	                                        coplanar,
	                                        isectpt1.data(), isectpt2.data());
}

bool
TriTriIntersect(const Ref<RVector3f> v0,
                const Ref<RVector3f> v1,
                const Ref<RVector3f> v2,
                const Ref<RVector3f> u0,
                const Ref<RVector3f> u1,
                const Ref<RVector3f> u2,
                int *coplanar,
                Ref<RVector3f> isectpt1,
                Ref<RVector3f> isectpt2)
{
	return tri_tri_intersect_with_isectline(v0.data(), v1.data(), v2.data(),
	                                        u0.data(), u1.data(), u2.data(),
	                                        coplanar,
	                                        isectpt1.data(), isectpt2.data());
}

bool
TriTriIntersect(const Ref<RVector3d> v0,
                const Ref<RVector3d> v1,
                const Ref<RVector3d> v2,
                const Ref<RVector3d> u0,
                const Ref<RVector3d> u1,
                const Ref<RVector3d> u2,
                int *coplanar,
                Ref<RVector3d> isectpt1,
                Ref<RVector3d> isectpt2)
{
	return tri_tri_intersect_with_isectline(v0.data(), v1.data(), v2.data(),
	                                        u0.data(), u1.data(), u2.data(),
	                                        coplanar,
	                                        isectpt1.data(), isectpt2.data());
}

}
