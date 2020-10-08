/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef TRITRI_TRITRI_H
#define TRITRI_TRITRI_H

/*
 * Triangle - Triangle Intersection Function
 *     based on "A Fast Triangle-Triangle Intersection Test" by Tomas Moller
 * 
 * Note: due to performance issues, FCL is more preferred. These functions
 *       should be reserved for intersecting segment calculation
 */

#include <Eigen/Core>

namespace tritri {

using Eigen::Ref;
using Eigen::Vector3f;
using Eigen::Vector3d;
using RVector3f = Eigen::Matrix<float, 1, 3>;
using RVector3d = Eigen::Matrix<double, 1, 3>;

/*
 * Engineering considerations: passing Eigen::Ref may perform extra copies,
 * but this is desired because Moller97's code assumes continuous storage by
 * using float [3] as their input/output data type.
 * 
 * A proper solution is to rewrite Moller97.c as a set of Eigen-based C++ template
 * functions.
 */

bool TriTriIntersect(const Ref<Vector3f> v0,
		     const Ref<Vector3f> v1,
		     const Ref<Vector3f> v2,
		     const Ref<Vector3f> u0,
		     const Ref<Vector3f> u1,
		     const Ref<Vector3f> u2,
                     int *coplanar,
                     Ref<Vector3f> isectpt1,
		     Ref<Vector3f> isectpt2);

bool TriTriIntersect(const Ref<Vector3d> v0,
		     const Ref<Vector3d> v1,
		     const Ref<Vector3d> v2,
		     const Ref<Vector3d> u0,
		     const Ref<Vector3d> u1,
		     const Ref<Vector3d> u2,
                     int *coplanar,
                     Ref<Vector3d> isectpt1,
		     Ref<Vector3d> isectpt2);

bool TriTriIntersect(const Ref<RVector3f> v0,
		     const Ref<RVector3f> v1,
		     const Ref<RVector3f> v2,
		     const Ref<RVector3f> u0,
		     const Ref<RVector3f> u1,
		     const Ref<RVector3f> u2,
                     int *coplanar,
                     Ref<RVector3f> isectpt1,
		     Ref<RVector3f> isectpt2);

bool TriTriIntersect(const Ref<RVector3f> v0,
		     const Ref<RVector3f> v1,
		     const Ref<RVector3f> v2,
		     const Ref<RVector3f> u0,
		     const Ref<RVector3f> u1,
		     const Ref<RVector3f> u2,
                     int *coplanar,
                     Ref<RVector3f> isectpt1,
		     Ref<RVector3f> isectpt2);

}

#endif
