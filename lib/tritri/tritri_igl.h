#ifndef TRITRI_TRITRI_IGL_H
#define TRITRI_TRITRI_IGL_H

#include "tritri.h"

/*
 * Triangle - Triangle Intersection Function with LIBIGL style interface
 */

namespace tritri {

/*
 * TRITRI Computes the Intersection Segment from pairs of triangles in two
 * trimeshes.
 *
 * Inputs:
 *   V0: #V by 3 list of mesh vertices
 *   F0: #F by 3 list of mesh faces
 *   V1, F1: similar to V0, F0
 *   CI: #C by 2 list of pair of colliding face indices 
 *
 * Outputs: 
 *   OV: #C by 6 list of intersecting segments
 */
template<
	typename DerivedV0,
	typename DerivedF0,
	typename DerivedV1,
	typename DerivedF1,
	typename DerivedCI,
	typename DerivedOutV,
	typename DerivedCPI
	>
void TriTri(
    const Eigen::MatrixBase<DerivedV0>& V0,
    const Eigen::MatrixBase<DerivedF0>& F0,
    const Eigen::MatrixBase<DerivedV1>& V1,
    const Eigen::MatrixBase<DerivedF1>& F1,
    const Eigen::MatrixBase<DerivedCI>& CI,
    Eigen::PlainObjectBase<DerivedOutV>& OV,
    Eigen::PlainObjectBase<DerivedCPI>& OCPI);

}

#include "tritri_igl.hh"

#endif
