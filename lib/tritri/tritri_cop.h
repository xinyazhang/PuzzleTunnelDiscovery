#ifndef TRITRI_TRITRI_COP_H
#define TRITRI_TRITRI_COP_H

/*
 * Triangle - Triangle Coplanar Function with LIBIGL style interface
 */
#include "tritri.h"
#include <Eigen/SparseCore>

/*
 * TriTriCop Computes the coplanar faces from pairs of triangles in two
 * trimeshes.
 *
 * Inputs:
 *   V0: #V by 3 list of mesh vertices
 *   F0: #F by 3 list of mesh faces
 *   V1, F1: similar to V0, F0
 *
 * Outputs: 
 *   COP: Sparse Matrix with non-zero element (i,j) indicating Face i from
 *        Mesh 0 is coplanar with Face j from Mesh 1
 */
namespace tritri {
template<
	typename DerivedV0,
	typename DerivedF0,
	typename DerivedV1,
	typename DerivedF1,
	typename T>
void TriTriCop(
	const Eigen::MatrixBase<DerivedV0>& V0,
	const Eigen::MatrixBase<DerivedF0>& F0,
	const Eigen::MatrixBase<DerivedV1>& V1,
	const Eigen::MatrixBase<DerivedF1>& F1,
	Eigen::SparseMatrix<T>& COP);

}

#include "tritri_cop.hh"

#endif
