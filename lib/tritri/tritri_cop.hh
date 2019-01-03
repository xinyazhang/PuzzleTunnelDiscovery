#include "tritri_cop.h"
#include <cassert>
#include <vector>
#include <igl/per_face_normals.h>
#include <Eigen/Geometry>
#include "tritri.h"

namespace tritri {
#define EPSILON 0.000001

// Return true if coplanar
// Internal usage only
// Simplified from Moller97's compute_intervals_isectline
template<typename Scalar>
bool _coplanar_kernel(Scalar D0, Scalar D1,Scalar D2, Scalar D0D1,Scalar D0D2)
{
	if (D0D1>0.0f)
		return false;
	else if (D0D2>0.0f)
		return false;
	else if (D1*D2>0.0f || D0!=0.0f)
		return false;
	else if (D1!=0.0f)
		return false;
	else if (D2!=0.0f)
		return false;
	return true;
}

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
	Eigen::SparseMatrix<T>& COP)
{
	assert(V0.cols() == 3);
	assert(F0.cols() == 3);
	assert(V1.cols() == 3);
	assert(F1.cols() == 3);
	std::vector<Eigen::Triplet<T>> tups;
	COP.resize(F0.rows(), F1.rows());

	using Scalar = typename DerivedV0::Scalar;
	using VectorXS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using Vector3S = Eigen::Matrix<Scalar, 3, 1>;
	using Matrix3S = Eigen::Matrix<Scalar, 3, 3>;
	using MatrixXS = Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic>;
	MatrixXS Ns0, Ns1;
	igl::per_face_normals(V0, F0, Ns0);
	igl::per_face_normals(V1, F1, Ns1);

	// Simplified from Moller97's tri-tri intersection algorithm.
	VectorXS d1s(F0.rows()), d2s(F1.rows());
	for (size_t i = 0; i < F0.rows(); i++) {
		d1s(i) = -Ns0.row(i).dot(V0.row(F0(i,0)));
	}
	for (size_t j = 0; j < F1.rows(); j++) {
		d2s(j) = -Ns1.row(j).dot(V1.row(F1(j,0)));
	}

	for (size_t i = 0; i < F0.rows(); i++) {
		Vector3S N1 = Ns0.row(i);

		// Moller97 use V for first triangle vertices
		//          use U for second triangle vertices
		Matrix3S Vs;
		Vs.row(0) = V0.row(F0(i,0));
		Vs.row(1) = V0.row(F0(i,1));
		Vs.row(2) = V0.row(F0(i,2));
		for (size_t j = 0; j < F1.rows(); j++) {
			Matrix3S Us;
			Us.row(0) = V1.row(F1(j,0));
			Us.row(1) = V1.row(F1(j,1));
			Us.row(2) = V1.row(F1(j,2));
			Vector3S dus = (Us * N1).array() + d1s(i);
			for (int k = 0; k < 3; k++)
				if (std::abs(dus(k)) < EPSILON)
					dus(k) = 0.0;
			Scalar du0du1 = dus(0) * dus(1);
			Scalar du0du2 = dus(0) * dus(2);
			if (du0du1>0.0f && du0du2>0.0f)
				continue;

			Vector3S N2 = Ns1.row(j);
			Vector3S dvs = (Vs * N2).array() + d2s(j);
			for (int k = 0; k < 3; k++)
				if (std::abs(dvs(k)) < EPSILON)
					dvs(k) = 0.0;
			Scalar dv0dv1 = dvs(0) * dvs(1);
			Scalar dv0dv2 = dvs(0) * dvs(2);
			if (dv0dv1>0.0f && dv0dv2>0.0f)
				continue;
			Vector3S D = N1.cross(N2);
			if (!_coplanar_kernel(dvs(0), dvs(1), dvs(2), dv0dv1, dv0dv2))
				continue;
			tups.emplace_back(i, j, 1);
		}
	}
	COP.setFromTriplets(tups.begin(), tups.end());
}

template<
	typename DerivedV0,
	typename DerivedF0,
	typename DerivedV1,
	typename DerivedF1,
	typename T>
void TriTriCopIsect(
	const Eigen::MatrixBase<DerivedV0>& V0,
	const Eigen::MatrixBase<DerivedF0>& F0,
	const Eigen::MatrixBase<DerivedV1>& V1,
	const Eigen::MatrixBase<DerivedF1>& F1,
	Eigen::SparseMatrix<T>& COP)
{
	assert(V0.cols() == 3);
	assert(F0.cols() == 3);
	assert(V1.cols() == 3);
	assert(F1.cols() == 3);
	std::vector<Eigen::Triplet<T>> tups;
	COP.resize(F0.rows(), F1.rows());

	using Scalar = typename DerivedV0::Scalar;
	using Vector3S = Eigen::Matrix<Scalar, 3, 1>;
	Vector3S isectpt0, isectpt1;
	int cop;
	for (size_t i = 0; i < F0.rows(); i++) {
		Vector3S v0 = V0.row(F0(i, 0));
		Vector3S v1 = V0.row(F0(i, 1));
		Vector3S v2 = V0.row(F0(i, 2));
		for (size_t j = 0; j < F1.rows(); j++) {
			Vector3S u0 = V1.row(F1(j, 0));
			Vector3S u1 = V1.row(F1(j, 1));
			Vector3S u2 = V1.row(F1(j, 2));
			bool isect = TriTriIntersect(v0, v1, v2,
			                             u0, u1, u2,
			                             &cop,
			                             isectpt0, isectpt1);
			if (isect and cop) {
				tups.emplace_back(i, j, 1);
			}
		}
	}
	COP.setFromTriplets(tups.begin(), tups.end());
}


}
