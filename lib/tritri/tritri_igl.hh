#include "tritri_igl.h"
#include <cassert>

namespace tritri {

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
    Eigen::PlainObjectBase<DerivedCPI>& OCPI)
{
	assert(V0.cols() == 3);
	assert(F0.cols() == 3);
	assert(V1.cols() == 3);
	assert(F1.cols() == 3);
	assert(CI.cols() == 2);

	const size_t m = CI.rows();

	OV = DerivedOutV::Zero(m, 6);
	OCPI = DerivedCPI::Zero(m, 1);
	using Vector = Eigen::Matrix<typename DerivedV0::Scalar, 3, 1>;
	for (size_t i = 0; i < m; i++) {
		Vector pt0, pt1;
		Eigen::Matrix<typename DerivedF0::Scalar, 3, 1> f0, f1;
		f0 = F0.row(CI(i, 0));
		f1 = F1.row(CI(i, 1));
		int coplanar;
#if 1
		Vector v0 = V0.row(f0(0));
		Vector v1 = V0.row(f0(1));
		Vector v2 = V0.row(f0(2));
		Vector u0 = V1.row(f1(0));
		Vector u1 = V1.row(f1(1));
		Vector u2 = V1.row(f1(2));
		TriTriIntersect(v0, v1, v2,
		                u0, u1, u2,
		                &coplanar,
		                pt0, pt1);
#else
		TriTriIntersect(V0.row(f0(0)),
                                V0.row(f0(1)),
                                V0.row(f0(2)),
                                V1.row(f0(0)),
                                V1.row(f0(1)),
                                V1.row(f0(2)),
		                &coplanar,
		                pt0, pt1);
#endif

		OV.template block<1,3>(i, 0) = pt0;
		OV.template block<1,3>(i, 3) = pt1;
		OCPI(i) = coplanar;
	}
}

}
