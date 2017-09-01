#include "intersect.h"
#include <Eigen/Core>
#include <igl/copyleft/cgal/mesh_boolean.h>

using std::string;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

void mesh_intersect_out(const std::string& pattern, unsigned int p,
		double* points1, unsigned npoints1,
		int* triangles1, unsigned ntriangles1,
		double* points2, unsigned npoints2,
		int* triangles2, unsigned ntriangles2)
{
	Eigen::Map<MatrixXd> MV1(points1, npoints1, 3);
	Eigen::Map<MatrixXi> MF1(triangles1, ntriangles1, 3);
	Eigen::Map<MatrixXd> MV2(points2, npoints2, 3);
	Eigen::Map<MatrixXi> MF2(triangles2, ntriangles2, 3);

	MatrixXd V1(MV1.transpose()), V2(MV2.transpose());
	MatrixXi F1(MF1.transpose()), F2(MF2.transpose());
	MatrixXd VO;
	MatrixXi FO;

	igl::copyleft::cgal::mesh_boolean(
			V1, F1,
			V2, F2,
			igl::MESH_BOOLEAN_TYPE_INTERSECT,
			VO, FO);

	string fn(pattern);
	fn += ".cvx.";
	fn += std::to_string(p);
	fn += ".obj";
	
	igl::writeOBJ(fn, VO.transpose(), FO.transpose());
}
