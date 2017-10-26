#include "join.h"
#include <igl/copyleft/cgal/mesh_boolean.h>

void mesh_bool(
		const Eigen::MatrixXd& VA, const Eigen::MatrixXi& FA,
		const Eigen::MatrixXd& VB, const Eigen::MatrixXi& FB,
		igl::MeshBooleanType boolean_type,
		Eigen::MatrixXd& VC, Eigen::MatrixXi& FC)
{
	igl::copyleft::cgal::mesh_boolean(
			VA,FA,
			VB,FB,
			boolean_type,
			VC,FC);
}
