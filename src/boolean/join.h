#ifndef JOIN_H
#define JOIN_H

#include <Eigen/Core>
#include <igl/MeshBooleanType.h>

void mesh_bool(
		const Eigen::MatrixXd& VA, const Eigen::MatrixXi& FA,
		const Eigen::MatrixXd& VB, const Eigen::MatrixXi& FB,
		igl::MeshBooleanType,
		Eigen::MatrixXd& VC, Eigen::MatrixXi& FC);

#endif
