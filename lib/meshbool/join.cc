/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
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

void mesh_bool(const Eigen::Matrix<double, -1, 3>& VA, const Eigen::Matrix<int, -1, 3>& FA,
               const Eigen::Matrix<double, -1, 3>& VB, const Eigen::Matrix<int, -1, 3>& FB,
               igl::MeshBooleanType boolean_type,
               Eigen::Matrix<double, -1, 3>& VC, Eigen::Matrix<int, -1, 3>& FC)
{
	igl::copyleft::cgal::mesh_boolean(
			VA,FA,
			VB,FB,
			boolean_type,
			VC,FC);
}
