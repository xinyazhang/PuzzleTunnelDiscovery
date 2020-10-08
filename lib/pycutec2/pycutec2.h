/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef PYCUTEC2_PYCUTEC2_H
#define PYCUTEC2_PYCUTEC2_H

#include <Eigen/Core>
#include <tuple>

namespace pycutec2 {
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//
// Return Value
//   0: True = intersecting, False = Not
//   1: intersecting tau (in [0,1]) of the (from, to) line segment
//   2: intersecting element in of the mesh
//   3: intersecting tau of the element
//
std::tuple<bool, double, int, double>
line_segment_intersect_with_mesh(Eigen::Vector2d from,
                                 Eigen::Vector2d to,
                                 Eigen::Ref<const RowMatrixXd> V,
                                 Eigen::Ref<const RowMatrixXi> E);

// Batch version of line_segment_intersect_with_mesh
std::tuple<Eigen::VectorXi, Eigen::VectorXd, Eigen::VectorXi, Eigen::VectorXd>
line_segments_intersect_with_mesh(Eigen::Ref<const RowMatrixXd> inV,
                                  Eigen::Ref<const RowMatrixXi> inE,
                                  Eigen::Ref<const RowMatrixXd> obV,
                                  Eigen::Ref<const RowMatrixXi> obE);

std::tuple<
	Eigen::MatrixXd, // Grid V (in 3D with Z == 0.0)
	Eigen::MatrixXi  // Grid E
	  >
build_mesh_2d(const Eigen::Vector2d& lows,
	      const Eigen::Vector2d& highs,
	      const Eigen::Vector2d& nsegments);

void save_obj_1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn);
}

#endif
