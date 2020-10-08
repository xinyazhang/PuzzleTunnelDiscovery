/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "pycutec2.h"
#include <Eigen/Dense>

namespace pycutec2 {

std::tuple<bool, double, int, double>
line_segment_intersect_with_mesh(Eigen::Vector2d from,
                                 Eigen::Vector2d to,
                                 Eigen::Ref<const RowMatrixXd> V,
                                 Eigen::Ref<const RowMatrixXi> E)
{
	bool hit = false;
	double tau0 = 2.0;
	int hit_id = -1;
	double tau1 = 2.0;
	for (int i = 0; i < E.rows(); i++) {
		int fi = E(i, 0);
		int ti = E(i, 1);
		Eigen::Vector2d sfrom = V.row(fi).segment<2>(0);
		Eigen::Vector2d sto = V.row(ti).segment<2>(0);
		Eigen::Matrix2d A;
		A.col(0) = sto - sfrom;
		A.col(1) = from - to;
		Eigen::Vector2d b;
		b = from - sfrom;
		// Solve A * tau = b
		if (A.determinant() == 0)
			continue;
		Eigen::Vector2d tau = A.fullPivLu().solve(b);
		if (tau(0) < 0 || tau(0) > 1.0)
			continue;
		if (tau(1) < 0 || tau(1) > 1.0)
			continue;
		hit = true;
		if (tau(0) < tau0) {
			tau0 = tau(0);
			hit_id = i;
			tau1 = tau(1);
		}
	}
	return std::make_tuple(hit, tau0, hit_id, tau1);
}

std::tuple<Eigen::VectorXi, Eigen::VectorXd, Eigen::VectorXi, Eigen::VectorXd>
line_segments_intersect_with_mesh(Eigen::Ref<const RowMatrixXd> inV,
                                  Eigen::Ref<const RowMatrixXi> inE,
                                  Eigen::Ref<const RowMatrixXd> obV,
                                  Eigen::Ref<const RowMatrixXi> obE)
{
	int N = inE.rows();
	Eigen::VectorXi retbool(N);
       	Eigen::VectorXd rettau1(N);
	Eigen::VectorXi retelem(N);
	Eigen::VectorXd rettau2(N);
	for (int i = 0; i < inE.rows(); i++) {
		auto tup = line_segment_intersect_with_mesh(inV.row(inE(i, 0)),
				                            inV.row(inE(i, 1)),
							    obV,
							    obE);
		retbool(i) = std::get<0>(tup);
		rettau1(i) = std::get<1>(tup);
		retelem(i) = std::get<2>(tup);
		rettau2(i) = std::get<3>(tup);
	}
	return std::make_tuple(retbool, rettau1, retelem, rettau2);
}

}
