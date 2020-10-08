/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "pycutec2.h"

namespace pycutec2 {

Eigen::VectorXd interp(double low, double high, int nseg)
{
	Eigen::VectorXd ret(nseg + 1);
	for (int i = 0; i < nseg; i++) {
		ret(i) = low + i * (high - low)/nseg;
	}
	ret(nseg) = high;
	return ret;
}

std::tuple<
	Eigen::MatrixXd, // Grid V (in 3D with Z == 0.0)
	Eigen::MatrixXi  // Grid E
	  >
build_mesh_2d(const Eigen::Vector2d& lows,
	      const Eigen::Vector2d& highs,
	      const Eigen::Vector2d& nsegments)
{
	Eigen::MatrixXd retV; 
	Eigen::MatrixXi retE; 
	int nX = nsegments(0);
	int nY = nsegments(1);
	auto X = interp(lows(0), highs(0), nX);
	auto Y = interp(lows(1), highs(1), nY);
	retV.resize(X.rows() * Y.rows(), 3);
	int index = 0;
	for (int j = 0; j < Y.rows(); j++) {
		for (int i = 0; i < X.rows(); i++) {
			retV.row(index++) << X(i), Y(j), 0.0;
		}
	}
	index = 0;
	retE.resize(nX * nY * 2, 3);
	for (int j = 0; j < nY; j++) {
		for (int i = 0; i < nX; i++) {
			int tl = (j + 0) * X.rows() + i + 0;
			int tr = (j + 0) * X.rows() + i + 1;
			int bl = (j + 1) * X.rows() + i + 0;
			int br = (j + 1) * X.rows() + i + 1;
			retE.row(index++) << tr, tl, bl;
			retE.row(index++) << bl, br, tr;
		}
	}
	return std::make_tuple(retV, retE);
}

}
