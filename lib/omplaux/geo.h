/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OMPLAUX_GEO_H
#define OMPLAUX_GEO_H

#include <Eigen/Core>
#include <vector>

struct Geo {
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> GPUV;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N;

	Eigen::Vector3d center;

	void read(const std::string& fn);

	std::vector<Eigen::MatrixXd> cvxV;
	std::vector<Eigen::MatrixXi> cvxF;

	void readcvx(const std::string& prefix);
	void readtet(const std::string& prefix);
};

#endif
