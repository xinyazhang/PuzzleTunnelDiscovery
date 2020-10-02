/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#include <Eigen/Core>

void san_check(
	const Eigen::MatrixXf& IV,
	const Eigen::MatrixXi& IF,
	const Eigen::MatrixXf& OV,
	const Eigen::MatrixXi& OF,
	double expected_distance
	);
