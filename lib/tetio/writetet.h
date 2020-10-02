/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef WRITETET_H
#define WRITETET_H

#include <Eigen/Core>
#include <string>

void writetet(const std::string& oprefix,
	     const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P
	     );

void
writetet_face(const std::string& oprefix,
	      const Eigen::MatrixXi& F,
	      const Eigen::VectorXi* FBM = nullptr
	      );

#endif
