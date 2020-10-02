/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef READTET_H
#define READTET_H

#include <Eigen/Core>
#include <string>

// = readtet
//
// [out] V: vertices
// [out] E: edges
// [out] P: simplex primitives
// [out] EBMarker: vector of edge boundary markers, set to null to read boundary edges only
// [in] prefix: the common prefix of tetgen .node and .ele files.
//
// [ret] base: the base of vertex index, tetgen doesn't specifiy it must be
// zero
//
// Throws std::runtime_error
//
void readtet(const std::string& prefix,
	     Eigen::MatrixXd& V,
	     Eigen::MatrixXi& E,
	     Eigen::MatrixXi& P,
	     Eigen::VectorXi* EBMarker
	     );

int readtet(const std::string& prefix,
	     Eigen::MatrixXd& V,
	     Eigen::MatrixXi& P);

void readtet_face(const std::string& prefix,
	     Eigen::MatrixXi& F,
	     Eigen::VectorXi* FBMarker);

#endif
