/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef READ_VORONOI_H
#define READ_VORONOI_H

#include <Eigen/Core>
#include <vector>

struct VoronoiEdge {
	int edgeno;
	int v0idx;
	int v1idx;
	Eigen::Vector3d ray;
};

struct VoronoiFace {
	int faceno;
	int cell0, cell1;
	std::vector<int> edgenos;
};

struct VoronoiCell {
	int cellno;
	std::vector<int> facenos;
};

// We also need cell info for the Voronoi Volume, as the weight of vertices
int readvoronoi(const std::string& prefix,
		Eigen::MatrixXd& vnodes,
		std::vector<VoronoiEdge>& vedges,
		std::vector<VoronoiFace>& vfaces,
		std::vector<VoronoiCell>& vcells
		);

#endif
