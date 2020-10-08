/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef OMPLAUX_MAP_H
#define OMPLAUX_MAP_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <utility>

namespace omplaux {
struct Map {
	std::vector<Eigen::Vector3d> T;
	std::vector<Eigen::Quaternion<double>> Q;
	std::vector<std::pair<int, int>> E;

	void readMap(const std::string& fn);
};
}

#endif
