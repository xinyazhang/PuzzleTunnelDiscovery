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
