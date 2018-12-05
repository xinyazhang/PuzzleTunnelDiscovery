#ifndef OSR_UTIL_H
#define OSR_UTIL_H

#include <Eigen/Core>
#include <string>
#include <tuple>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn);

std::tuple<
	Eigen::Matrix<double, -1, -1>,
	Eigen::Matrix<int, -1, -1>
>
loadOBJ1(const std::string& fn);
}

#endif
