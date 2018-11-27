#ifndef OSR_UTIL_H
#define OSR_UTIL_H

#include <Eigen/Core>
#include <string>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn);
}

#endif
