#ifndef OSR_UTIL_H
#define OSR_UTIL_H

#include <Eigen/Core>
#include <string>
#include <tuple>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn);

void saveOBJ2(const Eigen::Matrix<double, -1, -1>& V,
              const Eigen::Matrix<int, -1, -1>& F,
              const Eigen::Matrix<double, -1, -1>& CN,
              const Eigen::Matrix<int, -1, -1>& FN,
              const Eigen::Matrix<double, -1, -1>& TC,
              const Eigen::Matrix<int, -1, -1>& FTC,
              const std::string& fn);

//
// OBJ file does not support vertex normal on point cloud
// Hence we need ply
// 
void savePLY2(const Eigen::Matrix<double, -1, -1>& V,
              const Eigen::Matrix<int, -1, -1>& F,
              const Eigen::Matrix<double, -1, -1>& N,
              const Eigen::Matrix<double, -1, -1>& UV,
              const std::string& fn);

std::tuple<
	Eigen::Matrix<double, -1, -1>,
	Eigen::Matrix<int, -1, -1>
>
loadOBJ1(const std::string& fn);
}

#endif
