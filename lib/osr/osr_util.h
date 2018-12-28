#ifndef OSR_UTIL_H
#define OSR_UTIL_H

/*
 * Helper functions Implmeneted in LibIGL
 * 
 * We knew PyMesh already did this, but ... with pyOSR infrastructure it is
 * faster to re-implement in pyOSR than deploy PyMesh to all
 * the machines we are using...
 * 
 * Updates:
 * We may move to PyMesh after it becomes available in pip.
 *                                              - Dec 27 2018
 */

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

#if PYOSR_HAS_CGAL
std::tuple<
	Eigen::Matrix<double, -1, -1>,
	Eigen::Matrix<int, -1, -1>
>
meshBool(const Eigen::Matrix<double, -1, -1>& V0,
         const Eigen::Matrix<int, -1, -1>& F0,
         const Eigen::Matrix<double, -1, -1>& V1,
         const Eigen::Matrix<int, -1, -1>& F1,
         uint32_t op);

extern const uint32_t MESH_BOOL_UNION;
extern const uint32_t MESH_BOOL_INTERSECT;
extern const uint32_t MESH_BOOL_MINUS;
extern const uint32_t MESH_BOOL_XOR;
extern const uint32_t MESH_BOOL_RESOLVE;
#endif

}

#endif
