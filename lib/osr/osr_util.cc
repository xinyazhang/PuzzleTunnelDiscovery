#include "osr_util.h"
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/writePLY.h>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn)
{
	igl::writeOBJ(fn, V, F);
}

void saveOBJ2(const Eigen::Matrix<double, -1, -1>& V,
              const Eigen::Matrix<int, -1, -1>& F,
              const Eigen::Matrix<double, -1, -1>& CN,
              const Eigen::Matrix<int, -1, -1>& FN,
              const Eigen::Matrix<double, -1, -1>& TC,
              const Eigen::Matrix<int, -1, -1>& FTC,
              const std::string& fn)
{
	igl::writeOBJ(fn, V, F, CN, FN, TC, FTC);
}

void savePLY2(const Eigen::Matrix<double, -1, -1>& V,
              const Eigen::Matrix<int, -1, -1>& F,
              const Eigen::Matrix<double, -1, -1>& N,
              const Eigen::Matrix<double, -1, -1>& UV,
              const std::string& fn)
{
	igl::writePLY(fn, V, F, N, UV);
}


std::tuple<
	Eigen::Matrix<double, -1, -1>,
	Eigen::Matrix<int, -1, -1>
>
loadOBJ1(const std::string& fn)
{
	Eigen::Matrix<double, -1, -1> V;
	Eigen::Matrix<int, -1, -1> F;
	igl::readOBJ(fn, V, F);
	return std::make_tuple(V, F);
}

}
