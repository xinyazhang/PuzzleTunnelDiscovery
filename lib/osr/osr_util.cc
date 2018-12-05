#include "osr_util.h"
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn)
{
	igl::writeOBJ(fn, V, F);
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
