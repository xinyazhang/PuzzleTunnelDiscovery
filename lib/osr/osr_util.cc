#include "osr_util.h"
#include <igl/writeOBJ.h>

namespace osr {
void saveOBJ1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn)
{
	igl::writeOBJ(fn, V, F);
}

}
