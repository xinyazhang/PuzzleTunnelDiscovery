/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/writeOBJ.h>

namespace pycutec2 {

void save_obj_1(const Eigen::Matrix<double, -1, -1>& V,
	      const Eigen::Matrix<int, -1, -1>& F,
	      const std::string& fn)
{
	igl::writeOBJ(fn, V, F);
}

}
