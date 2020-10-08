/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "osr_util.h"
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/writePLY.h>
#include <meshbool/join.h>
#include <tritri/tritri_cop.h>

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

#if PYOSR_HAS_MESHBOOL
std::tuple<
	Eigen::Matrix<double, -1, -1>,
	Eigen::Matrix<int, -1, -1>
>
meshBool(const Eigen::Matrix<double, -1, -1>& V0,
         const Eigen::Matrix<int, -1, -1>& F0,
         const Eigen::Matrix<double, -1, -1>& V1,
         const Eigen::Matrix<int, -1, -1>& F1,
         uint32_t op)
{
	if (op >= igl::NUM_MESH_BOOLEAN_TYPES)
		throw std::runtime_error("meshBool: invalid op argument");
	Eigen::Matrix<double, -1, -1> V_out;
	Eigen::Matrix<int, -1, -1> F_out;
	::mesh_bool(V0, F0, V1, F1, static_cast<igl::MeshBooleanType>(op),
	            V_out, F_out);
	return std::make_tuple(V_out, F_out);
}

const uint32_t MESH_BOOL_UNION = igl::MESH_BOOLEAN_TYPE_UNION;
const uint32_t MESH_BOOL_INTERSECT = igl::MESH_BOOLEAN_TYPE_INTERSECT;
const uint32_t MESH_BOOL_MINUS = igl::MESH_BOOLEAN_TYPE_MINUS;
const uint32_t MESH_BOOL_XOR = igl::MESH_BOOLEAN_TYPE_XOR;
const uint32_t MESH_BOOL_RESOLVE = igl::MESH_BOOLEAN_TYPE_RESOLVE;

#endif

Eigen::SparseMatrix<int>
tritriCop(const Eigen::Matrix<double, -1, -1>& V0,
          const Eigen::Matrix<int, -1, -1>& F0,
          const Eigen::Matrix<double, -1, -1>& V1,
          const Eigen::Matrix<int, -1, -1>& F1)
{
	Eigen::SparseMatrix<int> COP;
	tritri::TriTriCop(V0, F0, V1, F1, COP);
	return COP;
}

}
