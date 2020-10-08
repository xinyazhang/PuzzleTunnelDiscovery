/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/MeshBooleanType.h>
//#define IGL_NO_CORK
//#undef IGL_STATIC_LIBRARY
#include <meshbool/join.h>

#include <iostream>

Eigen::MatrixXd VC;
Eigen::VectorXi J,I;
Eigen::MatrixXi FC;
igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_UNION);

int main(int argc, char *argv[])
{
	if (argc < 2) {
		std::cerr << "Need OBJ file names" << std::endl;
		return -1;
	}

	using namespace Eigen;
	using namespace std;
	std::vector<Eigen::MatrixXd> VArray(argc - 1);
	std::vector<Eigen::MatrixXi> FArray(argc - 1);
	std::cerr << "Loading " << argc << " meshes...";
	
	for(int i = 1; i < argc; i++) {
		igl::readOBJ(argv[i], VArray[i-1], FArray[i-1]);
		std::cerr << "  " << i <<"(f: " << FArray[i-1].rows() <<") ";
	}
	std::cerr << " Done" << std::endl;;

	VC = VArray[0];
	FC = FArray[0];
	std::cerr << "Mesh unoin... " ;
	std::cerr << "  " << 0 <<"(f: " << FC.rows() <<") ";
	for(size_t i = 1; i < VArray.size(); i++) {
		Eigen::MatrixXd RV;
		Eigen::MatrixXi RF;
		try {
			mesh_bool(
					VC,FC,
					VArray[i], FArray[i],
					igl::MESH_BOOLEAN_TYPE_MINUS,
					RV,RF);
			VC.noalias() = RV;
			FC.noalias() = RF;
		} catch (...) {
			std::cerr << "  (" << i << ")";
		}
		igl::writeOBJ("blend-"+std::to_string(i)+".obj", VC, FC);
		std::cerr << "  " << i <<"(f: " << FC.rows() <<") ";
	}
	std::cerr << " Done" << std::endl;;
	igl::writeOBJ("blend.obj", VC, FC);
	return 0;
}
