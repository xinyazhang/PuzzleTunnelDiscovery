/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/MeshBooleanType.h>
#include <unistd.h>
//#define IGL_NO_CORK
//#undef IGL_STATIC_LIBRARY
#include <iostream>

#include <meshbool/join.h>

int main(int argc, char *argv[])
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	double margin = 0.1;
	int opt;

	while ((opt = getopt(argc, argv, "m:")) != -1) {
		switch (opt) {
			case 'm':
				margin = atof(optarg);
				break;
			case '?':
				fprintf(stderr, "Unrecognized option\n");
				return -1;
		};
	}
	igl::readOBJ("/dev/stdin", V, F);
	Eigen::VectorXd maxV = V.colwise().maxCoeff();
	Eigen::VectorXd minV = V.colwise().minCoeff();

	using namespace Eigen;
	using namespace std;
	std::vector<Eigen::MatrixXd> VArray(argc - 1);
	std::vector<Eigen::MatrixXi> FArray(argc - 1);
	double coords[3][2] = { {minV(0) - margin, maxV(0) + margin},
				{minV(1) - margin, maxV(1) + margin},
				{0, 2 * M_PI} };
	Eigen::MatrixXd BV(8, 3);
	Eigen::ArrayXXi BF(12, 3);

	size_t i = 0;
	for(int ix = 0; ix < 2; ix++) {
		for(int iy = 0; iy < 2; iy++) {
			for(int iz = 0; iz < 2; iz++) {
				BV(i, 0) = coords[0][ix];
				BV(i, 1) = coords[1][iy];
				BV(i, 2) = coords[2][iz];
				i++;
			}
		}
	}
	BF << 1, 5, 2,
	      5, 6, 2,
	      5, 7, 6,
	      6, 7, 8,
	      2, 6, 4,
	      4, 6, 8,
	      3, 2, 4,
	      2, 3, 1,
	      1, 3, 5,
	      5, 3, 7,
	      4, 8, 7,
	      3, 4, 7;
	BF = BF - 1;
	
	Eigen::MatrixXd RV;
	Eigen::MatrixXi RF;
	mesh_bool(BV, BF,
		  V, F,
		  igl::MESH_BOOLEAN_TYPE_MINUS,
		  RV, RF);
#if 1
	std::cout << RV.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << (RF.array()+1).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"));
#endif
#if 0
	std::cout << BV.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << (BF.array()+1).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"));
#endif
	return 0;
}
