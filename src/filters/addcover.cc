/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/readOBJ.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <meshbool/join.h>

void set_coords(Eigen::MatrixXd& BV, double coords[3][2])
{
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
}

int main(int argc, char* argv[])
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	double margin = 0.1/2;
	double zmargin = M_PI/256.0;
	int opt;

	while ((opt = getopt(argc, argv, "m:z:")) != -1) {
		switch (opt) {
			case 'm':
				margin = atof(optarg);
				break;
			case 'z':
				zmargin = atof(optarg);
				break;
			case '?':
				fprintf(stderr, "Unrecognized option\n");
				return -1;
		};
	}
	fprintf(stderr, "Margin = %f, ZMargin = %f\n", margin, zmargin);

	igl::readOBJ("/dev/stdin", V, F);
	Eigen::VectorXd maxV = V.colwise().maxCoeff();
	Eigen::VectorXd minV = V.colwise().minCoeff();

	double zmlow = minV(2) + zmargin*1;
	double zmhigh = minV(2) + zmargin*3;
	double bottomcover[3][2] = 
		{
			{minV(0) - margin, maxV(0) + margin},
			{minV(1) - margin, maxV(1) + margin},
			{zmlow, zmhigh}
	        };
	zmlow = maxV(2) - zmargin*3;
	zmhigh = maxV(2) - zmargin*1;
	double topcover[3][2] = 
		{
			{minV(0) - margin, maxV(0) + margin},
			{minV(1) - margin, maxV(1) + margin},
			{zmlow, zmhigh}
	        };

	Eigen::MatrixXd BV(8, 3);
	Eigen::ArrayXXi BF(12, 3);
	set_coords(BV, bottomcover);
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

	Eigen::MatrixXd VC; // V with Cover
	Eigen::MatrixXi FC; // F with Cover
	mesh_bool(BV, BF, V, F, igl::MESH_BOOLEAN_TYPE_UNION, VC, FC);

	V.noalias() = VC;
	F.noalias() = FC;
	set_coords(BV, topcover);
	mesh_bool(BV, BF, V, F, igl::MESH_BOOLEAN_TYPE_UNION, VC, FC);

	std::cout << VC.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << (FC.array()+1).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"));

	return 0;
}
