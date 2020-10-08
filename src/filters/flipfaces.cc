/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/readOBJ.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char* argv[])
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::readOBJ("/dev/stdin", V, F);
	F.col(0).swap(F.col(1)); // swap column
	std::cout << V.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << (F.array()+1).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"));
	return 0;
}
