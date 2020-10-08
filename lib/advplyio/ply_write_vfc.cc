/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "ply_write_vfc.h"
#include <fstream>
using std::endl;

/*
ply
format ascii 1.0
element vertex 37536
property double x
property double y
property double z
element face 75076
property list uchar int vertex_indices
end_header
*/

void ply_write_naive_header(std::ostream& fout, size_t vn, size_t fn)
{
	fout << "ply" << endl << "format ascii 1.0" << endl;
	fout << "element vertex " << vn << endl;
	fout << "property double x" << endl;
	fout << "property double y" << endl;
	fout << "property double z" << endl;
	fout << "property uchar red" << endl;
	fout << "property uchar green" << endl;
	fout << "property uchar blue" << endl;
	fout << "element face " << fn << endl;
	fout << "property list uchar int vertex_indices" << endl;
	fout << "end_header" << endl;
}

void ply_write_vfc(const std::string& fn,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& Cfloat)
{
	std::ofstream fout(fn);
	ply_write_naive_header(fout, V.rows(), F.rows());
	fout.precision(17);
	Eigen::MatrixXi C = (Cfloat * 255.0).cast<int>();
	for(int i = 0; i < V.rows(); i++) {
		fout << V(i,0) << ' ' << V(i,1) << ' ' << V(i,2) << ' ';
		fout << C(i,0) << ' ' << C(i,1) << ' ' << C(i,2) << endl;
	}
	for(int i = 0; i < F.rows(); i++) {
		fout << F.cols();
		for(int j = 0; j < F.cols(); j++)
			fout << ' ' << F(i, j);
		fout << endl;
	}
}
