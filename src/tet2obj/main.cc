/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <iostream>
#include <string>
#include <unistd.h>
#include <igl/writeOBJ.h>

#include <geopick/pick2d.h>
#include <tetio/readtet.h>

using std::endl;
using std::string;

void usage()
{
	std::cerr <<
R"zzz(This program translate tetgen output files into an Wavefront OBJ file
Usage: -i <prefix> -o <file>
	-i prefix: tet file prefix
	-o file: OBJ file
NOTE: This program only prints faces present in the .faces file, usually it only contains boundary faces.
)zzz";
}

int main(int argc, char* argv[])
{
	string iprefix, ofn;
	int opt;
	while ((opt = getopt(argc, argv, "i:o:h")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 'h':
				usage();
				return 0;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (iprefix.empty() || ofn.empty()) {
		std::cerr << "Missing options" << std::endl;
		usage();
		return -1;
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi F;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	Eigen::VectorXi FBM;
	try {
		readtet(iprefix, V, E, P, &EBM);
		readtet_face(iprefix, F, &FBM);
		Eigen::MatrixXd outV;
		Eigen::MatrixXi outF;
		geopick(V, {F}, outV, outF);
		igl::writeOBJ(ofn, outV, outF);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
}
