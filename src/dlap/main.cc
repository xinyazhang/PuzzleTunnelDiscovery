/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <unistd.h>
#include <string>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

#include <tetio/readtet.h>
#include "tet2lap.h"

using std::string;
using std::endl;
using std::cerr;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> [-o output_file -s scale_factor -U]" << endl;
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn;
	double scale_factor = 1.0;
	bool unit_weight = false;
	while ((opt = getopt(argc, argv, "i:o:s:U")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 's':
				scale_factor = atof(optarg);
				break;
			case 'U':
				unit_weight = true;
				break;
			default:
				usage();
				return -1;
		}
	}
	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	if (ofn.empty()) {
		ofn = iprefix + ".mat";
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	Eigen::MatrixXd VNodes;
#if 0
	std::vector<VoronoiEdge> VEdges;
	std::vector<VoronoiFace> VFaces;
	std::vector<VoronoiCell> VCells;
#endif

	Eigen::SparseMatrix<double> lap;
	try {
		readtet(iprefix, V, E, P, &EBM);
		//readvoronoi(iprefix, VNodes, VEdges, VFaces, VCells);
		//tet2lap(V, E, P, VNodes, VEdges, VFaces, VCells, lap);
		V.block(0, 2, V.rows(), 1) *= scale_factor;
		tet2lap(V, E, P, lap, unit_weight);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	if (V.rows() < 16)
		std::cerr << lap << endl;
	Eigen::saveMarket(lap, ofn);
	return 0;
}
