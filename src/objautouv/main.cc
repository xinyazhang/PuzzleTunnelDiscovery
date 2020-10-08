/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "mesh.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edge_lengths.h>
#include <igl/per_face_normals.h>
#include <iostream>
#include <fstream>
#include <getopt.h>

namespace {
int res = 1024;;
double boxw = -1;
double boxh = -1;
int margin = 0;
};

void usage()
{
	std::cerr << R"xxx(Usage objautouv [OPTIONS] <INPUT OBJ> <OUTPUT OBJ>
Options:
	-r <resolution> Default: 1024
	-w <box width>  Default: -1 (probing)
	-h <box height> Default: -1 (probing)
	-m <margin>     Unit: pixel. Requires -r -w and -h. Default: 0
	-f              Overwrite the output obj.
	-n              Do not pick optimal box to place in each iteration.
                        This makes large mesh packed faster.
)xxx";
	std::cerr << "\tDefault Resolution: " << res << "\n";
}

int main(int argc, char* argv[])
{
	int opt;
	bool overwrite = false;
	bool pair = true;
	bool optimized = true;
	while ((opt = getopt(argc, argv, "r:w:h:m:fsn")) != -1) {
		switch (opt) {
			case 'r':
				res = std::atoi(optarg);
				break;
			case 'w':
				boxw = std::atof(optarg);
				break;
			case 'h':
				boxh = std::atof(optarg);
				break;
			case 'm':
				margin = std::atoi(optarg);
				break;
			case 'f':
				overwrite = true;
				break;
			case 's': // -s == single
				pair = false;
				break;
			case 'n':
				optimized = false;
				break;
			default:
				std::cerr << "Unrecognized argument -" << char(opt) << std::endl;
				usage();
				return -1;
		}
	}
	if (argc < optind + 2) {
		usage();
		return -1;
	}
	if (boxw * boxh <= 0) {
		throw std::runtime_error("boxw and boxh must have the same sign");
	}
	std::cerr << "Configuration res: " << res << "\tboxw: " << boxw << "\tboxh: " << boxh << std::endl;
	std::string ifn(argv[optind]), ofn(argv[optind + 1]);
	if (!overwrite) {
		std::ifstream infile(ofn);
		if (infile.good()) {
			std::cerr << ofn << " already existed\n";
			return -1;
		}
	}
	Mesh m;
	Eigen::MatrixXd tmpV;
	igl::readOBJ(ifn, tmpV, m.UV, m.N, m.F, m.FUV, m.FN);
	//igl::readOBJ(ifn, tmpV, m.F);
	m.V = tmpV.block(0, 0, tmpV.rows(), 3); // Trims off extra columns (not sure why meshlab write 6 numbers for obj files)
#if 0
	igl::writeOBJ(ofn, m.V, m.F);
	return 0;
#endif
	if (m.UV.rows() > 0) {
		std::cerr << ifn << " already consists of UV coordinates\n";
		igl::writeOBJ(ofn, m.V, m.F, m.N, m.FN, m.UV, m.FUV);
		return 0;
	}
	if (m.N.rows() > m.V.rows()) {
		std::cerr << ifn << " has multi-valued vertex normal\n";
		return -1;
	}
	igl::edge_lengths(m.V, m.F, m.el);
	// std::cerr << "m.F\n" << m.F << std::endl;
	igl::per_face_normals(m.V, m.F, m.face_normals);
	m.PairWithLongEdge(pair);
	m.Program(res, boxw, boxh, margin, optimized);
	igl::writeOBJ(ofn, m.V, m.F, m.N, m.FN, m.UV, m.FUV);
	std::ofstream fout("out.svg");
	fout << "<svg height=\"" << res << "\" width=\"" << res << "\">\n";
	for (int i = 0; i < m.FUV.rows(); i++) {
		double u0 = m.UV(m.FUV(i,0), 0) * res;
		double v0 = m.UV(m.FUV(i,0), 1) * res;
		double u1 = m.UV(m.FUV(i,1), 0) * res;
		double v1 = m.UV(m.FUV(i,1), 1) * res;
		double u2 = m.UV(m.FUV(i,2), 0) * res;
		double v2 = m.UV(m.FUV(i,2), 1) * res;
		fout << R"xxx(<polygon points=")xxx"
		     << u0 <<',' << v0 << " "
		     << u1 <<',' << v1 << " "
		     << u2 <<',' << v2 << " "
		     << R"xxx(" style="fill:lime;stroke:purple;stroke-width:1" />)xxx"
		     << std::endl;
	}
	fout << R"xxx(</svg>)xxx";

	// igl::writeOBJ(ofn, m.V, m.F);
	return 0;
}
