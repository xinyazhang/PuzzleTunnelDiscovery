#include "mesh.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edge_lengths.h>
#include <igl/per_face_normals.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
	if (argc < 3) {
		std::cerr << "Usage objautouv <INPUT OBJ> <OUTPUT OBJ>\n";
		return -1;
	}
	std::string ifn(argv[1]), ofn(argv[2]);
	{
		std::ifstream infile(ofn);
		if (infile.good()) {
			std::cerr << ofn << " already existed\n";
			return -1;
		}
	}
	Mesh m;
	igl::readOBJ(ifn, m.V, m.UV, m.N, m.F, m.FUV, m.FN);
	if (m.UV.rows() > 0) {
		std::cerr << ifn << " already consists of UV coordinates\n";
		return -1;
	}
	if (m.N.rows() > m.V.rows()) {
		std::cerr << ifn << " has multi-valued vertex normal\n";
		return -1;
	}
	igl::edge_lengths(m.V, m.F, m.el);
	igl::per_face_normals(m.V, m.F, m.face_normals);
	m.PairWithLongEdge();
	m.Program();
	igl::writeOBJ(ofn, m.V, m.F, m.N, m.FN, m.UV, m.FUV);
	return 0;
}
