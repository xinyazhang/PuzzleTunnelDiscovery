#include "mesh.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edge_lengths.h>
#include <igl/per_face_normals.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
	int res = 1024;
	if (argc < 3) {
		std::cerr << "Usage objautouv <INPUT OBJ> <OUTPUT OBJ> [Target Resolution]\n";
		std::cerr << "\tDefault Resolution: " << res << "\n";
		return -1;
	}
	if (argc >= 4) {
		res = std::atoi(argv[3]);
		std::cerr << "Change Target Resolution to " << res << std::endl;
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
		return -1;
	}
	if (m.N.rows() > m.V.rows()) {
		std::cerr << ifn << " has multi-valued vertex normal\n";
		return -1;
	}
	igl::edge_lengths(m.V, m.F, m.el);
	// std::cerr << "m.F\n" << m.F << std::endl;
	igl::per_face_normals(m.V, m.F, m.face_normals);
	m.PairWithLongEdge();
	m.Program(res);
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
