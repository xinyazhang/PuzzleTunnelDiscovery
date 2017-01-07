#include "levelset.h"
#include <openvdb/openvdb.h>
#include <igl/readOBJ.h>
#include <iostream>

int main(int argc, const char* argv[])
{
	openvdb::initialize();
	Eigen::MatrixXf V;
	Eigen::MatrixXi F;
	if (!igl::readOBJ(argv[1], V, F)) {
		std::cerr << "Fail to read " << argv[1] << " as OBJ file" << std::endl;
		return -1;
	}
	levelset::generate(V, F, 2.0, 4.0/4.0, argv[2]);
	return 0;
}
