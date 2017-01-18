#include "geo.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>
// #include <iostream>

using std::string;

void Geo::read(const std::string& fn)
{
	igl::readOBJ(fn, V, F);
	//center << 0.0, 0.0, 0.0; // Origin
	center = V.colwise().mean().cast<double>();
	//center << 16.973146438598633, 1.2278236150741577, 10.204807281494141;
	// From OMPL.app, no idea how they get this.
	GPUV = V.cast<float>();
	igl::per_vertex_normals(GPUV, F, N);
	std::cerr << "center: " << center << std::endl;
#if 0
	std::cerr << N << std::endl;;
#endif
}

void Geo::readcvx(const std::string& prefix)
{
	int p = 0;
	while (true) {
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		string fn;
		fn = prefix + ".cvx." + std::to_string(p) + ".obj";
		bool success = igl::readOBJ(fn, V, F);
		if (!success)
			break;
		cvxV.emplace_back(std::move(V));
		cvxF.emplace_back(std::move(F));
		p++;
	}
}
