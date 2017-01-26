#include "geo.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>
#include <tetio/readtet.h>
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

void Geo::readtet(const std::string& prefix)
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi P;
	::readtet(prefix, V, P);
	Eigen::MatrixXi tF;
	tF.resize(4,3);
	tF << 
	1, 2, 3,
	0, 3, 2,
	3, 0, 1,
	0, 2, 1;

	for (int i = 0; i < P.rows(); i++) {
		Eigen::MatrixXd tV(P.cols(), V.cols());
		for (int j = 0; j < P.cols(); j++) {
			tV.row(j) = V.row(P(i,j));
		}
		cvxV.emplace_back(tV);
		cvxF.push_back(tF);
	}
}
