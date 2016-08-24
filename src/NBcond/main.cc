#include "readtet.h"
#include <unistd.h>
#include <omp.h>
#include <strings.h>
#include <string>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <unsupported/Eigen/SparseExtra>
#include <igl/ray_mesh_intersect.h>
#include <igl/readOBJ.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

const double kEpsilon = 1e-6;

void usage()
{
	std::cerr << "Options: -i <file> -g <file> -m <margin> [-o file -0 boundary_value]" << endl;
	std::cerr << "\t-i: tetgen file prefix for the obstacle-free geometry." << endl
		  << "\t-g: obstacle geometry file" << endl
		  << "\t-m: specify the margin used in create the obstacle-free geometry from -g geometry." << endl
		  << "\t-o: output the heat supply vector to file instead of stdout" << endl
		  << "\t-0: Neumann boundary value, default to -1" << endl;
}

bool is_oob(const Eigen::Vector3d& vert, 
		const Eigen::VectorXd& bMax,
		const Eigen::VectorXd& bMin)
{
	if (vert(0) > bMax(0) || vert(0) < bMin(0))
		return true;
	if (vert(1) > bMax(1) || vert(1) < bMin(1))
		return true;
	if (vert(2) > bMax(1) || vert(1) < bMin(1))
		return true;
	return false;
}

void set_boundary_value(const Eigen::VectorXi& Vfree,
		Eigen::VectorXd& IV,
		double bv0)
{
	for (int i = 0; i < Vfree.rows(); i++) {
		if (Vfree(i))
			IV(i) = bv0;
	}
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn, obfn;
	double nv0 = -1.0;
	double margin = -1.0;
	while ((opt = getopt(argc, argv, "i:o:m:0:g:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case '0':
				nv0 = atof(optarg);
				break;
			case 'g':
				obfn = optarg;
				break;
			case 'm':
				margin = atof(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (margin < 0) {
		std::cerr << "Missing -m option, or -m provdes a negative value" << endl;
		usage();
		return -1;
	}

	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	Eigen::MatrixXd ObV;
	Eigen::MatrixXi ObF;
	if (obfn.empty()) {
		std::cerr << "Missing -g option" << endl;
		usage();
		return -1;
	}
	if (!igl::readOBJ(obfn, ObV, ObF)) {
		std::cerr << "Cannot read " << obfn << " as OBJ file" << endl;
		return -1;
	}
	Eigen::VectorXd maxObV = ObV.colwise().maxCoeff();
	Eigen::VectorXd minObV = ObV.colwise().minCoeff();
	Eigen::VectorXd bMax = maxObV.array() + margin / 2;
	Eigen::VectorXd bMin = minObV.array() - margin / 2;
	bMax(2) = maxObV(2) - kEpsilon;
	bMin(2) = minObV(2) - kEpsilon;

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	Eigen::MatrixXd IV;
	try {
		readtet(iprefix, V, E, P, &EBM);
		Eigen::VectorXd IV;
		Eigen::VectorXi isBV;
		IV.setZero(V.rows());
		isBV.setZero(V.rows());

		for (int i = 0; i < E.rows(); i++) {
			int vid0 = E(i, 0);
			int vid1 = E(i, 1);
			Eigen::Vector3d v0 = V.row(vid0);
			Eigen::Vector3d v1 = V.row(vid1);
			if (is_oob(v0, bMax, bMin) || is_oob(v1, bMax, bMin))
				continue;
			isBV(vid0) = 1;
			isBV(vid1) = 1;
		}
		set_boundary_value(isBV, IV, nv0);

		std::ostream* pfout;
		std::ofstream fout;
		if (ofn.empty()) {
			pfout = &std::cout;
		} else {
			fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
			fout.open(ofn);
			pfout = &fout;
		}
		(*pfout) << IV.rows() << endl << IV << endl;
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
