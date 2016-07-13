#include "readtet.h"
#include <unistd.h>
#include <strings.h>
#include <string>
#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> [-o output_file]" << endl;
}

const int kCoords[][2] = {
	{0, -1}, // X-
	{0,  1}, // X+
	{1, -1}, // Y-
	{1,  1}, // Y+
	{2, -1}, // Z-
	{2,  1}, // Z+
};

const char* kCoordsName[] = {
	"X-",
        "X+",
        "Y-",
        "Y+",
        "Z-",
        "Z+"
};

inline bool fpclose(double f0, double f1)
{
	if (std::abs(f0 - f1) < 1e-4)
		return true;
	return false;
}

void set_IV(Eigen::VectorXd& iv,
	    const Eigen::MatrixXd& V,
	    int coord,
	    double target_value,
	    double bv0
	    )
{
	for (int i = 0; i < V.rows(); i++) {
		double coord_value = V(i, coord);
		if (fpclose(target_value, coord_value))
			iv(i) = bv0;
	}
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn;
	double bv0 = 1.0;
	while ((opt = getopt(argc, argv, "i:o:0:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case '0':
				bv0 = atof(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	bool boundary_enabled[6] = {false, false, false, false, false, false};
	for(int i = optind; i < argc; i++) {
		for(int k = 0; k < 6; k++) {
			if (strcasecmp(kCoordsName[k], argv[i]) == 0) {
				boundary_enabled[k] = true;
			}
		}
	}

	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	if (ofn.empty()) {
		ofn = iprefix + ".Bcond";
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::MatrixXd IV;
	try {
		readtet(V, E, P, iprefix);
		Eigen::VectorXd IV;
		IV.setZero(V.rows());
		Eigen::VectorXd minV, maxV;
		minV = V.colwise().minCoeff();
		maxV = V.colwise().maxCoeff();
		for(int k = 0; k < 6; k++) {
			if (!boundary_enabled[k])
				continue;
			int coord = kCoords[k][0];
			set_IV(IV,
				V,
				coord,
				kCoords[k][1] > 0 ? maxV(coord) : minV(coord),
				bv0
			      );
		}
		std::ofstream fout;
		fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
		fout.open(ofn);
		fout << IV.rows() << endl << IV << endl;
		fout.close();
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
