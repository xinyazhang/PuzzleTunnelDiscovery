#include <unistd.h>
#include <iostream>
#include <igl/volume.h>
#include <Eigen/Core>
#include <stdexcept>
#include <tetio/readtet.h>

using std::endl;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> [-o <output file name>]" << endl;
}

int main(int argc, char* argv[])
{
	std::string iprefix, ofn; // Input/Output Geometry Filename
	int opt;
	while ((opt = getopt(argc, argv, "i:o:")) != -1) {
		switch (opt) {
			case 'i':
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::MatrixXi F;
	Eigen::VectorXi EBM, FBM;
	try {
		readtet(iprefix, V, E, P, &EBM);
		Eigen::VectorXd tetvolumes;
		igl::volume(V, P, tetvolumes);
		Eigen::VectorXd massvector;
		massvector.setZero(P.rows());
		for (int i = 0; i < P.rows(); i++) {
			double per_vert_mass = tetvolumes(i) / 4;
			for (int j = 0; j < P.cols(); j++)
				massvector(P(i,j)) += per_vert_mass;
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
