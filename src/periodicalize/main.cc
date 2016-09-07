#include <string>
#include <iostream>
#include <math.h>
#include <readtet.h>
#include <geopick/pick2d.h>

using std::string;
using std::endl;
using std::cerr;

void usage()
{
	std::cerr << 
R"xxx(This program generate the periodical part geometry from the obstacle space geometry
Options: -i <prefix> [-o file]
	-i prefix: input tetgen prefix" << endl
	-o file: output the result to file instead of stdout
)xxx";
}

void slice(const Eigen::MatrixXd& V,
           const Eigen::MatrixXi& E,
           const Eigen::MatrixXi& P,
           const Eigen::VectorXi& sliceV,
           const Eigen::MatrixXi& sliceF,
           double z)
{
}

void glue_boundary(const Eigen::MatrixXd& V,
                   const Eigen::VectorXi& btmV,
                   const Eigen::MatrixXi& btmF,
                   const Eigen::VectorXi& topV,
                   const Eigen::MatrixXi& topF,
                   const Eigen::MatrixXi& glueF)
{
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn;
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
	Eigen::VectorXi EBM;
	try {
		readtet(iprefix, V, E, P, &EBM);
		Eigen::VectorXi btmVI, topVI;
		Eigen::MatrixXi btmF, topF, glueF;
		slice(V, E, P, btmVI, btmF, 0);
		slice(V, E, P, btmVI, btmF, 2 * M_PI);
		glue_boundary(V, btmVI, btmF, topVI, topF, glueF);
		Eigen::MatrixXd prdcV; // PeRioDiCal Vertices
		Eigen::MatrixXi prdcF; // PeRioDiCal Faces
		geopick(V, {btmF, topF, glueF}, prdcV, prdcF);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
