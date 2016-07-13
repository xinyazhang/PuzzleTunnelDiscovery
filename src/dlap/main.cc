#include "readtet.h"
#include "tet2lap.h"
#include <unistd.h>
#include <string>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

bool has_suffix(const string &str, const string &suffix)
{
	return str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string replace_suffix(const string &str,
			   const string &suffix,
			   const string &newsuffix)
{
	if (!has_suffix(str, suffix))
		return std::string();
	std::string ret(str);
	ret.replace(str.size() - suffix.size(), suffix.size(), newsuffix);
	return ret;
}

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> [-o output_file]" << endl;
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
	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	if (ofn.empty()) {
		ofn = iprefix + ".mat";
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::SparseMatrix<double> lap;
	try {
		readtet(V, E, P, iprefix, ofn);
		tet2lap(lap, V, E, P);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	Eigen::saveMarket(lap, ofn);
	return 0;
}
