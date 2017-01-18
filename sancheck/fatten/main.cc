#include "sancheck.h"
#include <igl/readOBJ.h>
#include <unistd.h>
#include <string>
#include <iostream>

using std::string;
using std::endl;

void usage()
{
	std::cerr << "Options: -i <original OBJ file> -o <fatten OBJ file> -w <fattening width>" << endl;
}

int main(int argc, char* argv[])
{
	int opt;
	string ifn, ofn;
	double fatten = 1.0;
	while ((opt = getopt(argc, argv, "i:o:w:")) != -1) {
		switch (opt) {
			case 'i': 
				ifn = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 'w':
				fatten = atof(optarg);
				break;
			default:
				usage();
				return -1;
		}
	}
	if (ifn.empty() || ofn.empty()) {
		usage();
		return -1;
	}

	Eigen::MatrixXf IV, OV;
	Eigen::MatrixXi IF, OF;
	if (!igl::readOBJ(ifn, IV, IF)) {
		std::cerr << "Fail to read " << ifn << " as OBJ file" << std::endl;
		return -1;
	} else if (!igl::readOBJ(ofn, OV, OF)) {
		std::cerr << "Fail to read " << ofn << " as OBJ file" << std::endl;
		return -1;
	}
	std::cerr << "SAN Check...";
	san_check(IV, IF, OV, OF, fatten);
	return 0;
}
