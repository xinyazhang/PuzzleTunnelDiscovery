#include "levelset.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <fat/fat.h>

using std::string;
using std::endl;

void usage()
{
	std::cerr << "Options: -i <input OBJ file> -o <output OBJ file> -w <fattening width> -s <scale factor>" << endl;
	std::cerr << "\tscale factor: default to " << fat::default_scale_factor << ", increase this for finer mesh" << endl;
}

int main(int argc, char* argv[])
{
	int opt;
	string ifn, ofn;
	double fatten = 1.0;
	double scale = fat::default_scale_factor;
	bool trianglize = true;
	while ((opt = getopt(argc, argv, "i:o:w:s:q")) != -1) {
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
			case 's':
				scale = atof(optarg);
				break;
			case 'q':
				trianglize = false;
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

	fat::initialize();

	Eigen::MatrixXf IV, OV;
	Eigen::MatrixXi IF, OF;
	if (!igl::readOBJ(ifn, IV, IF)) {
		std::cerr << "Fail to read " << argv[1] << " as OBJ file" << std::endl;
		return -1;
	}
	fat::mkfatter(IV, IF, fatten, OV, OF, trianglize, scale);
	igl::writeOBJ(ofn, OV, OF);
	return 0;
}
