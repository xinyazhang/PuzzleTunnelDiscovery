#include "options.h"
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <sstream>
#include <getopt.h>

std::stringstream ss("1\n0 0\n3 0\n1\n0 0\n0 1\n");

Options::Options(int argc, char* argv[])
{
}

std::istream& Options::get_input_stream()
{
#if 1
	return std::cin;
#else
	return ss;
#endif
}

void Options::write_geo(const std::string& suffix,
		       const Eigen::MatrixXd& V,
		       const Eigen::MatrixXi& F)
{
	igl::writeOBJ("obs-"+suffix+".obj", V, F);
	igl::writePLY("obs-"+suffix+".ply", V, F);
}
