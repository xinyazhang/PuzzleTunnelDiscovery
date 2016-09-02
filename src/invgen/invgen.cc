#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

using std::endl;

void usage()
{
	std::cerr << "Options: [-r <radius> -w <width> -d <delta theta> -s <start theta> -f <final theta>" << endl;
}

void print_circular(std::ostream& fout, const std::vector<Eigen::Vector2d>& vertlist)
{
	Eigen::Vector2d prev = vertlist.back();
	fout << vertlist.size() << endl;
	for(const auto& vert : vertlist) {
		fout << prev.transpose() << "\t" << vert.transpose() << endl;
		prev = vert;
	}
}

int main(int argc, char* argv[])
{
	double radius = 1.0;
	double width = 0.8;
	double dtheta = M_PI / 32;
	double theta_start = M_PI;
	double theta_final = 6 * M_PI;
	int opt;
	while ((opt = getopt(argc, argv, "r:w:d:s:f:")) != -1) {
		switch (opt) {
			case 'r':
				radius = atof(optarg);
				break;
			case 'w':
				width = atof(optarg);
				break;
			case 'd':
				dtheta = atof(optarg);
				break;
			case 's':
				theta_start = atof(optarg);
				break;
			case 'f':
				theta_final = atof(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	double theta = theta_start;
	std::vector<Eigen::Vector2d> inner, outer;
	while (theta < theta_final) {
		Eigen::Vector2d middle(cos(theta)+theta*sin(theta), sin(theta)-theta*cos(theta));
		//std::cerr << middle.transpose() << endl;
		middle *= radius;
		Eigen::Vector2d span = middle.normalized() * width;
		inner.emplace_back(middle - span);
		outer.emplace_back(middle + span);
		theta += dtheta;
	}
	inner.insert(inner.end(), outer.rbegin(), outer.rend());
	//outer.insert(outer.end(), inner.rbegin(), inner.rend());
	std::cout.precision(17);
	print_circular(std::cout, inner);

	return 0;
}
