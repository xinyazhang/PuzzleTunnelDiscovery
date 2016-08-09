#include <readheat.h>
#include <readtet.h>
#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>
#include <memory>
#include <unsupported/Eigen/SparseExtra>
#include <boost/progress.hpp>
//#include <Eigen/SparseLU> 
//#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> -t <temperature file> -b <boundary vertex file> [-o <output path file> -f time_frame] <x y z>" << endl;
}

const int
proto_face_number[4][3] = {
	{1, 3, 2},
	{0, 2, 3},
	{3, 1, 0},
	{0, 1, 2}
};

bool
in_tet(const Eigen::Vector3d& start_point,
       const Eigen::MatrixXd& T)
{
	for (int i = 0; i < 4; i++) {
		int v0i = proto_face_number[i][0];
		int v1i = proto_face_number[i][1];
		int v2i = proto_face_number[i][2];
		Eigen::Vector3d v0 = T.col(v0i);
		Eigen::Vector3d v1 = T.col(v1i);
		Eigen::Vector3d v2 = T.col(v2i);
		Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
		if (n.dot(start_point - v0) > 0)
			return false;
	}
	return true;
}

int
find_tet(const Eigen::Vector3d& start_point,
	 const Eigen::MatrixXd& V,
	 const Eigen::MatrixXi& P)
{
	Eigen::MatrixXd tet;
	tet.resize(3, 4);
	for (int i = 0; i < P.rows(); i++) {
		for (int j = 0; j < P.cols(); j++) {
			tet.block<3, 1>(0, j) = V.row(P(i,j));
		}
		if (in_tet(start_point, tet))
			return i;
	}
	return -1;
}

void
follow(const Eigen::Vector3d& start_point,
       const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& E,
       const Eigen::MatrixXi& P,
       const Eigen::VectorXd& MBM,
       const Eigen::VectorXd& H,
       std::ostream& fout)
{
	fout.precision(17);
	int tet_id = find_tet(start_point, V, P);
	if (tet_id < 0)
		throw std::runtime_error("Start point isn't in any tetrahedron.");
	int next_vert = P(tet_id, 0);
	double max_temp = V(next_vert);
	for (int j = 1; j < P.cols(); j++) {
		int vert = P(tet_id, j);
		if (V(vert) > max_temp) {
			max_temp = V(vert);
			next_vert = vert;
		}
	}
	std::unordered_map<int, std::vector<int>> neigh;
	for (int i = 0; i < E.rows(); i++) {
		int ei = E(i, 0);
		int ej = E(i, 1);
		neigh[ei].emplace_back(ej);
		neigh[ej].emplace_back(ei);
	}
	while (MBM(next_vert) == 0.0) {
		fout << V.row(next_vert) << "\t" << next_vert << endl;
		const auto& nei = neigh[next_vert];
		next_vert = nei.front();
		max_temp = V(next_vert);
		for (int vert : nei) {
			if (V(vert) > max_temp) {
				max_temp = V(vert);
				next_vert = vert;
			}
		}
	}
	fout << V.row(next_vert) << "\t" << next_vert << endl;
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();
	int opt;
	string iprefix, tfn, ofn, ivf;
	int frame_to_pick = INT_MAX;
	while ((opt = getopt(argc, argv, "i:t:o:f:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 't':
				tfn = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 'b':
				ivf = optarg;
				break;
			case 'f':
				frame_to_pick = atoi(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	Eigen::Vector3d start_point;
	{
		int i;
		for (i = optind; i < argc && i < optind + 3; i++) {
			start_point(i - optind) = atof(argv[i]);
		}
		if (i < optind + 3) {
			std::cerr << "<x y z> is mandantory" << endl;
			usage();
			return -1;
		}
	}

	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	if (tfn.empty()) {
		std::cerr << "Missing temperature file" << endl;
		usage();
		return -1;
	}

	if (ivf.empty()) {
		std::cerr << "Missing boundary condition file" << endl;
		usage();
		return -1;
	} else {
	}

	std::ostream* os;
	if (ofn.empty()) {
		std::cerr << "Missing output file name" << endl
			  << "Printing to stdout" << endl;
		os = &std::cout;
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	Eigen::VectorXd MBM; // Maze boundary marker;
	try {
		readtet(iprefix, V, E, P, &EBM);
		{
			std::ifstream fin(ivf);
			if (!fin.is_open())
				throw std::runtime_error("Cannot open file: " + ivf);
			int nnode;
			fin >> nnode;
			MBM.resize(nnode);
			for(int i = 0; i < nnode; i++) {
				double v;
				fin >> v;
				MBM(i) = v;
			}
		}

		std::ifstream tf(tfn);
		HeatReader hreader(tf);
		HeatFrame hframe;
		int frameid = 0;
		while (frameid > frame_to_pick && hreader.read_frame(hframe))
			frameid++;
		std::ofstream fout;
		if (!ofn.empty()) {
			fout.open(ofn);
			if (!fout.is_open())
				throw std::runtime_error("Cannot open " + ofn + " for write.");
			os = &fout;
		}
		follow(start_point, V, E, P, MBM, hframe.hvec, *os);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
