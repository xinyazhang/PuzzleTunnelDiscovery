#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <string>
#include <set>
#include <memory>
#include <unsupported/Eigen/SparseExtra>
#include <boost/progress.hpp>
//#include <Eigen/SparseLU> 
//#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>

// tetio is our custom library
#include <tetio/readtet.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> -l <Laplacian matrix> [List of vertices to print]" << endl;
	std::cerr << "\t When list of vertices is given, these vertices will be print even if they are not internal vertices." << endl;
}

enum BOUNDARY_CONDITION {
	BC_NONE,
	BC_DIRICHLET,
	BC_NEUMANN, // FIXME: add support for Neumann BC
};

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int opt;
	string igf, lmf;
	while ((opt = getopt(argc, argv, "i:l:")) != -1) {
		switch (opt) {
			case 'i':
				igf = optarg;
				break;
			case 'l':
				lmf = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	std::vector<int> probelist;
	for (int i = optind; i < argc; i++)
		probelist.emplace_back(atoi(argv[i]));
	// Laplacian
	if (lmf.empty()) {
		std::cerr << "Missing Laplacian matrix file (-l)" << endl;
		usage();
		return -1;
	}

	if (igf.empty()) {
		std::cerr << "Missing Geometry from tetgen output (-i)" << endl;
		usage();
		return -1;
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi F;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM, FBM;
	try {
		Eigen::SparseMatrix<double, Eigen::RowMajor> lap;
		if (!Eigen::loadMarket(lap, lmf)) {
			std::cerr << "Failed to load Laplacian matrix from file: " << lmf << endl;
			return -1;
		}
		readtet(igf, V, E, P, &EBM);
		readtet_face(igf, F, &FBM);
		Eigen::VectorXd XV(V.rows()), YV(V.rows()), ZV(V.rows());
		Eigen::VectorXi VBM;
		VBM.setZero(V.rows());
		if (probelist.empty()) {
			for(int i = 0; i < FBM.size(); i++) {
				if (FBM(i) == 0)
					continue;
				for(int j = 0; j < F.cols(); j++) {
					VBM(F(i, j)) = 1;
				}
			}
#if 0
			for(int i = 0; i < VBM.size(); i++) {
				if (VBM(i) > 0)
					continue;
				XV(i) = V(i, 0);
				YV(i) = V(i, 1);
				ZV(i) = V(i, 2);
			}
#else
#endif
		} else {
			for(int i = 0; i < V.rows(); i++)
				VBM(i) = 1;
			for(auto pi : probelist)
				VBM(pi) = 0;
		}
		XV = V.col(0);
		YV = V.col(1);
		ZV = V.col(2);

		XV = lap * XV;
		YV = lap * YV;
		ZV = lap * ZV;
		double Xsum = 0, Ysum = 0, Zsum = 0;
		std::cerr << VBM << endl;
		for(int i = 0; i < VBM.size(); i++) {
			Xsum += XV(i);
			Ysum += YV(i);
			Zsum += ZV(i);
			if (VBM(i) > 0)
				continue;
			if (fabs(XV(i)) > 1e-6)
				std::cerr << "Lx <> 0 at node " << i << " value: " << XV(i) << endl;
			else
				std::cout << "Lx == 0 at node " << i << endl;
			if (fabs(YV(i)) > 1e-6)
				std::cerr << "Ly <> 0 at node " << i << " value: " << YV(i) << endl;
			else
				std::cout << "Ly == 0 at node " << i << endl;
			if (fabs(ZV(i)) > 1e-6)
				std::cerr << "Lz <> 0 at node " << i << " value: " << ZV(i) << endl;
			else
				std::cout << "Lz == 0 at node " << i << endl;
#if 0
			double ttl = XV(i) + YV(i) + ZV(i);
			if (fabs(ttl) > 1e-6)
				std::cerr << "L(x+y+z) > 0 at node " << i << " value: " << ttl << endl;
			else
				std::cout << "L(x+y+z) == 0 at node " << i << endl;
#endif
		}
		std::cerr << "Xsum " << Xsum << endl
			<< "Ysum " << Ysum << endl
			<< "Zsum " << Zsum << endl;
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
