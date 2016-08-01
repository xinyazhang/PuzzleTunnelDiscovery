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
#include <readtet.h>
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
	std::cerr << "Options: -i <tetgen file prefix> -0 <boundary condition file> -l <Laplacian matrix>" << endl;
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
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	try {
		Eigen::SparseMatrix<double, Eigen::RowMajor> lap;
		if (!Eigen::loadMarket(lap, lmf)) {
			std::cerr << "Failed to load Laplacian matrix from file: " << lmf << endl;
			return -1;
		}
		readtet(igf, V, E, P, &EBM);
		Eigen::VectorXi VBM;
		VBM.setZero(V.rows());
		for(int i = 0; i < EBM.size(); i++) {
			if (EBM(i) == 0)
				continue;
			int v0 = E(i, 0);
			int v1 = E(i, 1);
			VBM(v0) = 1;
			VBM(v1) = 1;
		}
		Eigen::VectorXd XV(V.rows()), YV(V.rows()), ZV(V.rows());
#if 0
		for(int i = 0; i < VBM.size(); i++) {
			if (VBM(i) > 0)
				continue;
			XV(i) = V(i, 0);
			YV(i) = V(i, 1);
			ZV(i) = V(i, 2);
		}
#endif
		XV = lap * V.col(0);
		YV = lap * V.col(1);
		ZV = lap * V.col(2);
		for(int i = 0; i < VBM.size(); i++) {
			if (VBM(i) > 0)
				continue;
			if (fabs(XV(i)) > 1e-6)
				std::cerr << "Lx > 0 at node " << i << " value: " << XV(i) << endl;
			else
				std::cout << "Lx == 0 at node " << i << endl;
			if (fabs(YV(i)) > 1e-6)
				std::cerr << "Ly > 0 at node " << i << " value: " << YV(i) << endl;
			else
				std::cout << "Ly == 0 at node " << i << endl;
			if (fabs(ZV(i)) > 1e-6)
				std::cerr << "Lz > 0 at node " << i << " value: " << ZV(i) << endl;
			else
				std::cout << "Lz == 0 at node " << i << endl;
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
