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

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr <<
R"xxx(
Required Options:
	-0 file: specify initial boundary condition
	-l file: specify Laplacian matrix
Optional options:
	-o file: output file
	-t number: end time
	-d number: time delta
	-a number: thermal conductivity factor
	-b: enable binary output
	-v: enable SPD check
	-D: use initial boundary condition as Dirichlet condition
	-N file: Neumann boundary condition file, aka Heat Source Vector File
)xxx";
}

void simulate(std::ostream& fout,
	      const Eigen::SparseMatrix<double, Eigen::RowMajor>& lap,
	      const Eigen::VectorXd& IV,
	      const Eigen::VectorXd& HSV,
	      double alpha,
	      double delta_t,
	      double end_t,
	      bool binary,
	      double snapshot_interval,
	      bool check_spd = false
	      )
{
	Eigen::VectorXd VF = IV;

	fout.precision(17);
	if (binary) {
		char zero[] = "\0\n";
		fout.write(zero, 2);
	} else {
		fout << "#\n";
	}
	Eigen::SparseMatrix<double, Eigen::RowMajor> factor;
	factor.resize(lap.rows(), lap.cols());
	factor.setIdentity();
	factor -= (alpha * delta_t) * lap;
	//Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
	solver.compute(factor);

	Eigen::MatrixXd VPair(IV.rows(), 2);
	VPair.block(0, 1, IV.rows(), 1) = IV;

	boost::progress_display prog(end_t / delta_t);
	for(double tnow = 0.0, last_snapshot = tnow; tnow < end_t; tnow += delta_t) {
#if 0
		Eigen::VectorXd nextVF(VF.rows());
		for(int i = 0; i < V.rows(); i++) {
			nextVF(i) = lap.row(i).dot(VF);
		}
#endif
		if (tnow - last_snapshot >= snapshot_interval) {
			if (!binary) {
				fout << "t: " << tnow << "\t" << VF.rows() << endl;
				fout << VF << endl;
				fout << "sum: " << VF.sum() << endl;
			} else {
				fout.write((const char*)&tnow, sizeof(tnow));
				uint32_t nrow = VF.rows();
				fout.write((const char*)&nrow, sizeof(nrow));
				fout.write((const char*)VF.data(), VF.size() * sizeof(double));
				double sum = VF.sum();
				fout.write((const char*)&sum, sizeof(sum));
			}
			last_snapshot += snapshot_interval;
		}
#if 0
		Eigen::VectorXd delta = lap * VF;
		if (check_spd && VF.dot(delta) < 0) {
			std::cerr << "DLap Matrix is NOT SPD" << endl;
		}
		VF += delta;
#else
		VF += HSV;
		VPair.block(0, 0, IV.rows(), 1) = solver.solve(VF);
		VF = VPair.rowwise().maxCoeff();
#endif
		++prog;
	}
}

enum BOUNDARY_CONDITION {
	BC_NONE = 0,
	BC_DIRICHLET = 1,
	BC_NEUMANN = 2,
};

// FIXME: refactor this piece of mess
int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int opt;
	string ofn, lmf, ivf, nbcvfn;
	double end_t = 10.0, delta_t = 0.1, alpha = 1;
	double snapshot_interval = -1.0;
	bool binary = false;
	bool check_spd = false;
	int bc = BC_NONE;
	while ((opt = getopt(argc, argv, "0:o:t:d:a:l:bDs:vN:")) != -1) {
		switch (opt) {
			case 'o':
				ofn = optarg;
				break;
			case '0':
				ivf = optarg;
				break;
			case 'l':
				lmf = optarg;
				break;
			case 't': end_t = atof(optarg);
				break;
			case 'd':
				delta_t = atof(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			case 'b':
				binary = true;
				break;
			case 'D':
				bc |= BC_DIRICHLET;
				break;
			case 'N':
				bc |= BC_NEUMANN;
				nbcvfn = optarg; // Neumann Boundary Condition Vector File Name
				break;
			case 's':
				snapshot_interval = atof(optarg);
				break;
			case 'v':
				check_spd = true;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (snapshot_interval < 0)
		snapshot_interval = delta_t;
	Eigen::VectorXd F; // F means 'field' not 'faces'
	if (ivf.empty()) {
		std::cerr << "Missing boundary condition file" << endl;
		usage();
		return -1;
	} else {
		std::ifstream fin(ivf);
		if (!fin.is_open()) {
			std::cerr << "Cannot open file: " << ivf << endl;
			return -1;
		}
		int nnode;
		fin >> nnode;
		F.resize(nnode);
		for(int i = 0; i < nnode; i++) {
			double v;
			fin >> v;
			F(i) = v;
		}
	}
	if (lmf.empty()) {
		std::cerr << "Missing Laplacian matrix file" << endl;
		return -1;
	}
	Eigen::SparseMatrix<double, Eigen::RowMajor> lap;
	if (!Eigen::loadMarket(lap, lmf)) {
		std::cerr << "Failed to load Laplacian matrix from file: " << lmf << endl;
		return -1;
	}
	// Fix dlap matrix for Dirichlet condition
	if (bc & BC_DIRICHLET) {
#if 0 // Don't do this. It makes the matrix non-symmetric.
		std::set<int> to_prune;
		for(int i = 0; i < F.size(); i++) {
			if (F(i) != 0) {
				to_prune.emplace(i);
			}
		}
		lap.prune([&to_prune](const int& row, const int&, const int&) -> bool
				{
					if (to_prune.find(row) == to_prune.end())
						return true; // Keep
					return false;
				}
			 );
#endif
	}

	Eigen::VectorXd HSV; // Heat Supply Vector
	if (bc & BC_NEUMANN) {
		std::ifstream fin(nbcvfn);
		if (!fin.is_open()) {
			std::cerr << "Cannot open file: " << ivf << endl;
			return -1;
		}
		int nnode;
		fin >> nnode;
		HSV.resize(nnode);
		for(int i = 0; i < nnode; i++) {
			double v;
			fin >> v;
			HSV(i) = v;
		}
	} else {
		HSV.setZero(F.size()); // No heat source
	}

	std::unique_ptr<std::ostream> pfout_guard;
	std::ostream* pfout;
	if (ofn.empty()) {
		std::cerr << "Missing output file name, output results to stdout instead." << endl;
		if (binary && isatty(fileno(stdout)))
			std::cerr << "Binary output format is disabled for stdout" << endl;
		binary = false;
		pfout = &std::cout;
	} else {
		pfout_guard.reset(new std::ofstream(ofn));
		pfout = pfout_guard.get();
	}
	simulate(*pfout, lap, F, HSV, alpha, delta_t, end_t, binary, snapshot_interval, check_spd);
	pfout_guard.reset();

	return 0;
}
