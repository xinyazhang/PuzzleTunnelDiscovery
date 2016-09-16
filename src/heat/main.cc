#define EIGEN_USE_MKL_ALL
//#undef EIGEN_USE_MKL_ALL

#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <string>
#include <set>
#include <chrono>
#include <memory>
#include <unsupported/Eigen/SparseExtra>
#include <boost/progress.hpp>
//#include <Eigen/SparseLU> 
//#include <Eigen/SparseCholesky>
#ifdef EIGEN_USE_MKL_ALL
#       include <Eigen/PardisoSupport>
#else
#       include <Eigen/CholmodSupport>
//#       include <Eigen/PaStiXSupport>
#endif
#include <vecio/vecin.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr <<
R"xxx(
Usage: heat -0 file -l file [-o file -t number -d number -b -v -D -N file]
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

enum BOUNDARY_CONDITION {
	BC_NONE = 0,
	BC_DIRICHLET = 1,
	BC_NEUMANN = 2,
};

struct Simulator {
	Eigen::SparseMatrix<double, Eigen::RowMajor> lap;
	Eigen::VectorXd IV;
	Eigen::VectorXd HSV;
	double alpha;
	double delta_t;
	double end_t;
	bool binary;
	double snapshot_interval;
	bool check_spd = false;
	Eigen::VectorXd MVec; // Mass matrix
	int bc;

	void calibrate_for_hidden_nodes()
	{
		int nnodes = IV.rows();
		std::cerr << "Calibrate IVs from " << nnodes << " to " << lap.rows() << endl;
		IV.conservativeResize(lap.rows());
		HSV.conservativeResize(lap.rows());
		for (int i = nnodes; i < lap.rows(); i++) {
			IV(i) = 0;
			HSV(i) = 0;
		}
		if (MVec.size() > 0) {
			MVec.conservativeResize(lap.rows());
			for (int i = nnodes; i < lap.rows(); i++)
				MVec(i) = 1;
		}
	}

	void simulate(std::ostream& fout) const
	{
		Eigen::VectorXd VF = IV;

		Eigen::SparseMatrix<double, Eigen::RowMajor> IvM;
		IvM.resize(lap.rows(), lap.cols());
		IvM.setIdentity();
		if (MVec.rows() == lap.rows()) {
			typedef Eigen::Triplet<double> tri_t;
			std::vector<tri_t> tris;
			for(int i = 0; i < MVec.size(); i++)
				tris.emplace_back(i, i, 1/MVec(i));
			IvM.setFromTriplets(tris.begin(), tris.end());
		} else if (MVec.size() != 0) {
			std::cerr << "Mass vector size mismatch";
		}

		fout.precision(17);
		if (binary) {
			char zero[] = "\0\n";
			fout.write(zero, 2);
		} else {
			fout << "#\n";
		}
		write_frame(fout, 0, IV);
		Eigen::SparseMatrix<double, Eigen::RowMajor> factor;
		factor.resize(lap.rows(), lap.cols());
		factor.setIdentity();
		factor -= (alpha * delta_t) * lap;
#ifdef EIGEN_USE_MKL_ALL
		Eigen::PardisoLU<decltype(factor)> solver;
#else
		//Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
		//Eigen::PastixLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Lower> solver;
#endif
		solver.compute(factor);

		Eigen::MatrixXd VPair(IV.rows(), 2);
		VPair.block(0, 1, IV.rows(), 1) = IV;

		boost::progress_display prog(end_t / delta_t);
		auto start_point = std::chrono::system_clock::now();
		for(double tnow = 0.0, last_snapshot = tnow; tnow < end_t; tnow += delta_t) {
#if 0
			Eigen::VectorXd nextVF(VF.rows());
			for(int i = 0; i < V.rows(); i++) {
				nextVF(i) = lap.row(i).dot(VF);
			}
#endif
			if (tnow - last_snapshot >= snapshot_interval) {
				write_frame(fout, tnow, VF);
				last_snapshot += snapshot_interval;
			}
#if 0
			Eigen::VectorXd delta = lap * VF;
			if (check_spd && VF.dot(delta) < 0) {
				std::cerr << "DLap Matrix is NOT SPD" << endl;
			}
			VF += delta;
#else
			VF += HSV * 0.0000125 * delta_t; // Apply HSV
			Eigen::VectorXd VFNext = solver.solve(IvM * VF);
			// The New Dirichlet Cond
			if (bc & BC_DIRICHLET) {
#pragma omp parallel for
				for(int i = 0; i < IV.rows(); i++) {
					if (IV(i) != 0)
						VFNext(i) = IV(i);
				}
			}
			VF.swap(VFNext);
			//VF = VPair.rowwise().maxCoeff(); // Dirichlet cond
			// VF = VPair.block(0, 0, IV.rows(), 1); // No dirichelt cond
#endif
			++prog;
		}
		auto finish_point = std::chrono::system_clock::now(); 
		std::chrono::duration<double> diff = finish_point - start_point;
		std::cerr << "Performance: " << double(prog.count()) / double(diff.count()) << " iteration/second" << endl;
	}

	void write_frame(std::ostream& fout, double tnow, const Eigen::VectorXd& VF) const
	{
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
	}
};

// FIXME: refactor this piece of mess
int main(int argc, char* argv[])
{
	Eigen::initParallel();
	//MPI_Init(1, argv);

	Simulator simulator;
	simulator.end_t = 10.0;
	simulator.delta_t = 0.1;
	simulator.alpha = 1;
	simulator.snapshot_interval = -1.0;
	simulator.bc = BC_NONE;

	int opt;
	string ofn, lmf, ivf, nbcvfn, massfn;
	simulator.binary = false;
	simulator.check_spd = false;
	while ((opt = getopt(argc, argv, "0:o:t:d:a:l:bDs:vN:m:")) != -1) {
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
			case 't':
				simulator.end_t = atof(optarg);
				break;
			case 'd':
				simulator.delta_t = atof(optarg);
				break;
			case 'a':
				simulator.alpha = atof(optarg);
				break;
			case 'b':
				simulator.binary = true;
				break;
			case 'D':
				simulator.bc |= BC_DIRICHLET;
				break;
			case 'N':
				simulator.bc |= BC_NEUMANN;
				nbcvfn = optarg; // Neumann Boundary Condition Vector File Name
				break;
			case 's':
				simulator.snapshot_interval = atof(optarg);
				break;
			case 'v':
				simulator.check_spd = true;
				break;
			case 'm':
				massfn = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (simulator.snapshot_interval < 0)
		simulator.snapshot_interval = simulator.delta_t;
	if (ivf.empty()) {
		std::cerr << "Missing boundary condition file" << endl;
		usage();
		return -1;
	}
	if (lmf.empty()) {
		std::cerr << "Missing Laplacian matrix file" << endl;
		return -1;
	}
	Eigen::SparseMatrix<double, Eigen::RowMajor>& lap = simulator.lap;
	if (!Eigen::loadMarket(lap, lmf)) {
		std::cerr << "Failed to load Laplacian matrix from file: " << lmf << endl;
		return -1;
	}

	try { 
		vecio::text_read(ivf, simulator.IV);

		if (simulator.bc & BC_NEUMANN) {
			vecio::text_read(nbcvfn, simulator.HSV);
		} else {
			simulator.HSV.setZero(simulator.IV.size()); // No heat source
		}
		if (!massfn.empty())
			vecio::text_read(massfn, simulator.MVec);
	} catch (std::exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	std::unique_ptr<std::ostream> pfout_guard;
	std::ostream* pfout;
	if (ofn.empty()) {
		std::cerr << "Missing output file name, output results to stdout instead." << endl;
		if (simulator.binary && isatty(fileno(stdout)))
			std::cerr << "Binary output format is disabled for tty stdout" << endl;
		simulator.binary = false;
		pfout = &std::cout;
	} else {
		pfout_guard.reset(new std::ofstream(ofn));
		pfout = pfout_guard.get();
	}
	simulator.calibrate_for_hidden_nodes();
	simulator.simulate(*pfout);
	pfout_guard.reset();
	//MPI_Finalize();

	return 0;
}
