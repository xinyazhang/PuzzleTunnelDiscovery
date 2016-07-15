#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <string>
#include <memory>
#include <unsupported/Eigen/SparseExtra>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -0 <boundary condition file> -l <Laplacian matrix> [-o output_file -t <end time> -d <time delta> -a <thermal conductivity factor>]" << endl;
}

void simulate(std::ostream& fout,
	      const Eigen::SparseMatrix<double, Eigen::RowMajor>& lap,
	      const Eigen::VectorXd& IV,
	      double delta_t,
	      double end_t,
	      double alpha)
{
	Eigen::VectorXd VF = IV;
	fout.precision(17);
	for(double tnow = 0.0; tnow < end_t; tnow += delta_t) {
#if 0
		Eigen::VectorXd nextVF(VF.rows());
		for(int i = 0; i < V.rows(); i++) {
			nextVF(i) = lap.row(i).dot(VF);
		}
#endif
		fout << "t: " << tnow << "\t" << VF.rows() << endl;
		fout << VF << endl;
		fout << "sum: " << VF.sum() << endl;
		VF += (alpha * lap) * VF;
	}
}

int main(int argc, char* argv[])
{
	int opt;
	string ofn, lmf, ivf;
	double end_t = 10.0, delta_t = 0.1, alpha = 1;
	while ((opt = getopt(argc, argv, "0:o:t:d:a:l:")) != -1) {
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
				end_t = atof(optarg);
				break;
			case 'd':
				delta_t = atof(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	Eigen::VectorXd F;
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

	std::unique_ptr<std::ostream> pfout_guard;
	std::ostream* pfout;
	if (ofn.empty()) {
		std::cerr << "Missing output file name, output results to stdout instead." << endl;
		pfout = &std::cout;
	} else {
		pfout_guard.reset(new std::ofstream(ofn));
		pfout = pfout_guard.get();
	}
	simulate(*pfout, lap, F, delta_t, end_t, alpha);
	pfout_guard.reset();

	return 0;
}
