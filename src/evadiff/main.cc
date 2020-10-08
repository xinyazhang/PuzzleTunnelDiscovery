/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <tetio/readtet.h>
#include <heatio/readheat.h>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include <igl/barycentric_coordinates.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	cerr <<
R"zzz(Options: -i <tetgen file prefix> -f <heat field data file> -p <path file name>
)zzz";
}

namespace {
	Eigen::Vector3d
	psubtract(const Eigen::Vector3d& lhs,
	          const Eigen::Vector3d& rhs)
	{
		Eigen::Vector3d lhsmin2pi = lhs;
		lhsmin2pi.z() -= M_PI * 2;
		Eigen::Vector3d lhsplus2pi = lhs;
		lhsplus2pi.z() += M_PI * 2;
		Eigen::Vector3d vec[3];
		vec[0] = lhsmin2pi - rhs;
		vec[1] = lhs - rhs;
		vec[2] = lhsplus2pi - rhs;
		double shortest = vec[0].norm();
		int ret = 0;
		for (int i = 1; i < 3; i++) {
			if (vec[i].norm() < shortest) {
				shortest = vec[i].norm();
				ret = i;
			}
		}
#if 0
		std::cerr << vec[0].transpose() << '\t' << vec[1].transpose() << '\t' << vec[2].transpose() << '\t' << endl;
#endif
		return vec[ret];
	}

	double
	vecangle(const Eigen::Vector3d& lhs,
	         const Eigen::Vector3d& rhs)
	{
		Eigen::Vector3d nlhs = lhs.normalized();
		Eigen::Vector3d nrhs = rhs.normalized();
		Eigen::Vector3d nhalf = ((nlhs + nrhs) / 2).normalized();
		double cosine = nlhs.dot(nhalf);
		double ret = std::sqrt(1 - cosine * cosine);
#if 0
		std::cerr << " vec nlhs " << nlhs.transpose() << "\tnrhs " << nrhs.transpose() << endl;
#endif
		return ret;
	}

	double
	curvature(const Eigen::Vector3d& prev,
	          const Eigen::Vector3d& now,
	          const Eigen::Vector3d& next)
	{
		Eigen::Vector3d prevv = psubtract(now, prev);
		Eigen::Vector3d nextv = psubtract(next, now);
		return 2 * vecangle(prevv, nextv) / (prevv.norm() + nextv.norm());
	}
};

class DiffEva {
private:
	Eigen::MatrixXd& V_;
	Eigen::MatrixXi& E_;
	Eigen::MatrixXi& P_;
	vector<HeatFrame>& frames_;

	vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pathvert_;
	vector<int> verttet_;

	double
	interpheat(const Eigen::VectorXd H,
	           const Eigen::Vector3d point,
	           int i)
	{
		Eigen::MatrixXd barycoord;
		Eigen::MatrixXd vertlist = point.transpose();
		Eigen::MatrixXd v0 = V_.row(P_(i,0));
		Eigen::MatrixXd v1 = V_.row(P_(i,1));
		Eigen::MatrixXd v2 = V_.row(P_(i,2));
		Eigen::MatrixXd v3 = V_.row(P_(i,3));

		igl::barycentric_coordinates(vertlist, v0, v1, v2, v3, barycoord);
		Eigen::VectorXd heatvec;
		heatvec.resize(4);
		heatvec(0) = H(P_(i,0));
		heatvec(1) = H(P_(i,1));
		heatvec(2) = H(P_(i,2));
		heatvec(3) = H(P_(i,3));
		double ret = heatvec.dot(barycoord.row(0));
		return ret;
	}

public:
	DiffEva(
		Eigen::MatrixXd& V,
		Eigen::MatrixXi& E,
		Eigen::MatrixXi& P,
		vector<HeatFrame>& frames
		)
		: V_(V), E_(E), P_(P), frames_(frames)
	{
	}

	void load_path(const string& path_file)
	{
		if (path_file.empty())
			return;
		std::ifstream fin(path_file);
		if (!fin.is_open())
			return;
		double x,y,z,tmp;
		int vert, tet;
		while (true) {
			while (fin.peek() == '#' && !fin.eof())
				fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			if (fin.eof())
				break;
			fin >> x >> y >> z >> vert >> tmp >> tet;
			
			//std::cerr << x << ' ' << y << ' ' << z << ' ' << vert << ' ' << tmp << endl;
			if (!fin.eof()) {
				pathvert_.emplace_back(x, y, z);
				verttet_.emplace_back(tet);
			} else {
				break;
			}
			fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		std::cerr << "Loading done" << endl;
	}

	void output(std::ostream& fout)
	{
		std::vector<double> dcurvature;
		dcurvature.resize(pathvert_.size());
#pragma omp parallel for
		for(size_t i = 1; i < pathvert_.size() - 1; i++) {
			Eigen::Vector3d prev = pathvert_[i - 1];
			Eigen::Vector3d now = pathvert_[i];
			Eigen::Vector3d next = pathvert_[i + 1];
			dcurvature[i] = curvature(prev, now, next);
		}
		std::vector<double> point_heat;
		point_heat.resize(pathvert_.size());
		Eigen::VectorXd path_components;
		path_components.setZero(pathvert_.size());
		for (size_t fid = 0 /*frames_.size() - 1*/; fid < frames_.size(); fid++) {
			const auto& frame = frames_[fid];
#pragma omp parallel for
			for(size_t i = 0; i < pathvert_.size(); i++) {
				point_heat[i] = interpheat(frame.hvec, pathvert_[i], verttet_[i]);
			}
#if 0
			for(size_t i = 0; i < pathvert_.size(); i++)
				std::cerr << "point heat " << i << " " << point_heat[i]
				          << "\t dcurvature " << dcurvature[i]
				          << endl;
#endif
#pragma omp parallel for
			for(size_t i = 1; i < pathvert_.size() - 1; i++) {
				double dheat = point_heat[i] - point_heat[i - 1];
				path_components(i) = dheat * dcurvature[i];
			}
			fout << "Frame " << fid << "\tDifficulty: " << path_components.sum() << endl;
		}
	}
};

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ffn, pfn;
	while ((opt = getopt(argc, argv, "i:f:p:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'f':
				ffn = optarg;
				break;
			case 'p':
				pfn = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (iprefix.empty() || ffn.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	vector<HeatFrame> frames;
	vector<double> times;
	try {
		readtet(iprefix, V, E, P, &EBM);

		std::ifstream fin(ffn);
		HeatReader hreader(fin);
		while (true) {
			HeatFrame frame;
			if (!hreader.read_frame(frame))
				break;
			frames.emplace_back(std::move(frame));
			frames.back().hvec.conservativeResize(V.rows()); // Trim hidden nodes.
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	DiffEva de(V,E,P, frames);
	de.load_path(pfn);
	de.output(std::cout);

	return 0;
}
