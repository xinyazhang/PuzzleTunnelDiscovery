/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>
#include <memory>
#include <boost/progress.hpp>
//#include <Eigen/SparseCore>
//#include <unsupported/Eigen/SparseExtra>
//#include <Eigen/SparseLU> 
//#include <Eigen/SparseCholesky>
//#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/BVH>
#include <igl/barycentric_coordinates.h>
#include <igl/ray_mesh_intersect.h>

#include <heatio/readheat.h>
#include <tetio/readtet.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

#define VERBOSE 0

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> -t <temperature file> -b <boundary vertex file> [-c -o <output path file> -f time_frame] <x y z>" << endl
		<< "\t-c: continuous follow" << endl;
}

#define FACE_PER_TET 4
#define EDGE_PER_TET 6

const int
proto_face_number[FACE_PER_TET][3] = {
	{1, 3, 2},
	{0, 2, 3},
	{3, 1, 0},
	{0, 1, 2}
};

const int
proto_edge_number[EDGE_PER_TET][2] = {
	{3, 0},
	{3, 1}, 
	{3, 2},
	{1, 2},
	{2, 0},
	{0, 1}
};

bool
in_tet(const Eigen::Vector3d& start_point,
       const Eigen::MatrixXd& T)
{
	for (int i = 0; i < FACE_PER_TET; i++) {
		int v0i = proto_face_number[i][0];
		int v1i = proto_face_number[i][1];
		int v2i = proto_face_number[i][2];
		Eigen::Vector3d v0 = T.col(v0i);
		Eigen::Vector3d v1 = T.col(v1i);
		Eigen::Vector3d v2 = T.col(v2i);
		Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
		if (n.dot(start_point - v0) < 0)
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

typedef Eigen::AlignedBox<double, 3> BBox;

BBox
tet_bb(const std::vector<Eigen::VectorXd>& tetverts)
{
	Eigen::MatrixXd mat;
	// each colume is a a tetvert
	mat.resize(tetverts.front().size(), tetverts.size());
	for(size_t i = 0; i < tetverts.size(); i++)
		mat.col(i) = tetverts[i];
	Eigen::VectorXd minV = mat.colwise().minCoeff();
	Eigen::VectorXd maxV = mat.colwise().maxCoeff();
	//std::cerr << minV.transpose() << "\t" << maxV.transpose() << endl;
	return BBox(minV, maxV);
}

double
tet_interp(const Eigen::MatrixXd& V,
           const Eigen::MatrixXi& P,
           const Eigen::VectorXd& H,
           int i,
           const Eigen::Vector3d& c)
{
	Eigen::MatrixXd barycoord;
	Eigen::MatrixXd vertlist = c.transpose();
	Eigen::MatrixXd v0 = V.row(P(i,0));
	Eigen::MatrixXd v1 = V.row(P(i,1));
	Eigen::MatrixXd v2 = V.row(P(i,2));
	Eigen::MatrixXd v3 = V.row(P(i,3));

	igl::barycentric_coordinates(vertlist, v0, v1, v2, v3, barycoord);
	Eigen::VectorXd heatvec;
	heatvec.resize(4);
	heatvec(0) = H(P(i,0));
	heatvec(1) = H(P(i,1));
	heatvec(2) = H(P(i,2));
	heatvec(3) = H(P(i,3));
	double ret = heatvec.dot(barycoord.row(0));
#if VERBOSE
	std::cerr << "B. coord: " << barycoord << endl;
	std::cerr << "Heat vec: " << heatvec.transpose() << endl;
	std::cerr << "Heat value: " << ret << endl;
#endif
	return ret;
}

Eigen::Vector3d
tet_gradient(const Eigen::MatrixXd& V,
             const Eigen::MatrixXi& P,
             const Eigen::VectorXd& H,
             int i)
{
	int vi0 = P(i,0);
	int vi1 = P(i,1);
	int vi2 = P(i,2);
	int vi3 = P(i,3);
	Eigen::Vector3d vec1 = V.row(vi1) - V.row(vi0);
	Eigen::Vector3d vec2 = V.row(vi2) - V.row(vi0);
	Eigen::Vector3d vec3 = V.row(vi3) - V.row(vi0);
	double d1 = H(vi1) - H(vi0);
	double d2 = H(vi2) - H(vi0);
	double d3 = H(vi3) - H(vi0);

	return (vec1 * d1) / vec1.squaredNorm() + 
	       (vec2 * d2) / vec2.squaredNorm() + 
	       (vec3 * d3) / vec3.squaredNorm();
}

double
tet_intersect(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& P,
              int tetid,
              const Eigen::Vector3d& center,
              const Eigen::Vector3d& grad)
{
	Eigen::MatrixXi F;
	F.resize(4, 3);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			F(i, j) = P(tetid, proto_face_number[i][j]);
		}
	}
	std::vector<igl::Hit> hits;
	igl::ray_mesh_intersect(center, grad, V, F, hits);
	if (hits.empty())
		return -1.0;
	return hits.front().t;
}

struct Intersector {
	Eigen::Vector3d center_;
	const Eigen::MatrixXd& V_;
	const Eigen::MatrixXi& P_;
	int result_ = -1;
	int vert_id_ = -1; // Neg value to represent we're not on any vertex

	Intersector(const Eigen::Vector3d& center,
	            const Eigen::MatrixXd& V,
	            const Eigen::MatrixXi& P)
		:center_(center), V_(V), P_(P)
	{
	}

	bool intersectVolume(const BBox &box)
	{
		//std::cerr << "Probing BBox " << box.min().transpose() << "\t" << box.max().transpose() << endl;
		return box.contains(center_);
	}

	bool intersectObject(int tetid)
	{
		Eigen::MatrixXd tet(3, 4);
		//std::cerr << "Probing tet " << tetid << endl;
		int i = tetid;
		for (int j = 0; j < P_.cols(); j++) {
			tet.block<3, 1>(0, j) = V_.row(P_(i,j));
		}
		bool ret = in_tet(center_, tet);
		if (ret)
			result_ = tetid;
		return ret;
	}
};

int max_temp_vert_in_tet(
        const Eigen::MatrixXi& P,
        const Eigen::VectorXd& H,
        int tet_id)
{
	int next_vert = P(tet_id, 0);
	double max_temp = H(next_vert);
	for (int j = 1; j < P.cols(); j++) {
		int vert = P(tet_id, j);
		if (H(vert) > max_temp) {
			max_temp = H(vert);
			next_vert = vert;
		}
	}
	return next_vert;
}

double tet_highest_temp(
        const Eigen::MatrixXi& P,
        const Eigen::VectorXd& H,
        int tet_id)
{
	return H(max_temp_vert_in_tet(P, H, tet_id));
}

double tet_average_temp(
        const Eigen::MatrixXi& P,
        const Eigen::VectorXd& H,
        int tet_id)
{
	double ret = 0;
	for (int i = 0; i < P.cols(); i++)
		ret += H(P(tet_id,i));
	return ret;
}

void
cfollow(const Eigen::Vector3d& start_point,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& P,
        const Eigen::VectorXd& MBM,
        const Eigen::VectorXd& H,
        std::ostream& fout)
{
	using namespace Eigen;

	fout.precision(17);
	std::vector<BBox> tetbbox(P.rows());
	std::vector<int> tetidx(P.rows());
	Eigen::VectorXd PBM;
	PBM.setZero(P.rows());
#pragma omp parallel for
	for (int i = 0; i < P.rows(); i++) {
		tetbbox[i] = tet_bb({V.row(P(i,0)), V.row(P(i,1)), V.row(P(i,2)), V.row(P(i,3))});
		tetidx[i] = i;
		for (int j = 0; j < P.cols(); j++) {
			if (MBM(P(i,j))) {
				PBM(i) = 1;
				break;
			}
		}
	}

	//std::cerr << "Test tet_bb" << tetbbox[0].min().transpose() << "\t" << tetbbox[0].max().transpose() << endl;
	KdBVH<double, 3, int> bvh(tetidx.begin(), tetidx.end(), tetbbox.begin(), tetbbox.end());
	Intersector isect(start_point, V, P);
	BVIntersect(bvh, isect);
	if (isect.result_ < 0)
		throw std::runtime_error("Start point isn't in any tetrahedron.");

	std::unordered_map<int, std::vector<int>> vert2tet;
	for (int i = 0; i < P.rows(); i++) {
		for (int j = 0; j < P.cols(); j++) {
			vert2tet[P(i,j)].emplace_back(i);
		}
	}

	std::unordered_map<int, std::set<int>> neigh;
	for (int i = 0; i < P.rows(); i++) {
		for (int ei = 0; ei < EDGE_PER_TET; ei++) {
			int vi = P(i, proto_edge_number[ei][0]);
			int vj = P(i, proto_edge_number[ei][1]);
			neigh[vi].emplace(vj);
			neigh[vj].emplace(vi);
		}
	}

	while (true) {
		double tem;
		if (isect.vert_id_ >= 0) {
			int tet_pick = vert2tet[isect.vert_id_].front();
			double highest_temp = tet_highest_temp(P, H, tet_pick);
			double average_temp = tet_average_temp(P, H, tet_pick);
			for (auto tet : vert2tet[isect.vert_id_]) {
				double high = tet_highest_temp(P, H, tet);
				double ave = tet_average_temp(P, H, tet);
#if VERBOSE
				fout << "# tet: " << tet
				     << " high: " << high
				     << " ave: " << ave
				     << endl;
#endif
				bool replace = false;
				if (high > highest_temp)
					replace = true;
				else if (high == highest_temp && ave > average_temp)
					replace = true;
				if (replace) {
					highest_temp = high;
					average_temp = ave;
					tet_pick = tet;
				}
			}
			Vector3d center(0,0,0);
			for (int i = 0; i < P.cols(); i++) {
				center += V.row(P(tet_pick, i));
			}
			fout << V.row(isect.vert_id_)
			     << "\t" << isect.vert_id_
			     << '\t' << H(isect.vert_id_)
			     << '\t' << tet_pick
			     << endl;
			//isect.center_ = V.row(isect.vert_id_);
			isect.center_ = center / P.cols();
			isect.result_ = tet_pick;
#if VERBOSE
			fout << "# tetpick: " << tet_pick << " for vert: " << isect.vert_id_ << endl
			     << "#\tnew center: " << isect.center_.transpose()
			     << " in tet? " << isect.intersectObject(tet_pick)
			     << endl;
			
			fout << "# Temperature list: " << endl;
			for (int i = 0; i < P.cols(); i++) {
				fout << "#\t" << H(P(tet_pick, i)) << endl;
			}
			fout << "# Neighbor Temperature list: " << endl;
			const auto& nei = neigh[isect.vert_id_];
			for (int vert : nei) {
				fout << "#\t" << H(vert) << "\t" << (H(vert) > H(isect.vert_id_))
				     << endl;
			}
#endif
		}
		tem = tet_interp(V, P, H, isect.result_, isect.center_);
		// In practice, they are all off vertex
		// Because we use the center of tet instead we need be on vertex
		fout << isect.center_.transpose() << "\t" << -1 << '\t' << tem << '\t' << isect.result_;
		auto grad = tet_gradient(V, P, H, isect.result_);
		Vector3d ngrad = grad.normalized();
		double t = tet_intersect(V, P, isect.result_, isect.center_, ngrad);
		fout << "\t#direction: " << ngrad.transpose()
		     << "\tT: " << t
		     << "\ton vert: " << isect.vert_id_;
		fout << endl;

		int prevtet = isect.result_;
		if (t >= 0) {
			isect.center_ += ngrad * (t + 1e-3);
			if (isect.center_(2) > 2 * M_PI)
				isect.center_(2) -= 2 * M_PI;
			else if (isect.center_(2) < 0.0)
				isect.center_(2) += 2 * M_PI;
			isect.result_ = -1;
			BVIntersect(bvh, isect);
			if (isect.result_ > 0) {
				isect.vert_id_ = -1; // In boundary, reset.
				continue;
			}
			if (PBM(prevtet) > 0) {
				std::cerr << "Successfully hit the maze boundary, halt" << endl;
				break;
			}
		}
		// We leaves an inner boundary,
		// or following the vertex doesn't help
		// Handling oob by picking up a highest temp vertex.
		int next_vert = max_temp_vert_in_tet(P, H, prevtet);
		isect.vert_id_ = next_vert;
	}
	fout << isect.center_.transpose() << "\t-1\t1\t-1#This is the End" << endl;
}

void
dfollow(const Eigen::Vector3d& start_point,
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
	int next_vert = max_temp_vert_in_tet(P, H, tet_id);
	double max_temp = H(next_vert);
	std::unordered_map<int, std::set<int>> neigh;
	for (int i = 0; i < P.rows(); i++) {
		for (int ei = 0; ei < EDGE_PER_TET; ei++) {
			int vi = P(i, proto_edge_number[ei][0]);
			int vj = P(i, proto_edge_number[ei][1]);
			neigh[vi].emplace(vj);
			neigh[vj].emplace(vi);
		}
	}
	while (MBM(next_vert) == 0.0) {
		fout << V.row(next_vert) << "\t" << next_vert << "\t" << max_temp << endl;
		const auto& nei = neigh[next_vert];
		Eigen::Vector3d cur_vert = V.row(next_vert);
		double max_grad = 0.0;
		double cur_temp = H(next_vert);
		bool halt = true;
		for (int vert : nei) {
			Eigen::Vector3d vert_pos = V.row(vert);
			double distance = (vert_pos - cur_vert).norm();
			double grad = (H(vert) - cur_temp) / distance;
			//double grad = (H(vert) - cur_temp);
			if (grad > max_grad) {
				max_grad = grad;
				max_temp = H(vert);
				next_vert = vert;
				halt = false;
			}
		}
		if (halt) {
			std::cerr << "No temp. change detected, halt" << endl;
			break;
		}
	}
	fout << V.row(next_vert) << "\t" << next_vert << "\t" << max_temp << endl;
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();
	int opt;
	string iprefix, tfn, ofn, ivf;
	int frame_to_pick = INT_MAX;
	bool continuous = false;
	while ((opt = getopt(argc, argv, "i:t:o:b:f:c")) != -1) {
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
			case 'c':
				continuous = true;
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
		while (frameid < frame_to_pick && hreader.read_frame(hframe))
			frameid++;
		std::ofstream fout;
		if (!ofn.empty()) {
			fout.open(ofn);
			if (!fout.is_open())
				throw std::runtime_error("Cannot open " + ofn + " for write.");
			os = &fout;
		}
		if (continuous)
			cfollow(start_point, V, E, P, MBM, hframe.hvec, *os);
		else 
			dfollow(start_point, V, E, P, MBM, hframe.hvec, *os);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
