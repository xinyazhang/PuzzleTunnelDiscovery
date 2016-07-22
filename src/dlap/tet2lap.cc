#include "tet2lap.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <igl/volume.h>
#include <igl/dihedral_angles.h>
#include <igl/edge_lengths.h>
#include <boost/progress.hpp>
#include <iostream>

using std::vector;
using std::endl;

#define VERBOSE 0
#define CUT_OFF_PERIODICAL_PART 0
#define SET_FROM_TRIPPLETS 1

// Not "real" Voronoi volume
void calc_voronoi_volumes(
		vector<double>& volumes,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& P)
{
	volumes.resize(V.rows());
	std::fill(volumes.begin(), volumes.end(), 0.0);
	vector<int> nadjtet(V.rows());
	Eigen::MatrixXd tetvolumes;
	igl::volume(V, P, tetvolumes);
	//std::cerr << P << std::endl;
	for(int i = 0; i < P.rows(); i++) {
		double quarter = tetvolumes(i, 0);
		for(int j = 0; j < 4; j++)
			volumes[P(i,j)] += quarter;
	}
	for(auto& w : volumes)
		w = 1/w;
}

#if 0
double calc_edge_weight_in_tet(int v0, int v1, const Eigen::MatrixXd& tet)
{
}

void apply_coe(Eigen::SparseMatrix<double>& lap,
	     const Eigen::VectorXi& P,
	     const Eigen::MatrixXd& V)
{
	Eigen::MatrixXd tet;
	tet.resize(P.cols(), V.cols());
	for(int i = 0; i < P.cols(); i++) {
		tet(i) = V(P(i));
	}
			double wij = calc_edge_weight_in_tet(v0, v1, tet);
			lap(i, j) += wij;
			lap(j, i) += wij;
			lap(i, i) -= wij;
			lap(j, j) -= wij;
		}
	}
}
#endif

// See igl/edge_lengths.cpp for the labeling.
//
// This sequence is the same as the angles returned by dihedral_angles, which
// returns the angles between the faces that share an edge in the tetrahedra.
const int proto_edge_number[][2] = {
	{3, 0},
	{3, 1}, 
	{3, 2},
	{1, 2},
	{2, 0},
	{0, 1}
};

const int opposite_edge[] = {
	3,
	4,
	5,
	0,
	1,
	2
};

#if CUT_OFF_PERIODICAL_PART
/*
 * Note: we want to cut off boundaries that intersect with Z = 0 or Z = 2Pi.
 */
inline bool cross_theta_boundary(const Eigen::VectorXd& v0, const Eigen::VectorXd& v1)
{
	bool v0in = (v0(2) <= M_PI * 2 && v0(2) >= 0);
	bool v1in = (v1(2) <= M_PI * 2 && v1(2) >= 0);
	return (v0in != v1in);
}
#endif

void tet2lap(Eigen::SparseMatrix<double>& lap,
	     const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P)
{
	vector<double> vertex_weight;
	calc_voronoi_volumes(vertex_weight, V, P);

	lap.resize(V.rows(), V.rows());
#if SET_FROM_TRIPPLETS
	typedef Eigen::Triplet<double> tri_t;
	std::vector<tri_t> tris;
	tris.reserve(P.rows() * P.cols() * 4);
#else
	lap.reserve(P.rows() * P.cols() * 4);
#endif

	Eigen::MatrixXd dihedral_angles, dihedral_cosines;
	igl::dihedral_angles(V, P, dihedral_angles, dihedral_cosines);

	Eigen::MatrixXd edge_lengths;
	igl::edge_lengths(V, P, edge_lengths);

#if !VERBOSE
	boost::progress_display prog(P.rows());
#endif
	for(int ti = 0; ti < P.rows(); ti++) {
		for(int ei = 0; ei < 6; ei++) {
			int i = P(ti, proto_edge_number[ei][0]);
			int j = P(ti, proto_edge_number[ei][1]);
#if CUT_OFF_PERIODICAL_PART
			if (cross_theta_boundary(V.row(i), V.row(j)))
				continue;
#endif
			double el = edge_lengths(ti, ei);
			int opposite_ei = opposite_edge[ei];
			//double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei)/2)) / 6.0;
			double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei))) / 6.0;
			double w = el * cot;
			//double w = 0.05;
#if VERBOSE
			std::cerr << " apply weight " << w << " = " << el << " * " << cot << " on edge " << V.row(i) <<"(id: "<< i << ") --- " << V.row(j) <<"(id: "<< j << ")" << endl;
#endif
#if SET_FROM_TRIPPLETS
			tris.emplace_back(i, j, w);
			tris.emplace_back(j, i, w);
			tris.emplace_back(i, i, -w);
			tris.emplace_back(j, j, -w);
#else
			lap.coeffRef(i, j) += w;
			lap.coeffRef(j, i) += w;
			lap.coeffRef(i, i) -= w;
			lap.coeffRef(j, j) -= w;
#endif
		}
#if !VERBOSE
		++prog;
#endif
	}
#if SET_FROM_TRIPPLETS
#if EIGEN_VERSION_AT_LEAST(3,3,0)
	lap.setFromTriplets(tris.begin(),
			    tris.end(),
			    [] (const double& a, const double &b) -> double { return a+b; }
			   );
#else // A slower fix for Eigen < 3.3
	lap.setFromTriplets(tris.begin(), tris.end()); // Stuffing data to it
	for (auto tri : tris) // Reset to zero
		lap.coeffRef(tri.row(), tri.col()) = 0.0;
	for (auto tri : tris) // Accumulate to the correct value
		lap.coeffRef(tri.row(), tri.col()) += tri.value();
#endif
#endif
}
