#include "tet2lap.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <igl/volume.h>
#include <igl/dihedral_angles.h>
#include <igl/edge_lengths.h>
#include <boost/progress.hpp>

using std::vector;

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

void tet2lap(Eigen::SparseMatrix<double>& lap,
	     const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P)
{
	vector<double> vertex_weight;
	calc_voronoi_volumes(vertex_weight, V, P);

	lap.resize(V.rows(), V.rows());
	lap.reserve(P.rows() * P.cols() * 4);

	Eigen::MatrixXd dihedral_angles, dihedral_cosines;
	igl::dihedral_angles(V, P, dihedral_angles, dihedral_cosines);

	Eigen::MatrixXd edge_lengths;
	igl::edge_lengths(V, P, edge_lengths);

	boost::progress_display prog(P.rows());
	for(int ti = 0; ti < P.rows(); ti++) {
		for(int ei = 0; ei < P.cols(); ei++) {
			int i = P(ti, proto_edge_number[ei][0]);
			int j = P(ti, proto_edge_number[ei][1]);
			double el = edge_lengths(ti, ei);
			double w = el * (1.0 / std::tan(dihedral_angles(ti, ei)));
			lap.coeffRef(i, j) += w;
			lap.coeffRef(j, i) += w;
			lap.coeffRef(i, i) -= w;
			lap.coeffRef(j, j) -= w;
		}
		++prog;
	}
}
