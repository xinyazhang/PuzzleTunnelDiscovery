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

double
sine(const Eigen::VectorXd& v0, const Eigen::VectorXd& v1)
{
	Eigen::Vector3d x, y;
	x << v0(0), v0(1), v0(2);
	y << v1(0), v1(1), v1(2);
	return x.cross(y).norm()/x.norm()/y.norm();
}

Eigen::VectorXd
project_to(const Eigen::VectorXd& vert,
		const Eigen::VectorXd& evert0,
		const Eigen::VectorXd& evert1)
{
	Eigen::VectorXd edge = evert1 - evert0;
	edge.normalize();
	Eigen::VectorXd e2v = vert - evert0;
	Eigen::VectorXd v_on_e = e2v.dot(edge) * edge + evert0;
	return v_on_e;
}

void
dihedral_sines_cosines(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& P,
		Eigen::MatrixXd& sines,
		Eigen::MatrixXd& cosines)
{
	using namespace Eigen;
	sines.resize(P.rows(), 6);
	cosines.resize(P.rows(), 6);
	for(int p = 0; p < P.rows(); p++) {
		Eigen::VectorXi vindices = P.row(p);
		for(int i = 0; i < 6; i++) {
			int weight_edge = i; // Which pair of vertices to apply the cos/sin weight
			int angle_edge = i; //opposite_edge[weight_edge]; // Which edge the dihedral angle comes from.
			int off_edge = opposite_edge[angle_edge]; // MUST NOT CHANGE
			int off_v0idx = vindices(proto_edge_number[off_edge][0]);
			int off_v1idx = vindices(proto_edge_number[off_edge][1]);
			int angle_v0idx = vindices(proto_edge_number[angle_edge][0]);
			int angle_v1idx = vindices(proto_edge_number[angle_edge][1]);
			VectorXd v0 = V.row(off_v0idx);
			VectorXd v1 = V.row(off_v1idx);
			VectorXd ev0 = V.row(angle_v0idx);
			VectorXd ev1 = V.row(angle_v1idx);
			VectorXd dv0 = project_to(v0, ev0, ev1) - v0;
			VectorXd dv1 = project_to(v1, ev0, ev1) - v1;
			dv0.normalize();
			dv1.normalize();
			sines(p, weight_edge) = sine(dv0, dv1);
			cosines(p, weight_edge) = dv0.dot(dv1);
		}
	}
}

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

void tet2lap(const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P,
#if 0
	     const Eigen::MatrixXd& VNodes,
	     const std::vector<VoronoiEdge>& VEdges,
	     const std::vector<VoronoiFace>& VFaces,
	     const std::vector<VoronoiCell>& VCells,
#endif
	     Eigen::SparseMatrix<double>& lap
	     )
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

	Eigen::MatrixXd dihedral_angles, dihedral_ref_cosines, dihedral_cosines, dihedral_sines;
	igl::dihedral_angles(V, P, dihedral_angles, dihedral_ref_cosines);
	dihedral_sines_cosines(V, P, dihedral_sines, dihedral_cosines);
	std::cerr << "== SINES == " << endl
		<< dihedral_sines << endl
		<< "== COSINES ==" << endl
		<< dihedral_cosines << endl
		<< "== REFCOSINES ==" << endl
		<< dihedral_ref_cosines << endl;

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
			//int opposite_ei = opposite_edge[ei];
			int opposite_ei = ei;
			//double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei))) / 6.0;
			double cot = (dihedral_cosines(ti, ei) / dihedral_sines(ti, ei)) / 6.0;
			//double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei)/2)) / 6.0;
			//double w = cot;
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
