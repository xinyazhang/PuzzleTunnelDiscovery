#include "tet2lap.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <igl/volume.h>
#include <igl/dihedral_angles.h>
#include <igl/face_areas.h>
#include <igl/edge_lengths.h>
#include <boost/progress.hpp>
#include <iostream>

using std::vector;
using std::endl;
using Eigen::Vector3d;
using Eigen::VectorXd;

#define VERBOSE 0
#define CUT_OFF_PERIODICAL_PART 0
#define SET_FROM_TRIPPLETS 1

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

const int
proto_face_number[4][3] = {
	{1, 3, 2},
	{0, 2, 3},
	{3, 1, 0},
	{0, 1, 2}
};

void
tet_face_normals(const Eigen::MatrixXd& V,
		 const Eigen::MatrixXi& P,
		 Eigen::MatrixXd& facenormals)
{
	facenormals.resize(P.rows(), 3 * 4);
	for(int i = 0; i < P.rows(); i++) {
		for(int vi = 0; vi < 4; vi++) {
			Vector3d v0 = V.row(P(i, proto_face_number[vi][0]));
			Vector3d v1 = V.row(P(i, proto_face_number[vi][1]));
			Vector3d v2 = V.row(P(i, proto_face_number[vi][2]));
			Vector3d n = (v1 - v0).cross(v2 - v0);
			facenormals.block<1, 3>(i, vi * 3) = n.normalized();
		}
	}
}

void tet2lap(const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P,
#if 0
	     const Eigen::MatrixXd& VNodes,
	     const std::vector<VoronoiEdge>& VEdges,
	     const std::vector<VoronoiFace>& VFaces,
	     const std::vector<VoronoiCell>& VCells,
#endif
	     Eigen::SparseMatrix<double>& lap,
	     bool unit_weight
	     )
{
	vector<double> vertex_weight;
	Eigen::VectorXd tetvolumes;
	igl::volume(V, P, tetvolumes);
	Eigen::MatrixXd facenormals;
	tet_face_normals(V, P, facenormals);
	Eigen::MatrixXd edge_lengths;
	igl::edge_lengths(V, P, edge_lengths);
	Eigen::MatrixXd surface_areas;
	igl::face_areas(edge_lengths, surface_areas);

	lap.resize(V.rows(), V.rows());
#if SET_FROM_TRIPPLETS
	typedef Eigen::Triplet<double> tri_t;
	std::vector<tri_t> tris;
	tris.reserve(P.rows() * P.cols() * 4);
#else
	lap.reserve(P.rows() * P.cols() * 4);
#endif

#if !VERBOSE
	boost::progress_display prog(P.rows());
#endif
	for(int ti = 0; ti < P.rows(); ti++) {
		for(int ei = 0; ei < 6; ei++) {
			int vi = proto_edge_number[ei][0];
			int vj = proto_edge_number[ei][1];
			int i = P(ti, vi);
			int j = P(ti, vj);
#if CUT_OFF_PERIODICAL_PART
			if (cross_theta_boundary(V.row(i), V.row(j)))
				continue;
#endif
			double w;
			if (!unit_weight) {
				Vector3d AiNi = surface_areas(ti, vi) * facenormals.block<1, 3>(ti, vi * 3);
				Vector3d AjNj = surface_areas(ti, vj) * facenormals.block<1, 3>(ti, vj * 3);
				w = - (AiNi.dot(AjNj) / tetvolumes(ti));
			} else {
				w = 1.0;
			}

#if VERBOSE
			std::cerr << " apply weight " << w << " from tet " << ti << " edge " << ei
				  << " on edge "
				  << V.row(i) <<" (id: "<< i << ") --- " << V.row(j) <<" (id: "<< j << ")"
				  << " surface areas: " << surface_areas(ti, vi) << "\t" << surface_areas(ti, vj)
				  << endl;
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
