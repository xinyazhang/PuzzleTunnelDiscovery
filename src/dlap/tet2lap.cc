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

#define VERBOSE 1
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

#if 0
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
			int angle_edge = opposite_edge[weight_edge]; // Which edge the dihedral angle comes from.
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

Vector3d
project_to_surface(const Vector3d& v_to_proj,
		   const Vector3d& v0,
		   const Vector3d& v1,
		   const Vector3d& v2)
{
	Vector3d normal = (v1 - v0).cross(v2 - v0);
	normal.normalize();
	double d = (v_to_proj - v0).dot(normal);
	return v_to_proj - d * normal;
}

double
dlap3d(const Vector3d& v0,
	      const Vector3d& v1,
	      const VectorXd& va,
	      const VectorXd& vb)
{
	Vector3d pva = project_to_surface(va, v0, v1, vb);
	//std::cerr << "Project " << va << " to (" << v0 << v1 << vb << "), get " << pva << endl;
	Vector3d pvb = project_to_surface(vb, v0, v1, va);
	Vector3d va_pvb = pvb - va;
	Vector3d vb_pva = pva - vb;
	Vector3d pva_va = pva - va;
	Vector3d pvb_vb = pvb - vb;
#if 0
	return ((ab.dot(a_pb) / ab.cross(a_pb).norm())
	     + (ba.dot(b_pa) / ba.cross(b_pa).norm())) / 2.0;
#endif
	//std::cerr << pvb.transpose() << endl;
	//std::cerr << b_pa.transpose() << endl;
	//std::cerr << a_pb.norm() << "\t" << b_pa.norm() << "\t" << ab.norm() << endl;
	return ((va_pvb.norm() / pvb_vb.norm()) + (vb_pva.norm() / pva_va.norm())) / 2.0;
}
#endif


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
	     Eigen::SparseMatrix<double>& lap
	     )
{
	vector<double> vertex_weight;
	//calc_voronoi_volumes(vertex_weight, V, P);
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

#if 0
	Eigen::MatrixXd dihedral_angles, dihedral_ref_cosines, dihedral_cosines, dihedral_sines;
	igl::dihedral_angles(V, P, dihedral_angles, dihedral_ref_cosines);
	dihedral_sines_cosines(V, P, dihedral_sines, dihedral_cosines);
	std::cerr << "== SINES == " << endl
		<< dihedral_sines << endl
		<< "== COSINES ==" << endl
		<< dihedral_cosines << endl
		<< "== REFCOSINES ==" << endl
		<< dihedral_ref_cosines << endl;
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
#if 0
			double el = edge_lengths(ti, ei);
			int opposite_ei = opposite_edge[ei];
#endif
			//int opposite_ei = ei;
			//double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei))) / 6.0;
			//double cot = (1.0 / std::tan(M_PI - dihedral_angles(ti, opposite_ei))) / 6.0;
			//double cot = (dihedral_cosines(ti, ei) / dihedral_sines(ti, ei)) / 6.0;
			//double cot = (1.0 / std::tan(dihedral_angles(ti, opposite_ei)/2)) / 6.0;
			//double w = cot;
			//double w = cot * el;
			//double w = cot / el;
			//double w = cot / (el * el);
			//double w = el * cot * 2;
			//double w = 0.05;
			Vector3d AiNi = surface_areas(ti, vi) * facenormals.block<1, 3>(ti, vi * 3);
			Vector3d AjNj = surface_areas(ti, vj) * facenormals.block<1, 3>(ti, vj * 3);
			double w = AiNi.dot(AjNj) / tetvolumes(ti);

			//int k = P(ti, proto_edge_number[opposite_ei][0]);
			//int l = P(ti, proto_edge_number[opposite_ei][1]);
			//double w = dlap3d(V.row(i), V.row(j), V.row(k), V.row(l));
#if VERBOSE
			std::cerr << " apply weight " << w << " from tet " << ti << " edge " << ei
				  //<< " , = " << dihedral_cosines(ti, ei) << " / " << dihedral_sines(ti, ei) << " / 6.0"
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
