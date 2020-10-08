/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "naiveclearance.h"
#include <math.h>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <igl/per_face_normals.h>
#include <igl/remove_unreferenced.h>
#include <igl/edges.h>
#include <igl/is_boundary_edge.h>
#include <igl/barycentric_coordinates.h>

struct NaiveClearance::NaiveClearancePrivate {
	using BV = fcl::OBBRSS<double>;
	using Scalar = typename BV::S;
	using BVHModel = fcl::BVHModel<BV>;
	using Transform3 = fcl::Transform3<Scalar>;
	fcl::detail::SplitMethodType split_method = fcl::detail::SPLIT_METHOD_MEDIAN;

	const Geo& env;
	BVHModel env_bvh;
	fcl::Sphere<Scalar> rob{0.01};

	Geo smash_env;
	Eigen::MatrixXd smashABC[3];
	Eigen::VectorXi smash_env_B;
	Geo silhouette;

	NaiveClearancePrivate(const Geo* _env)
		:env(*_env)
	{
		buildBVHs();
		smashSurface();
	}

	void buildBVHs()
	{
		initBVH(env_bvh, split_method, env);
	}

	void smashSurface()
	{
		static const Eigen::Vector3d posZ{0,0,1};
		{
			Eigen::MatrixXd N;
			igl::per_face_normals(env.V, env.F, N);
			Eigen::MatrixXi oldF = env.F;
			Eigen::VectorXi keepV;
			keepV.resize(oldF.rows());
			for (int i = 0; i < oldF.rows(); i++) {
				if (N.row(i).dot(posZ) >= 0) {
					oldF.row(i) << -1, -1, -1;
					keepV(i) = 0;
				} else
					keepV(i) = 1;
			}
			Eigen::VectorXi I;
			Eigen::MatrixXi newF;
			igl::remove_unreferenced(env.V, oldF, smash_env.V, newF, I);
			smash_env.F.resize(keepV.count(), oldF.cols());
			for (int i = 0, fi = 0; i < newF.rows(); i++) {
				if (keepV(i)) {
					smash_env.F.row(fi) = newF.row(i);
					fi++;
				}
			}
#if 0
			for (int i = 0; i < 3; i++) {
				smashABC[i].resize(newF.rows(), 2);
				for (int f = 0; f < smash_env.F.rows(); f++) {
					smashABC[i].row(f) = smash_env.V.row(smash_env.F(f, i));
				}
			}
#endif
		}
		Eigen::MatrixXi E;
		igl::edges(smash_env.F, E);
		igl::is_boundary_edge(E, smash_env.F, smash_env_B);

		Eigen::MatrixXd tmpV;
		tmpV.resize(smash_env.V.rows(), 2); // Trim the Z
		tmpV.block(0, 0, smash_env.V.rows(), 2) = smash_env.V.block(0, 0,
				smash_env.V.rows(), 2);

		std::vector<bool> keep(E.rows());
		for (int i = 0; i < E.rows(); i++) {
			if (smash_env_B(i)) {
				continue;
			}
			E.row(i) << -1, -1;
		}
		Eigen::VectorXi I;
		Eigen::MatrixXi renumberedE;
		igl::remove_unreferenced(tmpV, E, silhouette.V, renumberedE, I);
		silhouette.F.resize(smash_env_B.count(), 2);
		for (int i = 0, fi = 0; i < renumberedE.rows(); i++) {
			if (smash_env_B(i) == 0)
				continue;
			silhouette.F.row(fi) = renumberedE.row(i);
			fi++;
		}

		// std::cerr << silhouette.V << "\n" << silhouette.F << "\n";
	}

	static void initBVH(fcl::BVHModel<BV> &bvh,
			fcl::detail::SplitMethodType split_method,
			const Geo& geo)
	{
		bvh.bv_splitter.reset(new fcl::detail::BVSplitter<BV>(split_method)); bvh.beginModel(); std::vector<Eigen::Vector3d> Vs(geo.V.rows());
		std::vector<fcl::Triangle> Fs(geo.F.rows());
		for (int i = 0; i < geo.V.rows(); i++)
			Vs[i] = geo.V.row(i);
		for (int i = 0; i < geo.F.rows(); i++) {
			Eigen::Vector3i F = geo.F.row(i);
			Fs[i] = fcl::Triangle(F(0), F(1), F(2));
		}
		bvh.addSubModel(Vs, Fs);
		bvh.endModel();
	}

	static double distance2d(
			const Eigen::Vector2d& c,
			const Eigen::Vector2d& v0,
			const Eigen::Vector2d v1)
	{
		Eigen::Vector2d l = v1 - v0, n;
		n << -l(1), l(0);
		n = n.normalized();
		Eigen::Vector2d p = n * n.dot(v0 - c);
		double t = ((c + p - v0).dot(l))/l.squaredNorm();
		t = std::max(std::min(t, 1.0), 0.0);
		return (c - (v0 + t * l)).norm();
	}

	double getPDt(const Eigen::Vector2d& center)
	{
		double ret = distance2d(center,
				silhouette.V.row(silhouette.F(0,0)),
				silhouette.V.row(silhouette.F(0,1)));
		for (int i = 1; i < silhouette.F.rows(); i++) {
			double d = distance2d(center,
				silhouette.V.row(silhouette.F(i,0)),
				silhouette.V.row(silhouette.F(i,1)));
			ret = std::min(d, ret);
		}
#if VERBOSE
		if (std::abs(center(0)) < 0.1)
			std::cerr << "PDt from " << center.transpose() << " = " << ret << std::endl;
#endif
		return ret;
	}
};

NaiveClearance::NaiveClearance(Geo& env)
	:p_(new NaiveClearancePrivate(&env))
{
}

NaiveClearance::~NaiveClearance()
{
}

namespace {
	constexpr double sqrt2 = 1.41421356237309504880;
};

#define USE_FCL_FOR_2D 0

Eigen::VectorXd NaiveClearance::getCertainCube(const Eigen::Vector2d& state, bool &isfree)
{
#if USE_FCL_FOR_2D
	using Scalar = double;
	using Transform3 = fcl::Transform3<Scalar>;

	Transform3 tf{Transform3::Identity()};
	tf.translation() = fcl::Vector3<Scalar>(state.x(), state.y(), 0.0);
	
	fcl::DistanceRequest<Scalar> request(true);
#if 0
	request.enable_signed_distance = true;
	request.enable_nearest_points = true;
#endif
	request.gjk_solver_type = fcl::GST_LIBCCD;
	fcl::DistanceResult<Scalar> result;

	fcl::distance(&p_->rob, tf, &p_->env_bvh, Transform3::Identity(), request, result);
	auto d = result.min_distance;
	isfree = d > 1e-6;
	if (!isfree) {
		d = p_->getPDt(state);
	}
#else
	isfree = true;
	Eigen::MatrixXd cstate(1, 2);
	cstate << state(0), state(1);
	for (int i = 0; i < p_->smash_env.F.rows(); i++) {
		Eigen::MatrixXd bc;
		Eigen::Vector3i F = p_->smash_env.F.row(i);
		igl::barycentric_coordinates(
				cstate,
				p_->smash_env.V.block<1,2>(F(0), 0),
				p_->smash_env.V.block<1,2>(F(1), 0),
				p_->smash_env.V.block<1,2>(F(2), 0),
				bc);
		if (bc(0) >= 0 && bc(1) >= 0 && bc(2) >= 0) {
			isfree = false;
#if 0
			std::cerr << state.transpose() << "\thas barycenter coordinates "
				<< bc.transpose()
				<< " from triangle:"
				<< "\n\t" << p_->smash_env.V.row(F(0))
				<< "\n\t" << p_->smash_env.V.row(F(1))
				<< "\n\t" << p_->smash_env.V.row(F(2))
				<< std::endl;
#endif
			break;
		}
	}
	double d = p_->getPDt(state);
#endif

#if 0
	std::cerr << "\tState: " << state.transpose() << "\tDistance: " << d << std::endl;
	if (std::abs(state(0)) < 0.1)
		std::cerr << "distance from " << state.transpose() << " = " << d << " free: " << isfree << std::endl;
#endif
	double cd = fabs(d) / sqrt2;

	return Eigen::Vector2d(cd, cd);
}

