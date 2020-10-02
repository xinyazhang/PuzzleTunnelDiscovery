/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OMPLAUX_CLEARANCE_H
#define OMPLAUX_CLEARANCE_H

#include "convex.h"
#include <ccdxx/EigenCCD.h>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <string>
#include <random>
#include <iostream>
#include <math.h>
#include "geo.h"
#include "path.h"
#include "bvh_helper.h"

/*
 * BV: bounding volume, like fcl::OBBRSS<double>
 */
template<typename BV>
class ClearanceCalculator {
private:
	const Geo &rob_, &env_;
	using Scalar = typename BV::S;
	using BVHModel = fcl::BVHModel<BV>;
	using Transform3 = fcl::Transform3<Scalar>;
	using TraversalNode = fcl::detail::MeshDistanceTraversalNodeOBBRSS<Scalar>;
	static constexpr int qsize = 2; // What's qsize?
	//static constexpr double invsqrt3 = 1.0/1.73205080756887729352;
	static constexpr double invsqrt3 = 0.577350269189625764509148780502;

	BVHModel rob_bvh_, env_bvh_;
	fcl::detail::SplitMethodType split_method_ = fcl::detail::SPLIT_METHOD_MEDIAN;

	using Convex = EigenCCD;
	using ConvexPtr = std::unique_ptr<Convex>;
	ConvexPtr rob_cvx_;
	std::vector<ConvexPtr> env_cvxs_;

public:
	using TransformMatrix = Eigen::Matrix<double, 4, 4>;
	using State = Eigen::Matrix<double, 6, 1>;
	static constexpr bool DiscreteFullCubeOnly = true;

	ClearanceCalculator(const Geo& rob, Geo& env)
		:rob_(rob), env_(env)
	{
		buildCVXs();
		buildBVHs();
	}

	// Nobody uses C
	void setC(double min, double max)
	{
		mintr_ = min;
		maxtr_ = max;
		dtr_ = (max - min)/2.0;
	}

	void setDAlpha(double dalpha)
	{
		dalpha_ = dalpha;
	}

	Eigen::Vector2d getCSize() const
	{
		return {};
	}

	static double getSingleConvexPD(EigenCCD* rob,
			const EigenCCD* env,
			const Transform3& tf)
	{
		rob->setTransform(tf);
		EigenCCD::PenetrationInfo pinfo;
		if (!EigenCCD::penetrate(rob, env, pinfo))
			return 0.0;
		return pinfo.depth;
	}

	double getPenDepth(const Transform3& tf) const
	{
		// TODO: Support non-convex robot.
		if (env_cvxs_.empty())
			throw std::string("Convex Decomposition is mandantory");
		double ret = 0;
		int i = 0;
		int maxi = -1;
		for (const auto& envcvx : env_cvxs_) {
			double pd = getSingleConvexPD(rob_cvx_.get(), envcvx.get(), tf);
			if (pd > ret) {
				ret = pd;
				maxi = i;
			}
			i++;
			// std::cerr << "getPenDepth: " << i << std::endl;
		}
		// std::cerr << "PD comes from " << maxi << ", value: " << ret << std::endl;
		return ret;
	}

	State getCertainCube(const State& state, bool& isfree, double* pd = nullptr) const
	{
		auto trmat = Path::stateToMatrix(state, rob_.center);
		Transform3 tf { Transform3::Identity() };
		tf = trmat.block<3,4>(0,0);

		fcl::DistanceRequest<Scalar> request;
		request.gjk_solver_type = fcl::GST_LIBCCD;

		fcl::DistanceResult<Scalar> result;
		fcl::distance(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
		double distance = result.min_distance;
		double d = distance;

		if (distance <= 0)
			d = getPenDepth(tf);

		isfree = distance > 0;
		// std::cerr << "Trying to bound " << state.transpose() << " from free: " << isfree << " and d: " << d << std::endl;
		State ret = bound(d, tf);
		if (d <= 0)
			std::cerr << "fcl distance/collide failed\n";

#if 0
		std::cerr << "state: " << state.transpose()
		          << "\tdistance: " << d
		          << "\tfree: " << isfree
		          << std::endl;
#endif
		if (pd)
			*pd = d;

		return ret;
	}

	// We need the range of C
	State bound(double d, const Transform3& tf) const
	{
		const Eigen::MatrixXd& RV = rob_.V;
		Eigen::Vector3d nrcenter = tf * rob_.center;

		double scale_ratio = 1.0;
		for (int i = 0; i < RV.rows(); i++) {
			// v: relative coordinates w.r.t. robot center.
			Eigen::Vector3d v = tf * Eigen::Vector3d(RV.row(i)) - nrcenter;
			double r = v.norm();
			double bs = binsolve(dtr_, dalpha_, r, d);
			scale_ratio = std::min(scale_ratio, bs);
			// std::cerr << "binsolve d: " << d << " and r: " << r
			//	<< "\n\treturns: " << bs << "\t or: " << 1.0/bs << std::endl;
		}
		State ret;
		ret << dtr_ * scale_ratio, dtr_ * scale_ratio, dtr_ * scale_ratio,
		       dalpha_ * scale_ratio, dalpha_/2.0 * scale_ratio, dalpha_ * scale_ratio;
		return ret;
	}
protected:
	void buildBVHs()
	{
		initBVH(rob_bvh_, new fcl::detail::BVSplitter<BV>(split_method_), rob_.V, rob_.F);
		initBVH(env_bvh_, new fcl::detail::BVSplitter<BV>(split_method_), env_.V, env_.F);
	}

	void buildCVXs()
	{
		rob_cvx_ = Convex::create(rob_.V, rob_.F, &rob_.center);

		env_cvxs_.resize(env_.cvxV.size());
		for(size_t i = 0; i < env_.cvxV.size(); i++) {
			const auto& V = env_.cvxV[i];
			const auto& F = env_.cvxF[i];
			env_cvxs_[i] = Convex::create(V, F);
		}
	}

	// Note: dx and dalpha is delta, which is the half size of the cube.
	static double binsolve(double maxdx, double maxdalpha, double r, double mindist)
	{
		double upperrange = 1.0;
		double lowerrange = 0.0;
		double sqrt3 = std::sqrt(3.0);
		bool mid_is_valid = false;
		while (upperrange - lowerrange > 1e-6) {
			double prob = (upperrange + lowerrange)/2;
			double dx = maxdx * prob;
			double dalpha = maxdalpha * prob;
			double value = 2.5 * r * dalpha + sqrt3 * dx;
			if (value > mindist) {
				upperrange = prob;
				mid_is_valid = false;
			} else if (value < mindist) {
				lowerrange = prob;
				mid_is_valid = true;
			} else {
				return prob;
			}
		}
		if (mid_is_valid)
			return (upperrange + lowerrange)/2;
		return lowerrange;
	}

	double mintr_, maxtr_;
	double dtr_;
	double dalpha_ = M_PI;
};

#endif
