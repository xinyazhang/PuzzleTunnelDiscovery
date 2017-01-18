#ifndef OMPLAUX_CLEARANCE_H
#define OMPLAUX_CLEARANCE_H

#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <string>
#include <random>
#include <iostream>
#include <math.h>
#include "geo.h"
#include "path.h"
#include "bvh_helper.h"

#define OMPL_CC_DISCRETE_PD 1

#if OMPL_CC_DISCRETE_PD
#include <erocol/hs.h>
#endif

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
	std::vector<BVHModel> rob_cvxbvhs_, env_cvxbvhs_;
	fcl::detail::SplitMethodType split_method_ = fcl::detail::SPLIT_METHOD_MEDIAN;
#if OMPL_CC_DISCRETE_PD
	mutable std::unique_ptr<erocol::HModels> hmodels_;
	erocol::HModels* getHModels() const
	{
		if (!hmodels_) {
			hmodels_.reset(new erocol::HModels(rob_, env_, dtr_, dalpha_));
		}
		return hmodels_.get();
	}
#endif
public:
	using TransformMatrix = Eigen::Matrix<double, 4, 4>;
	using State = Eigen::Matrix<double, 6, 1>;
	static constexpr bool DiscreteFullCubeOnly = true;

	ClearanceCalculator(const Geo& rob, Geo& env)
		:rob_(rob), env_(env)
	{
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

#if OMPL_CC_DISCRETE_PD
	double getPenDepth(const Transform3& tf) const
	{
		return getHModels()->getDiscretePD(tf);
	}
#else
	static double getSingleConvexPD(const BVHModel& rob,
			const BVHModel& env,
			const Transform3& tf)
	{
		fcl::CollisionRequest<Scalar> request;
		fcl::CollisionResult<Scalar> result;
		request.enable_contact = true;
		request.gjk_solver_type = fcl::GST_LIBCCD;
		fcl::collide(&rob, tf, &env, Transform3::Identity(), request, result);
		if (!result.isCollision()) // Note: after convex decomposition, two colliding objects may have non-collide sub-pieces.
			return 0;
		auto nContact = result.numContacts();
#if 0
		double pend = result.getContact(0).penetration_depth;
		for (decltype(nContact) i = 1; i < nContact; i++)
			pend = std::max(pend, result.getContact(i).penetration_depth);
#else
		double pend = -1.0;
		for (decltype(nContact) i = 0; i < nContact; i++)
			pend = std::max(pend, result.getContact(i).penetration_depth);
#endif
		return pend;
	}

	double getPenDepth(const Transform3& tf) const
	{
		// TODO: Support non-convex robot.
		if (env_cvxbvhs_.empty())
			return getSingleConvexPD(rob_bvh_, env_bvh_, tf);
		double ret = 0;
		for (const auto& envcvx : env_cvxbvhs_) {
			ret = std::max(ret, getSingleConvexPD(rob_bvh_, envcvx, tf));
		}
		return ret;
	}
#endif

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

		initConvexBVH(rob_cvxbvhs_, split_method_, rob_);
		initConvexBVH(env_cvxbvhs_, split_method_, env_);
	}

	static void initConvexBVH(std::vector<fcl::BVHModel<BV>> &cvxbvhs,
			fcl::detail::SplitMethodType split_method,
			const Geo& geo)
	{
		if (geo.cvxV.empty() || geo.cvxF.empty())
			return;
		for(size_t i = 0; i < geo.cvxV.size(); i++) {
			const auto& V = geo.cvxV[i];
			const auto& F = geo.cvxF[i];
			cvxbvhs.emplace_back();
			initBVH(cvxbvhs.back(), new fcl::detail::BVSplitter<BV>(split_method), V, F);
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
