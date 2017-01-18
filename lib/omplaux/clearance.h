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
public:
	using TransformMatrix = Eigen::Matrix<double, 4, 4>;
	using State = Eigen::Matrix<double, 6, 1>;

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
		double pend = result.getContact(0).penetration_depth;
		for (decltype(nContact) i = 1; i < nContact; i++)
			pend = std::max(pend, result.getContact(i).penetration_depth);
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

	State getCertainCube(const State& state, bool& isfree) const
	{
		auto trmat = Path::stateToMatrix(state, rob_.center);
		Transform3 tf;
		tf = trmat.block<3,4>(0,0);

		fcl::DistanceRequest<Scalar> request;
		request.gjk_solver_type = fcl::GST_LIBCCD;

		fcl::DistanceResult<Scalar> result;
		fcl::distance(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
		double distance = result.min_distance;
		double d = distance;
		if (distance <= 0)
			d = getPenDepth(tf);

		State ret = bound(d, tf);
		isfree = distance > 0;

#if 0
		std::cerr << "state: " << state.transpose()
		          << "\tdistance: " << d
		          << "\tfree: " << isfree
		          << std::endl;
#endif

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
			scale_ratio = std::min(scale_ratio, binsolve(dtr_, dalpha_, r, d));
		}
		State ret;
		ret << dtr_ * scale_ratio, dtr_ * scale_ratio, dtr_ * scale_ratio,
		       dalpha_ * scale_ratio, dalpha_/2.0 * scale_ratio, dalpha_ * scale_ratio;
		return ret;
	}
protected:
	void buildBVHs()
	{
		initBVH(rob_bvh_, split_method_, rob_.V, rob_.F);
		initBVH(env_bvh_, split_method_, env_.V, env_.F);

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
			initBVH(cvxbvhs.back(), split_method, V, F);
		}
	}

	template<typename VType, typename FType>
	static void initBVH(fcl::BVHModel<BV> &bvh,
			fcl::detail::SplitMethodType split_method,
			const VType& V,
			const FType& F)
	{
		bvh.bv_splitter.reset(new fcl::detail::BVSplitter<BV>(split_method));
		bvh.beginModel();
		std::vector<Eigen::Vector3d> Vs(V.rows());
		std::vector<fcl::Triangle> Fs(F.rows());
		for (int i = 0; i < V.rows(); i++)
			Vs[i] = V.row(i);
		for (int i = 0; i < F.rows(); i++) {
			const auto& f = F.row(i);
			Fs[i] = fcl::Triangle(f(0), f(1), f(2));
		}
		bvh.addSubModel(Vs, Fs);
		bvh.endModel();
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
			double value = 5.0/2.0 * r * fabs(dalpha) + sqrt3 * dx;
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
