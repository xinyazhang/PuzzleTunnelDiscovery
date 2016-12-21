#include "naiveclearance.h"
#include <math.h>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>

struct NaiveClearance::NaiveClearancePrivate {
	using BV = fcl::OBBRSS<double>;
	using Scalar = typename BV::S;
	using BVHModel = fcl::BVHModel<BV>;
	using Transform3 = fcl::Transform3<Scalar>;
	fcl::detail::SplitMethodType split_method = fcl::detail::SPLIT_METHOD_MEDIAN;

	const Geo& env;
	BVHModel env_bvh;
	fcl::Sphere<Scalar> rob{0.001};

	NaiveClearancePrivate(const Geo* _env)
		:env(*_env)
	{
		buildBVHs();
	}

	void buildBVHs()
	{
		initBVH(env_bvh, split_method, env);
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

};

NaiveClearance::NaiveClearance(Geo& env)
	:p_(new NaiveClearancePrivate(&env))
{
}

NaiveClearance::~NaiveClearance()
{
}

namespace {
	constexpr double sqrt2 = 1.41421356237;
};

Eigen::VectorXd NaiveClearance::getCertainCube(const Eigen::Vector2d& state, bool &isfree)
{
	using Scalar = double;
	using Transform3 = fcl::Transform3<Scalar>;

	Transform3 tf{Transform3::Identity()};
	tf.translation() = fcl::Vector3<Scalar>(state.x(), state.y(), 0.0);
	
	fcl::DistanceRequest<Scalar> request(true);
	request.enable_signed_distance = true;
	request.enable_nearest_points = true;
	request.gjk_solver_type = fcl::GST_LIBCCD;
	fcl::DistanceResult<Scalar> result;

	fcl::distance(&p_->rob, tf, &p_->env_bvh, Transform3::Identity(), request, result);
	auto d = result.min_distance;
	isfree = d > 0;

#if 0
	std::cerr << "\tState: " << state.transpose() << "\tDistance: " << d << std::endl;
#endif
	double cd = fabs(d) / sqrt2;

	return Eigen::Vector2d(cd, cd);
}

