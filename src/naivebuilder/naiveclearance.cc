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

struct NaiveClearance::NaiveClearancePrivate {
	using BV = fcl::OBBRSS<double>;
	using Scalar = typename BV::S;
	using BVHModel = fcl::BVHModel<BV>;
	using Transform3 = fcl::Transform3<Scalar>;
	fcl::detail::SplitMethodType split_method = fcl::detail::SPLIT_METHOD_MEDIAN;

	const Geo& env;
	BVHModel env_bvh;
	fcl::Sphere<Scalar> rob{0.001};

	Geo smash_env;
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
			for (int i = 0; i < oldF.rows(); i++) {
				if (N.row(i).dot(posZ) >= 0)
					oldF.row(i) << -1, -1, -1;
			}
			Eigen::VectorXi I;
			igl::remove_unreferenced(env.V, oldF, smash_env.V, smash_env.F, I);
		}
		Eigen::MatrixXi E;
		igl::edges(smash_env.F, E);
		igl::is_boundary_edge(E, smash_env.F, smash_env_B);
		Geo tmp = smash_env;
		tmp.V.conservativeResize(tmp.V.rows(), 2); // Trim the Z
		for (int i = 0; i < tmp.F.rows(); i++) {
			if (smash_env_B(i))
				continue;
			tmp.F.row(i) << -1, -1;
		}
		Eigen::VectorXi I;
		igl::remove_unreferenced(tmp.V, tmp.F, silhouette.V, silhouette.F, I);
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
			ret = std::max(d, ret);
		}
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
	if (!isfree) {
		d = p_->getPDt(state);
	}

#if 0
	std::cerr << "\tState: " << state.transpose() << "\tDistance: " << d << std::endl;
#endif
	double cd = fabs(d) / sqrt2;

	return Eigen::Vector2d(cd, cd);
}

