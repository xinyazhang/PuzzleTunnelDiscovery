#ifndef CLEARANCER_H
#define CLEARANCER_H

#include <omplaux/geo.h>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <fcl/narrowphase/collision.h>

template<typename BV>
class TranslationOnlyClearance {
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
public:
	using TransformMatrix = Eigen::Matrix<double, 4, 4>;
	using State = Eigen::Matrix<double, 3, 1>;

	TranslationOnlyClearance(const Geo& rob, Geo& env)
		:rob_(rob), env_(env)
	{
		buildBVHs();
	}

	// Nobody uses C
	void setC(double min, double max)
	{
	}

	Eigen::Vector2d getCSize() const
	{
		return {};
	}

	State getCertainCube(const State& state, bool& isfree) const
	{
		Transform3 tf { Transform3::Identity() };
		tf.translate(state);

		fcl::DistanceRequest<Scalar> request;
		request.gjk_solver_type = fcl::GST_LIBCCD;

		fcl::DistanceResult<Scalar> result;
		fcl::distance(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
		double distance = result.min_distance;
		double d = distance;
		if (distance <= 0) {
			fcl::CollisionRequest<Scalar> request;
			fcl::CollisionResult<Scalar> result;
			request.enable_contact = true;
			request.gjk_solver_type = fcl::GST_LIBCCD;
			fcl::collide(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
			if (!result.isCollision())
				std::cerr << "WIERD, distance = 0 but not collide\n";
			auto nContact = result.numContacts();
			double pend = result.getContact(0).penetration_depth;
			for (decltype(nContact) i = 1; i < nContact; i++)
				pend = std::max(pend, result.getContact(i).penetration_depth);
			d = pend;
		}

		State ret;
		ret << d*invsqrt3, d*invsqrt3, d*invsqrt3;
		isfree = distance > 0;

		return ret;
	}
protected:
	void buildBVHs()
	{
		initBVH(rob_bvh_, split_method_, rob_);
		initBVH(env_bvh_, split_method_, env_);
	}

	static void initBVH(fcl::BVHModel<BV> &bvh,
			fcl::detail::SplitMethodType split_method,
			const Geo& geo)
	{
		bvh.bv_splitter.reset(new fcl::detail::BVSplitter<BV>(split_method));
		bvh.beginModel();
		std::vector<Eigen::Vector3d> Vs(geo.V.rows());
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

#endif
