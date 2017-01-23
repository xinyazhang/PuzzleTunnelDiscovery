#ifndef CLEARANCER_H
#define CLEARANCER_H

#include <omplaux/geo.h>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <fcl/narrowphase/collision.h>

#ifndef ENABLE_FCL_PROFILING
#define ENABLE_FCL_PROFILING 1
#endif

#if ENABLE_FCL_PROFILING
#include <chrono>
#endif

#ifndef ENABLE_DISCRETE_PD
#define ENABLE_DISCRETE_PD 0
#endif

#if ENABLE_DISCRETE_PD
#include <erocol/hs.h>
#endif

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
	std::vector<BVHModel> rob_cvxbvhs_, env_cvxbvhs_;
	fcl::detail::SplitMethodType split_method_ = fcl::detail::SPLIT_METHOD_MEDIAN;

	double dtr_;
#if ENABLE_DISCRETE_PD
	mutable std::unique_ptr<erocol::HModels> hmodels_;
	erocol::HModels* getHModels() const
	{
		if (!hmodels_) {
			hmodels_.reset(new erocol::HModels(rob_, env_, dtr_, 0.0));
		}
		return hmodels_.get();
	}
#endif

public:
#if ENABLE_FCL_PROFILING
	struct Profiler {
		enum {
			DISTANCE,
			COLLISION,
			TOTAL_NUMBER
		};
		void start()
		{
			tstart = std::chrono::high_resolution_clock::now();
		}

		void stop(int timer)
		{
			tend = std::chrono::high_resolution_clock::now();
			sum[timer] += std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
			++count[timer];
		}

		double getAverageClockMs(int timer) const { return sum[timer]/count[timer]; }
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> tstart, tend;
		double sum[TOTAL_NUMBER] = {0.0};
		int count[TOTAL_NUMBER] = {0};
	};
#else
	struct Profiler {
		enum {
			DISTANCE,
			COLLISION
		};
		void start() {}
		void stop(int) {}
		double getAverageClockMs(int) const { return -1.0;}
	};
#endif

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
		dtr_ = (max - min)/2;
	}

	Eigen::Vector2d getCSize() const
	{
		return {};
	}

#if ENABLE_DISCRETE_PD
	double getPenDepth(const Transform3& tf) const
	{
		return getHModels()->getDiscretePD(tf) + 1e-6;
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
#endif

	State getCertainCube(const State& state, bool& isfree) const
	{
		Transform3 tf { Transform3::Identity() };
		tf.translate(state);

		fcl::DistanceRequest<Scalar> request;
		request.gjk_solver_type = fcl::GST_LIBCCD;

		fcl::DistanceResult<Scalar> result;
		profiler_.start();
		fcl::distance(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
		profiler_.stop(Profiler::DISTANCE);
		double distance = result.min_distance;
		double d = distance;
		if (distance <= 0) {
			profiler_.start();
			d = getPenDepth(tf);
			profiler_.stop(Profiler::COLLISION);
		}

		State ret;
		ret << d*invsqrt3, d*invsqrt3, d*invsqrt3;
		isfree = distance > 0;

		return ret;
	}

	const Profiler& getProfiler() const { return profiler_; }
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

	mutable Profiler profiler_;
};

#endif
