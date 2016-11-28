#ifndef OMPLAUX_CLEARANCE_H
#define OMPLAUX_CLEARANCE_H

#include <fcl/fcl.h> // This incldued eigen as well.
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/narrowphase/distance_result.h>
#include <string>
#include <random>
#include <iostream>
#include <math.h>

struct Geo {
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> GPUV;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N;

	Eigen::Vector3d center;

	void read(const std::string& fn)
	{
		igl::readOBJ(fn, V, F);
		//center << 0.0, 0.0, 0.0; // Origin
		center = V.colwise().mean().cast<double>();
		//center << 16.973146438598633, 1.2278236150741577, 10.204807281494141;
		// From OMPL.app, no idea how they get this.
		GPUV = V.cast<float>();
		igl::per_vertex_normals(GPUV, F, N);
		std::cerr << "center: " << center << std::endl;
#if 0
		std::cerr << N << std::endl;;
#endif
	}
};

template<typename BV>
class ClearanceCalculator {
private:
	const Geo &rob_, &env_;
	using Scalar = typename BV::S;
	using BVHModel = fcl::BVHModel<BV>;
	using Transform3 = fcl::Transform3<Scalar>;
	using TraversalNode = fcl::detail::MeshDistanceTraversalNodeOBBRSS<Scalar>;
	static constexpr int qsize = 2; // What's qsize?

	BVHModel rob_bvh_, env_bvh_;
	fcl::detail::SplitMethodType split_method_ = fcl::detail::SPLIT_METHOD_MEDIAN;
public:
	using TransformMatrix = Eigen::Matrix<double, 4, 4>;

	ClearanceCalculator(const Geo& rob, Geo& env)
		:rob_(rob), env_(env), distribution_(-1.0, 1.0)
	{
		buildBVHs();
	}

	double getDistance(const TransformMatrix& trmat) const
	{
		fcl::DistanceResult<Scalar> result;
		TraversalNode node;
		Transform3 tf;
		tf = trmat.block<3,4>(0,0);
		fcl::DistanceRequest<Scalar> request(true);
		if(!fcl::detail::initialize(node,
		                    rob_bvh_, tf,
		                    env_bvh_, Transform3::Identity(),
		                    request,
				    result)
		  ) {
			std::cerr << "initialize error" << std::endl;
		}
#if 1
		fcl::detail::distance(&node, nullptr, qsize);
#endif
		if (result.min_distance > 0)
			return result.min_distance;
		return getPenetrationDepth(tf);
	}

	double getPenetrationDepth(const Transform3& tf) const
	{
		std::cerr << __func__ << " called" << std::endl;
		Transform3 tf2{Transform3::Identity()};

		fcl::DistanceRequest<Scalar> request;
		request.enable_signed_distance = true;
		request.enable_nearest_points = true;
		request.gjk_solver_type = fcl::GST_LIBCCD;

		fcl::DistanceResult<Scalar> result;
		fcl::distance(&rob_bvh_, tf, &env_bvh_, Transform3::Identity(), request, result);
		return result.min_distance;
	}

	// Wait, what's this...
	void setC(double min, double max)
	{
		min_tr_ = min;
		max_tr_ = max;
		csize_ << max_tr_ - min_tr_, 2 * M_PI;
	}

	Eigen::Vector2d getCSize() const
	{
		return csize_;
	}

	Eigen::VectorXd getClearanceCube(const TransformMatrix& trmat, double distance = -1) const
	{
		Transform3 tf;
		tf = trmat.block<3,4>(0,0);
		Eigen::Vector3d nrcenter = tf * rob_.center;

		if (distance < 0)
			distance = getDistance(trmat);

		const Eigen::MatrixXd& RV = rob_.V;
		double dscale = 1.0;
		auto csize = getCSize();
		for (int i = 0; i < RV.rows(); i++) {
			// v: relative coordinates w.r.t. robot center.
			Eigen::Vector3d v = tf * Eigen::Vector3d(RV.row(i)) - nrcenter;
			double r = v.norm();
			dscale = std::min(dscale, binsolve(csize.x(), csize.y(), r, distance));
		}
		Eigen::VectorXd ret;
#if 0
		ret << 2 * M_PI, 2 * M_PI, 2 * M_PI,
		       2 * M_PI, 2 * M_PI, 2 * M_PI;
#endif
		ret.resize(8);
		ret << csize.x() * dscale, csize.x() * dscale, csize.x() * dscale,
		       csize.y() * dscale, csize.y() * dscale, csize.y() * dscale,
		       dscale, std::log2(1.0/dscale);
		return ret;
	}

	Eigen::VectorXd getSolidCube(const TransformMatrix& trmat, double pendepth = NAN) const
	{
		if (isnan(pendepth))
			pendepth = -getDistance(trmat);
		return getClearanceCube(trmat, pendepth); // The bounding formula is the same.
	}

	int sanityCheck(const TransformMatrix& trmat, const Eigen::VectorXd& clearance)
	{
		double dx = clearance(0);
		double dalpha = clearance(3);
		constexpr int nsample = 100;
		Eigen::VectorXd nfailed;
		nfailed.resize(nsample);
		nfailed.setZero();
		for (int i = 0; i < nsample; i++) {
			TransformMatrix randtr = randomTransform(dx, dalpha);
			TransformMatrix tr = randtr * trmat;
			double dist = getDistance(tr);
#if 0
			std::cerr << "\t\t\tSanity distance: " << dist << std::endl;
			std::cerr << "\t\t\tSanity randtr matrix: " << randtr << std::endl;
			std::cerr << "\t\t\tSanity tr matrix: " << tr << std::endl;
#endif
			if (dist <= 0) {
				nfailed(i) = 1;
			}
		}
		return nfailed.array().sum();
	}

	double neg1pos1()
	{
		return distribution_(generator_);
	}

	TransformMatrix randomTransform(double dx, double dalpha)
	{
		// FIXME: use Path::stateToMatrix
		Transform3 tr;
		tr.setIdentity();
		//std::cerr << "\t\t\tInitial Sanity random rotation matrix: " << tr.matrix() << std::endl;
		tr.rotate(Eigen::AngleAxisd(neg1pos1()*dalpha, Eigen::Vector3d::UnitX()));
		tr.rotate(Eigen::AngleAxisd(neg1pos1()*dalpha, Eigen::Vector3d::UnitY()));
		tr.rotate(Eigen::AngleAxisd(neg1pos1()*dalpha, Eigen::Vector3d::UnitZ()));
#if 1
		Eigen::Vector3d vec;
		vec << neg1pos1() * dx, neg1pos1() * dx, neg1pos1() * dx; 
		tr.translate(vec);
#endif
		TransformMatrix ret;
		ret.setIdentity();
		//std::cerr << "\t\t\tSanity random rotation matrix: " << tr.matrix() << std::endl;
		ret.block<3, 4>(0, 0) = tr.matrix();
		return ret;
	}
	
protected:
	void buildBVHs()
	{
		initBVH(rob_bvh_, split_method_, rob_);
		initBVH(env_bvh_, split_method_, env_);
	}

	static void initBVH(fcl::BVHModel<BV> &bvh, fcl::detail::SplitMethodType split_method, const Geo& geo)
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
			double value = std::sin(dalpha/2) * 6 * r + sqrt3 * dx;
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

	double min_tr_, max_tr_;
	Eigen::Vector2d csize_; // translation size, rotation size
	std::mt19937 generator_;
	std::uniform_real_distribution<double> distribution_;
};
#endif
