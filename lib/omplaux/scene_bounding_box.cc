#include "scene_bounding_box.h"
#include "bounding_sphere.h"

namespace omplaux {

class BoundingSphereMerger {
public:
	BoundingSphereMerger(const Eigen::VectorXd& center, double radius)
		:center_(center), radius_(radius)
	{
	}

	void merge(const Eigen::VectorXd& center, double radius)
	{
		Eigen::VectorXd old2new = (center - center_).normalized();
		Eigen::VectorXd endnew = center + radius * old2new;
		Eigen::VectorXd endold = center_ - radius_ * old2new;
		center_ = (endnew + endold) / 2.0;
		radius_ = (endnew - endold).norm();
	}

	Eigen::VectorXd getCenter() const { return center_; }
	double getRadius() const { return radius_; }
private:
	Eigen::VectorXd center_;
	double radius_ = 0.0;
};

void calculateSceneBoundingBox(const Geo& robot,
				   const Geo& env,
				   const Path& path,
				   double& bbmin,
				   double& bbmax)
{
	Eigen::VectorXd robot_bs_center, env_bs_center;
	double robot_bs_radius, env_bs_radius;
	getBoundingSphere(robot.V, robot_bs_center, robot_bs_radius);
	getBoundingSphere(env.V, env_bs_center, env_bs_radius);

	BoundingSphereMerger bsm(path.T.front(), robot_bs_radius);;
	bsm.merge(env_bs_center, env_bs_radius);
	bsm.merge(path.T.back(), robot_bs_radius);
	bbmin = bsm.getCenter().minCoeff() - bsm.getRadius();
	bbmax = bsm.getCenter().maxCoeff() + bsm.getRadius();
}

}
