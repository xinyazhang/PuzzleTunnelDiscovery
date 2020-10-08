/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
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
	// TODO: fix robot.center for bs_center and calculate radius
	// accordingly.
	getBoundingSphere(robot.V, robot_bs_center, robot_bs_radius);
	robot_bs_center -= robot.center;
	std::cerr << "Robot bounding sphere: " << robot_bs_center.transpose()
	          << "\traidus: " << robot_bs_radius << std::endl;
	getBoundingSphere(env.V, env_bs_center, env_bs_radius);
	std::cerr << "Env bounding sphere: " << env_bs_center.transpose()
	          << "\traidus: " << env_bs_radius << std::endl;

	std::cerr << "Init trans: " << (path.T.front() - robot.center).transpose() << std::endl;
	BoundingSphereMerger bsm(path.T.front() - robot.center, robot_bs_radius);
	std::cerr << "Basic BSM sphere: " << bsm.getCenter().transpose()
	          << "\traidus: " << bsm.getRadius() << std::endl;
	bsm.merge(env_bs_center, env_bs_radius + robot_bs_radius);
	std::cerr << "With Env BSM sphere: " << bsm.getCenter().transpose()
	          << "\traidus: " << bsm.getRadius() << std::endl;
	bsm.merge(path.T.back() - robot.center, robot_bs_radius);
	std::cerr << "Final BSM sphere: " << bsm.getCenter().transpose()
	          << "\traidus: " << bsm.getRadius() << std::endl;
	bbmin = bsm.getCenter().minCoeff() - bsm.getRadius();
	bbmax = bsm.getCenter().maxCoeff() + bsm.getRadius();
}

}
