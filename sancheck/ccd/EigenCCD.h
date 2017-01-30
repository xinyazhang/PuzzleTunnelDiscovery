#ifndef EIGEN_CCD_H
#define EIGEN_CCD_H

#include <string>
#include <memory>
#include <ccd/ccd.h>
#include <Eigen/Core>

class EigenCCD {
public:
	using State = Eigen::Matrix<double, 6, 1>;

	virtual void support(const ccd_vec3_t *dir, ccd_vec3_t *vec) = 0;
	virtual void center(ccd_vec3_t *center) = 0;

	virtual void setTransform(const State&, const Eigen::Vector3d& rot_center = Eigen::Vector3d::Zero()) = 0;
	static std::unique_ptr<EigenCCD> create(const std::string& fn);
};

#endif
