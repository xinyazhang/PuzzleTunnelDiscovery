#ifndef EIGEN_CCD_H
#define EIGEN_CCD_H

#include <string>
#include <memory>
#include <ccd/ccd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

class EigenCCD {
public:
	using State = Eigen::Matrix<double, 6, 1>;
	using Transform3 = Eigen::Transform<double, 3, Eigen::AffineCompact>;

	virtual void support(const ccd_vec3_t *dir, ccd_vec3_t *vec) const = 0;
	virtual void center(ccd_vec3_t *center) const = 0;

	virtual void setTransform(const State&, const Eigen::Vector3d& rot_center = Eigen::Vector3d::Zero()) = 0;
	virtual void setTransform(const Transform3&) = 0;
	static std::unique_ptr<EigenCCD> create(const std::string& fn);
	static std::unique_ptr<EigenCCD> create(
			Eigen::MatrixXd&& V,
			Eigen::MatrixXi&& F,
			const Eigen::Vector3d* pgeocenter = nullptr
			);
	static std::unique_ptr<EigenCCD> create(
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			const Eigen::Vector3d* pgeocenter = nullptr
			);

	struct PenetrationInfo {
		double depth;
		Eigen::Vector3d dir;
		Eigen::Vector3d pos;
	};

	static bool penetrate(const EigenCCD*, const EigenCCD*, PenetrationInfo&);
};

#endif
