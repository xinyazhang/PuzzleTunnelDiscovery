#ifndef OMPLAUX_CONVEX_H
#define OMPLAUX_CONVEX_H

#include <fcl/geometry/shape/convex.h>
#include <Eigen/Core>
#include <memory>

namespace omplaux {
class ConvexAdapter {
public:
	ConvexAdapter();
	~ConvexAdapter();

	ConvexAdapter(const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F);
	static void adapt(const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		ConvexAdapter& cvx);
	const fcl::Convex<double>& getFCL() const;
	fcl::Convex<double>& getFCL();
private:
	struct Private;
	std::shared_ptr<Private> p_; // dirty but works for std::vector.resize
};
};

#endif
