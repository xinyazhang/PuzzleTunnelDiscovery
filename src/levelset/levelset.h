#ifndef LEVELSET_GRID_H
#define LEVELSET_GRID_H

#include <Eigen/Core>
#include <string>

namespace levelset {

	void generate(
			const Eigen::MatrixXf& V,
			const Eigen::MatrixXi& F,
			double mtov_width,
			double vtom_width,
			const std::string& fn
		     );

};

#endif
