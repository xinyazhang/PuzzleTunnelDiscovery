#ifndef VERT_COLOR_PLY_WRITE_PLY_VC_H
#define VERT_COLOR_PLY_WRITE_PLY_VC_H

#include <Eigen/Core>
#include <string>

void write_ply_vc(const std::string& fn,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& C);

#endif
