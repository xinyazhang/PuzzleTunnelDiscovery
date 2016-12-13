#ifndef OMPLAUX_GEO_H
#define OMPLAUX_GEO_H

#include <Eigen/Core>

struct Geo {
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> GPUV;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N;

	Eigen::Vector3d center;

	void read(const std::string& fn);
};

#endif
