#ifndef GEOPICK_PICK2D_H
#define GEOPICK_PICK2D_H

#include <Eigen/Core>
#include <vector>
#include <functional>

void geopick(const Eigen::MatrixXd& V,
	     std::vector<std::reference_wrapper<Eigen::MatrixXi>> Fs, // Copy matrix is way too expensive
             Eigen::MatrixXd &prdcV,   // PeRioDiCal Vertices
             Eigen::MatrixXi &prdcF);  // PeRioDiCal Faces

#endif
