#ifndef READTET_H
#define READTET_H

#include <Eigen/Core>
#include <string>

// = readtet
//
// [out] V: vertices
// [out] E: edges
// [out] P: simplex primitives
// [in] prefix: the common prefix of tetgen .node and .ele files.
//
// Throws std::runtime_error
void readtet(Eigen::MatrixXd& V,
	     Eigen::MatrixXi& E,
	     Eigen::MatrixXi& P,
	     const std::string& prefix,
	     const std::string& ofn);

#endif
