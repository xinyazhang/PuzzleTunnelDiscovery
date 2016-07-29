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
// [ret] base: the base of vertex index, tetgen doesn't specifiy it must be
// zero
//
// Throws std::runtime_error
//
void readtet(const std::string& prefix,
	     Eigen::MatrixXd& V,
	     Eigen::MatrixXi& E,
	     Eigen::MatrixXi& P);

int readtet(const std::string& prefix,
	     Eigen::MatrixXd& V,
	     Eigen::MatrixXi& P);


#endif
