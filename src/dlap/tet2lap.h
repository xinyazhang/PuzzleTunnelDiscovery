#ifndef TET2LAP_H
#define TET2LAP_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

void tet2lap(const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P,
	     Eigen::SparseMatrix<double>& lap
	     );

#endif
