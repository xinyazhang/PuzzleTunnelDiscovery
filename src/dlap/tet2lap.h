#ifndef TET2LAP_H
#define TET2LAP_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <readvoronoi.h>

void tet2lap(const Eigen::MatrixXd& V,
	     const Eigen::MatrixXi& E,
	     const Eigen::MatrixXi& P,
#if 0 
	     const Eigen::MatrixXd& VNodes,
	     const std::vector<VoronoiEdge>& VEdges,
	     const std::vector<VoronoiFace>& VFaces,
	     const std::vector<VoronoiCell>& VCells,
#endif
	     Eigen::SparseMatrix<double>& lap,
	     bool unit_weight = false
	     );

#endif
