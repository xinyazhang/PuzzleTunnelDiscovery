#ifndef OMPLAUX_BOUNDING_SPHERE_H
#define OMPLAUX_BOUNDING_SPHERE_H

#include <Eigen/Core>

namespace omplaux {
void getBoundingSphere(const Eigen::MatrixXd& V,
                       Eigen::VectorXd& center,
                       double& radius);
}

#endif
